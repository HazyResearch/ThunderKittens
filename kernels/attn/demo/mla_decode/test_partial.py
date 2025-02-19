import torch
import mla_decode

def main():
    device = torch.device("cuda")
    
    #---------------------------------------------------------------------------
    # Kernel Constants (must match the C++ code)
    QK_D = 576
    VO_D = 512
    NUM_ROWS = 32
    PAGE_SIZE = 256
    ITERS_PER_PAGE = PAGE_SIZE // NUM_ROWS  # 8

    # Define sizes for the global views:
    batch = 1            # B in Q and O
    seq_len = 32         # R in Q and O
    grid = 132           # as set in config::globals::grid()
    num_task_iters = 2   # task iterations per block (partial then reduction)
    num_pages = 1        # for Cache
    max_uid = 1024       # maximum uid index for scratch buffers

    #---------------------------------------------------------------------------
    # Global Buffers (using 4-D tensors as required by our gl<> types)
    #
    # instructions_global: gl<int, 1, -1, -1, 8>
    #   Expected external shape: (1, grid, num_task_iters, 8)
    instructions = torch.zeros((1, grid, num_task_iters, 8), dtype=torch.int32, device=device)
    
    # q_global: gl<bf16, -1, -1, -1, QK_D>
    #   Expected external shape: (B, R, H, QK_D); we use H = 1.
    q = torch.randn((batch, seq_len, 1, QK_D), dtype=torch.bfloat16, device=device)
    
    # cache_global: gl<bf16, 1, -1, PAGE_SIZE, QK_D>
    #   Expected external shape: (1, num_pages, PAGE_SIZE, QK_D)
    cache = torch.randn((1, num_pages, PAGE_SIZE, QK_D), dtype=torch.bfloat16, device=device)
    
    # table_global: gl<int, 1, 1, -1, -1>
    #   Expected external shape: (1, 1, max_uid, 1)
    table = torch.zeros((1, 1, max_uid, 1), dtype=torch.int32, device=device)
    
    # o_global: gl<bf16, -1, -1, -1, VO_D>
    #   Expected external shape: (B, R, H, VO_D)
    O = torch.empty((batch, seq_len, 1, VO_D), dtype=torch.bfloat16, device=device)
    
    # o_scratch_global: gl<float, 1, -1, 64, VO_D>
    #   Expected external shape: (1, max_uid, 64, VO_D)
    O_scratch = torch.empty((1, max_uid, 64, VO_D), dtype=torch.float32, device=device)
    
    # lvec_scratch_global: gl<float, 1, 1, -1, 64>
    #   Expected external shape: (1, 1, max_uid, 64)
    Lvec_scratch = torch.empty((1, 1, max_uid, 64), dtype=torch.float32, device=device)
    
    # semaphore_global: gl<int, 1, 1, 1, -1>
    #   Expected external shape: (1, 1, 1, max_uid)
    semaphore = torch.zeros((1, 1, 1, max_uid), dtype=torch.int32, device=device)
    
    softmax_scale = 1.0

    #---------------------------------------------------------------------------
    # Set up Instructions:
    #
    # The instructions layout (last dim = 8) is interpreted as follows:
    #   For a partial op (opcode 1):
    #     [0]: opcode (1)
    #     [1]: uid (scratch index)
    #     [2]: dst.batch_idx (set to -1 for non-batch mode)
    #     [3]: dst.seq_idx (we use uid so that scratch buffers are indexed by uid)
    #     [4]: q_batch_idx
    #     [5]: q_seq_idx
    #     [6]: length (e.g. 128)
    #     [7]: unused
    #
    #   For a reduction op (opcode 2):
    #     [0]: opcode (2)
    #     [1]: dst.batch_idx (non-batch: -1)
    #     [2]: dst.seq_idx (destination index; here 0)
    #     [3]: src_uid[0]
    #     [4]: src_uid[1]
    #     [5]-[7]: unused
    #
    # --- Partial op 1 (uid 100) ---
    instructions[0, 0, 0, 0] = 1      # opcode for partial op
    instructions[0, 0, 0, 1] = 100    # uid for this op
    instructions[0, 0, 0, 2] = -1     # dst.batch_idx (non-batch mode)
    instructions[0, 0, 0, 3] = 100    # dst.seq_idx (using uid for scratch index)
    instructions[0, 0, 0, 4] = 0      # q_batch_idx
    instructions[0, 0, 0, 5] = 0      # q_seq_idx
    instructions[0, 0, 0, 6] = 128    # length (example value)
    instructions[0, 0, 0, 7] = 0

    # --- Partial op 2 (uid 101) ---
    instructions[0, 1, 0, 0] = 1      # opcode for partial op
    instructions[0, 1, 0, 1] = 101    # uid for this op
    instructions[0, 1, 0, 2] = -1     # dst.batch_idx (non-batch mode)
    instructions[0, 1, 0, 3] = 101    # dst.seq_idx (using uid)
    instructions[0, 1, 0, 4] = 0      # q_batch_idx
    instructions[0, 1, 0, 5] = 0      # q_seq_idx
    instructions[0, 1, 0, 6] = 128    # length
    instructions[0, 1, 0, 7] = 0

    # --- Reduction op ---
    # Place the reduction op in grid index 0, task_iter 1.
    instructions[0, 0, 1, 0] = 2      # opcode for reduction op
    instructions[0, 0, 1, 1] = -1     # dst.batch_idx (non-batch)
    instructions[0, 0, 1, 2] = 0      # dst.seq_idx (destination index)
    instructions[0, 0, 1, 3] = 100    # src_uid[0]
    instructions[0, 0, 1, 4] = 101    # src_uid[1]
    instructions[0, 0, 1, 5] = 0
    instructions[0, 0, 1, 6] = 0
    instructions[0, 0, 1, 7] = 0

    #---------------------------------------------------------------------------
    # Simulate Partial Op Outputs:
    #
    # In non-batch mode, the partial kernel writes its outputs to scratch buffers
    # at an index equal to its uid. (The kernel uses store() with coordinates:
    #   O_scratch: {dst.seq_idx, 0, warpgroup::groupid()}
    #   Lvec_scratch: {dst.seq_idx, 0})
    #
    # Here we “pre-fill” these scratch buffers for uid 100 and 101.
    for uid in [100, 101]:
        # For O_scratch, the dynamic (second) dimension is used to index uid.
        # We simulate a single tile (64 rows) filled with a constant value.
        O_scratch[0, uid, :, :].fill_(float(uid))
        # For Lvec_scratch, the dynamic (third) dimension is used to index uid.
        Lvec_scratch[0, 0, uid, :].fill_(float(uid))
    
    # Also, set the semaphores for uid 100 and 101 so the reduction op does not wait.
    semaphore[0, 0, 0, 100] = 1
    semaphore[0, 0, 0, 101] = 1

    #---------------------------------------------------------------------------
    # Launch the Kernel.
    mla_decode.mla_decode(
        instructions, q, cache, table, O, O_scratch, Lvec_scratch, semaphore, softmax_scale
    )
    
    #---------------------------------------------------------------------------
    # For debugging, print slices from the scratch buffers.
    print("O_scratch for uid 100:")
    print(O_scratch[0, 100])
    print("\nLvec_scratch for uid 100:")
    print(Lvec_scratch[0, 0, 100])
    
if __name__ == '__main__':
    main()
