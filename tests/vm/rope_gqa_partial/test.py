import torch
from rope_gqa_partial import rope_gqa_partial

# TODO: use paged attention
# TODO: variable sequence lengths

NUM_BLOCKS = 148
INSTRUCTION_WIDTH = 32
ROPE_GQA_PARTIAL_OPCODE = 2

def generate_inputs(N: int, N_new: int, H_q: int, H_kv: int, D_h: int):

    '''Generate inputs for the GQA kernel.
    Args:
        B (int): Batch size.
        N (int): Sequence length.
        N_new (int): Number of new tokens.
        H_q (int): Number of query heads.
        H_kv (int): Number of key/value heads (number of query groups).
        D_h (int): Dimension of each head.
    '''
    torch.manual_seed(42)

    # Normal random does not support fp8 yet
    Q   = torch.randn(B, N_new, H_q, D_h, dtype=torch.bfloat16, device='cuda')
    K_c = torch.randn(B, N, H_kv, D_h, dtype=torch.bfloat16, device='cuda')
    V_c = torch.randn(B, N, H_kv, D_h, dtype=torch.bfloat16, device='cuda')
    O   = torch.randn(B, N_new, H_q, D_h, dtype=torch.bfloat16, device='cuda')

    return Q, K_c, V_c, O

def generate_instructions_and_timings():

    instructions = [[] for _ in range(NUM_BLOCKS)]
    instruction_idx = 0

    # arbitrary instruction, for testing
    instructions[0].append([ROPE_GQA_PARTIAL_OPCODE] + [0] * (INSTRUCTION_WIDTH - 1))
    instruction_idx += 1

    while instruction_idx % NUM_BLOCKS != 0:
        instructions[instruction_idx % NUM_BLOCKS].append([0] * INSTRUCTION_WIDTH)
        instruction_idx += 1

    # (BlockIdx, InstructionIdx, Instruction)
    instructions = torch.tensor(instructions, dtype=torch.int32).to(device=0)
    timings = torch.zeros((NUM_BLOCKS, instruction_idx // NUM_BLOCKS, 128), dtype=torch.int32).to(device=0)

    return instructions, timings

# Llama 3.2 1B Model Parameters
B = 1 # this is fixed for now!
N_max = 131072
N = 16 # for testing for now
H_q = 32
H_kv = 8
D_h = 64

print("Generating inputs and instructions...")
Q, K_c, V_c, O = generate_inputs(B, N, H_q, H_kv, D_h)
instructions, timings = generate_instructions_and_timings()

# Run the matmul kernel)
print("Running the kernel...")
rope_gqa_partial(instructions, timings, Q, K_c, V_c, O)
torch.cuda.synchronize()
