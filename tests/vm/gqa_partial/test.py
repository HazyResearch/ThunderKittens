import torch
# from gqa_partial import gqa_partial
from matmul import matmul as gqa_partial
from kvm import make_globals, make_instruction, partial_attention


NUM_BLOCKS = 148
INSTRUCTION_WIDTH = 32
GQA_PARTIAL_OPCODE = 1


##### PYTORCH #####

# # Generate inputs
# globals = make_globals()
# instructions = make_instruction(globals)

# # Run single instruction
# instruction = instructions[0]
# partial_attention(globals, instruction)

# # Print out results
# print('globals.attn_lse_intermediates.shape:', globals.attn_lse_intermediates.shape)
# print('globals.attn_out_intermediates.shape:', globals.attn_out_intermediates.shape)
# print(globals.attn_lse_intermediates)
# print(globals.attn_out_intermediates)


# ##### KITTENS VM #####

# # Generate inputs
# globals = make_globals()
# instructions = make_instruction(globals)

# # Change instruction format
# # (BlockIdx, InstructionIdx, Instruction)
# instruction = instructions[0]
# instructions = [[[instruction.opcode(), instruction.layer_idx, instruction.kv_head_idx, instruction.num_partials, instruction.partial_idx] + [0] * (INSTRUCTION_WIDTH - 5)]]
# for i in range(NUM_BLOCKS - 1): instructions.append([[0] * INSTRUCTION_WIDTH])
# instructions = torch.tensor(instructions, dtype=torch.int32).to(device=0)
# print(instructions[0, 0])

# # Change timings format
# timings = torch.reshape(globals.timings, (-1, 128))

# # print all shapes
# print('instructions.shape:', instructions.shape)
# print('globals.timings.shape:', globals.timings.shape)
# print('globals.post_ln_rope_q.shape:', globals.post_ln_rope_q.shape)
# print('globals.k_cache.shape:', globals.k_cache.shape)
# print('globals.v_cache.shape:', globals.v_cache.shape)
# print('globals.attn_lse_intermediates.shape:', globals.attn_lse_intermediates.shape)
# print('globals.attn_out_intermediates.shape:', globals.attn_out_intermediates.shape)
# print('globals.pos_id:', globals.pos_id)
# print('globals.attn_kv_block_size:', globals.attn_kv_block_size)
# print('globals.num_attention_heads:', globals.num_attention_heads)
# print('globals.num_kv_heads:', globals.num_kv_heads)
# print('globals.softmax_temp:', globals.softmax_temp)

# torch.cuda.synchronize(0)
# instructions = torch.clone(instructions).to(device=0)
# timings = torch.clone(timings).to(device=0)

# torch.cuda.synchronize(0)
# Run single instruction
insts = torch.zeros((NUM_BLOCKS, 1, INSTRUCTION_WIDTH), dtype=torch.int32, device=0)
timings = torch.zeros((NUM_BLOCKS, 128), dtype=torch.int32, device=0)
gqa_partial(
    insts, 
    timings,
    # globals.post_ln_rope_q,
    # globals.k_cache, 
    # globals.v_cache, 
    # globals.attn_lse_intermediates,
    # globals.attn_out_intermediates,
    # globals.pos_id,
    # globals.attn_kv_block_size,
    # globals.num_attention_heads,
    # globals.num_kv_heads,
    # globals.softmax_temp,
)
torch.cuda.synchronize(0)

# Print out results
# print('globals.attn_lse_intermediates.shape:', globals.attn_lse_intermediates.shape)
# print('globals.attn_out_intermediates.shape:', globals.attn_out_intermediates.shape)
# print(globals.attn_lse_intermediates)
# print(globals.attn_out_intermediates)


# Q = torch.reshape(globals.post_ln_rope_q, (32, 64))
# K = torch.reshape(globals.k_cache, (16, 32, 8, 64))






# def generate_inputs(B: int, N: int, N_new: int, H_q: int, H_kv: int, D_h: int):

#     '''Generate inputs for the GQA kernel.
#     Args:
#         B (int): Batch size.
#         N (int): Sequence length.
#         N_new (int): Number of new tokens.
#         H_q (int): Number of query heads.
#         H_kv (int): Number of key/value heads (number of query groups).
#         D_h (int): Dimension of each head.
#     '''
#     torch.manual_seed(42)

#     # Normal random does not support fp8 yet
#     Q   = torch.randn(B, N_new, H_q, D_h, dtype=torch.bfloat16, device='cuda:0')
#     K_c = torch.randn(B, N_new, H_kv, D_h, dtype=torch.bfloat16, device='cuda:0')
#     V_c = torch.randn(B, N_new, H_kv, D_h, dtype=torch.bfloat16, device='cuda:0')
#     O   = torch.zeros(B, N_new, H_q, 32, dtype=torch.bfloat16, device='cuda:0')

#     return Q, K_c, V_c, O

# def generate_instructions_and_timings():

#     instructions = [[] for _ in range(NUM_BLOCKS)]
#     instruction_idx = 0

#     # arbitrary instruction, for testing
#     instructions[0].append([GQA_PARTIAL_OPCODE] + [0] * (INSTRUCTION_WIDTH - 1))
#     instruction_idx += 1

#     # all blocks must have same number of instructions
#     max_instructions = -1
#     for i in range(NUM_BLOCKS):
#         max_instructions = max(max_instructions, len(instructions[i]))
#     for i in range(NUM_BLOCKS):
#         while len(instructions[i]) < max_instructions:
#             instructions[i].append([0] * INSTRUCTION_WIDTH)
#         instruction_idx += 1

#     # (BlockIdx, InstructionIdx, Instruction)
#     # If opcode (instructions[:, :, 0]) is invalid, the instruction is ignored
#     instructions = torch.tensor(instructions, dtype=torch.int32).to(device=0)
#     timings = torch.zeros((NUM_BLOCKS, instruction_idx // NUM_BLOCKS, 128), dtype=torch.int32).to(device=0)

#     return instructions, timings

# # Llama 3.2 1B Model Parameters
# B = 1 # this is fixed for now!
# N_max = 131072
# N = 16 # for testing for now
# H_q = 32
# H_kv = 8
# D_h = 64

# N = 1 # for testing for now
# H_q = 32
# H_kv = 32
# D_h = 64

# print("Generating inputs and instructions...")
# Q, K_c, V_c, O = generate_inputs(B, N, N, H_q, H_kv, D_h)
# instructions, timings = generate_instructions_and_timings()


# print(torch.isnan(Q).any())
# print(torch.isnan(K_c).any())
# print(torch.isnan(V_c).any())
# print(torch.isnan(O).any())
# print(torch.sum(Q))
# print(torch.sum(K_c))
# print(torch.sum(V_c))
# print(torch.sum(O))



# # Run the matmul kernel)
# print('Q shape:', Q.shape)
# print('K_c shape:', K_c.shape)
# print('V_c shape:', V_c.shape)
# print('O shape:', O.shape)
# print('Instruction shape:', instructions.shape)
# print('Timings shape:', timings.shape) 
# print("Running the kernel...")
# gqa_partial(instructions, timings, Q, K_c, V_c, O)
# torch.cuda.synchronize(0)

# print(O)
# print(torch.sum(O))
# O_ref = torch.matmul(Q, K_c.transpose(-1, -2))
# print(O_ref)
# print(torch.sum(O_ref))
