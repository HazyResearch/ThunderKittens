import torch

# def make_instructions(M, K, N):
#     if M%256 != 0: raise ValueError("M must be divisible by 256")
#     if K%128 != 0: raise ValueError("K must be divisible by 128")
#     if N%256 != 0: raise ValueError("N must be divisible by 256")

#     arr = [[] for _ in range(148)]

#     SUPER_M = 3072

#     instruction_idx = 0
#     for i in range((M+SUPER_M-1)//SUPER_M): # ceil
#         for col in range(N//256):
#             for k in range(SUPER_M//256):
#                 row = (SUPER_M//256)*i + k
#                 if row >= M//256:
#                     break
#                 arr[instruction_idx%148].append([1, 2*row, 2*col, K//128]+[0]*28)
#                 instruction_idx += 1
#     while instruction_idx%148 != 0:
#         arr[instruction_idx%148].append([0]*32)
#         instruction_idx += 1

#     instructions = torch.tensor(arr, dtype=torch.int32)
#     timings = torch.zeros((148, instruction_idx//148, 128), dtype=torch.int32)

#     return instructions.to(0), timings.to(0)

def make_instructions(M, K, N):
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128

    if M % BLOCK_M != 0: raise ValueError(f"M ({M}) must be divisible by {BLOCK_M}")
    if K % BLOCK_K != 0: raise ValueError(f"K ({K}) must be divisible by {BLOCK_K}")
    if N % BLOCK_N != 0: raise ValueError(f"N ({N}) must be divisible by {BLOCK_N}")

    arr = [[] for _ in range(148)]
    SUPER_M = 3072

    instruction_idx = 0
    for i in range((M + SUPER_M - 1) // SUPER_M):
        for col in range(N // BLOCK_N):
            for k in range(SUPER_M // BLOCK_M):
                row = (SUPER_M // BLOCK_M) * i + k
                if row >= M // BLOCK_M:
                    break

                opcode = 1
                row_offset_inst = 2 * row
                col_offset_inst = 2 * col
                k_iters = K // BLOCK_K

                instruction = [opcode, row_offset_inst, col_offset_inst, k_iters] + [0] * 28
                arr[instruction_idx % 148].append(instruction)
                instruction_idx += 1

    while instruction_idx % 148 != 0:
        arr[instruction_idx % 148].append([0] * 32)
        instruction_idx += 1

    if instruction_idx == 0:
         print("Warning: No instructions generated.")
         instructions = torch.zeros((148, 0, 32), dtype=torch.int32)
         timings = torch.zeros((148, 0, 128), dtype=torch.int32)
         return instructions.to(0), timings.to(0)


    num_instructions_per_sm = instruction_idx // 148
    instructions_tensor = torch.tensor(arr, dtype=torch.int32).view(148, num_instructions_per_sm, 32)
    timings = torch.zeros((148, num_instructions_per_sm, 128), dtype=torch.int32)

    return instructions_tensor.to(0), timings.to(0)


def make_instructions_1sm(M, K, N):
    if M%256 != 0: raise ValueError("M must be divisible by 256")
    if K%128 != 0: raise ValueError("K must be divisible by 128")
    if N%256 != 0: raise ValueError("N must be divisible by 256")

    arr = [[] for _ in range(148)]

    SUPER_M = 3072

    instruction_idx = 0
    for i in range((M+SUPER_M-1)//SUPER_M): # ceil
        for col in range(N//256):
            for k in range(SUPER_M//256):
                row = (SUPER_M//256)*i + k
                if row >= M//256:
                    break
                arr[0].append([1, 2*row, 2*col, K//128]+[0]*28)
                for j in range(1, 148):
                    arr[j].append([0]*32)
                instruction_idx += 1
    instructions = torch.tensor(arr, dtype=torch.int32)
    timings = torch.zeros((148, instruction_idx, 128), dtype=torch.int32)

    return instructions.to(0), timings.to(0)