import torch

def make_instructions(M, K, N):
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
                arr[instruction_idx%148].append([1, 2*row, 2*col, K//128]+[0]*28)
                instruction_idx += 1
    while instruction_idx%148 != 0:
        arr[instruction_idx%148].append([0]*32)
        instruction_idx += 1

    instructions = torch.tensor(arr, dtype=torch.int32)
    timings = torch.zeros((148, instruction_idx//148, 128), dtype=torch.int32)

    return instructions.to(0), timings.to(0)