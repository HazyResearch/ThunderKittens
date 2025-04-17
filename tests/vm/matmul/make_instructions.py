import torch

def make_instructions(M, K, N):
    if M%256 != 0: raise ValueError("M must be divisible by 256")
    if K%128 != 0: raise ValueError("K must be divisible by 128")
    if N%256 != 0: raise ValueError("N must be divisible by 256")

    arr = [[] for _ in range(148)]

    instruction_idx = 0
    for i in range(M//256):
        for j in range(N//256):
            arr[instruction_idx%148].append([1, 2*i, 2*j, K//128]+[0]*28)
            instruction_idx += 1
    while instruction_idx%148 != 0:
        arr[instruction_idx%148].append([0]*32)
        instruction_idx += 1

    instructions = torch.tensor(arr, dtype=torch.int32)
    timings = torch.zeros((148, instruction_idx//148, 128), dtype=torch.int32)

    return instructions.to(0), timings.to(0)

def make_instructions_1sm(M, K, N):
    if M%256 != 0: raise ValueError("M must be divisible by 256")
    if K%128 != 0: raise ValueError("K must be divisible by 128")
    if N%256 != 0: raise ValueError("N must be divisible by 256")

    arr = [[] for _ in range(148)]

    instruction_idx = 0
    for i in range(M//256):
        for j in range(N//256):
            arr[0].append([1, 2*i, 2*j, K//128]+[0]*28)
            instruction_idx += 1
            for k in range(1, 148):
                arr[k].append([0]*32)

    instructions = torch.tensor(arr, dtype=torch.int32)
    timings = torch.zeros((148, instruction_idx, 128), dtype=torch.int32)

    return instructions.to(0), timings.to(0)