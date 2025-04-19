import torch

def make_instructions():
    arr = [[] for _ in range(148)]

    instruction_idx = 0
    for i in range(128): # ceil
        arr[instruction_idx%148].append([2, 0, 16*i]+[0]*29)
        instruction_idx += 1
    while instruction_idx%148 != 0:
        arr[instruction_idx%148].append([0]*32)
        instruction_idx += 1

    instructions = torch.tensor(arr, dtype=torch.int32)
    timings = torch.zeros((148, instruction_idx//148, 128), dtype=torch.int32)

    return instructions.to(0), timings.to(0)