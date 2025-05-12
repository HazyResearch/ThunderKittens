import torch
import mlp

up = torch.randn(8192, 2048, dtype=torch.bfloat16, device=0) / 2048**.25
down = torch.randn(2048, 8192, dtype=torch.bfloat16, device=0) / 8192**.25

vec = torch.randn(2048, dtype=torch.bfloat16, device=0) / 2048**.5

ref = torch.relu(vec @ up.T) @ down.T

def make_schedule():
    instructions = []
    for i in range(8192//16):
        up_instruction = [1, 1, i] + [0]*29
        instructions.append(up_instruction)
    for i in range(2048//16):
        for j in range(4):
            down_instruction = [2, i, i+1, j] + [0]*28
            instructions.append(down_instruction)
    while len(instructions) % 132 != 0:
        instructions.append([0] * 32)
    instructions = torch.tensor(instructions, dtype=torch.int32, device=0).reshape(132,-1,32)
    timings = torch.zeros((instructions.shape[0], instructions.shape[1], 128), device=0, dtype=torch.int32)
    barriers = torch.zeros(4, device=0, dtype=torch.int32)
    return instructions, timings, barriers

instructions, timings, barriers = make_schedule()
print(instructions.shape)

inputs = vec.to(torch.float32)
hidden = torch.zeros(8192, device=0, dtype=torch.float32)
outputs = torch.zeros(2048, device=0, dtype=torch.float32)

mlp.mlp(barriers, instructions, timings, up, down, inputs, hidden, outputs)

print(ref.shape)
print(ref)
print(outputs)