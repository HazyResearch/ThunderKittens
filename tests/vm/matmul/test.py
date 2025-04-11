print("Importing...")

import torch
from matmul import matmul
from make_instructions import make_instructions
import sys

torch.manual_seed(1)

M, K, N = 256, 1024, 256
# M, K, N = 2048, 256, 2048

print("Starting test...")

# Create input and output tensors
# A = (torch.ones((M, K), device=0, dtype=torch.float32) / K**.5).to(torch.float8_e4m3fn)
# B = (torch.ones((N, K), device=0, dtype=torch.float32) / K**.5).to(torch.float8_e4m3fn)
A = (torch.randn((M, K), device=0, dtype=torch.float32) / K**.25).to(torch.float8_e4m3fn)
B = (torch.randn((N, K), device=0, dtype=torch.float32) / K**.25).to(torch.float8_e4m3fn)
# B[:64,:64] = (torch.randn((64, 64), device=0, dtype=torch.float32) / K**.25).to(torch.float8_e4m3fn)
C =  torch.zeros((M, N), device=0, dtype=torch.float8_e4m3fn)

print("Input tensors created")

sys.stdout.flush()

# Create instruction and timing tensors
instructions, timings = make_instructions(M, K, N)

print("Instruction and timing tensors created")

# Run the matmul kernel
matmul(instructions, timings, A, B, C)

print("Kernel launched")

torch.cuda.synchronize()

print("Test completed successfully!")

C = C.to(torch.float32).cpu().numpy()
print(C.shape)
print(C)

C2 = (A.to(torch.float16)@B.to(torch.float16).T).to(torch.float8_e4m3fn)
C2 = C2.to(torch.float32).cpu().numpy()
print(C2.shape)
print(C2)