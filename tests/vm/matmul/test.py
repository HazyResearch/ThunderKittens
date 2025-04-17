import torch
from matmul import matmul
from make_instructions import make_instructions
import sys
import time

torch.manual_seed(1)

# M, K, N = 4096, 4096, 4096
M, K, N = 16384, 16384, 16384
# M, K, N = 8192, 8192, 8192
# M, K, N = 3072, 16384*2, 3072
# M, K, N = 256, 4096, 256

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
# instructions, timings = make_instructions_1sm(M, K, N)

print(f"Instruction and timing tensors created, of shapes {instructions.shape} and {timings.shape}")

# Run the matmul kernel
matmul(instructions, timings, A, B, C)

print("Kernel launched")

torch.cuda.synchronize()


print('Starting timing loop...')

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for i in range(5):
    matmul(instructions, timings, A, B, C)
torch.cuda.synchronize()
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
t1 = time.time()  # Keep this for compatibility with the time_per_iter calculation
t0 = t1 - (elapsed_time / 1000)  # Convert ms to seconds
time_per_iter = ((t1-t0)*1e6)/5
print(f'Time per iter: {time_per_iter} us')
print(f'TFLOP/s: {(2*M*N*K*1e-12)/(time_per_iter*1e-6)}')


print("Test completed successfully!")

C = C.to(torch.float32).cpu().numpy()
print(C.shape)
print(C)

C2 = (A.to(torch.float16)@B.to(torch.float16).T).to(torch.float8_e4m3fn)
C2 = C2.to(torch.float32).cpu().numpy()
print(C2.shape)
print(C2)

print('TIMINGS')
for i in range(128):
    print(f'event {i}: {timings[0,0,i]}')

# Create histogram of differences
import matplotlib.pyplot as plt
differences = (C - C2).flatten()
plt.figure(figsize=(10, 6))
plt.hist(differences, bins=50, alpha=0.7)
plt.title('Histogram of Differences Between Matrices')
plt.xlabel('Difference Value')
plt.ylabel('Frequency')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.grid(True, alpha=0.3)
plt.savefig('matrix_diff_histogram.png')
plt.close()
print(f"Histogram saved to matrix_diff_histogram.png")
