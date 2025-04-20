import torch
from matvec import matvec
from make_instructions import make_instructions as make_instructions
import sys
import time

torch.manual_seed(1)

print("Starting test...")

DEPTH = 2

# Create input and output tensors
W = (torch.randn((2048, 2048), device=0, dtype=torch.float32) / 2048**.25).to(torch.bfloat16)
A = (torch.randn((1, 2048), device=0, dtype=torch.float32) / 2048**.25).to(torch.bfloat16)
O =  torch.zeros((1, 2048), device=0, dtype=torch.bfloat16)  # noqa: E741
Bar = torch.ones((1, 6, 32), device=0, dtype=torch.int32)

print("Input tensors created, of shapes", A.shape, W.shape, O.shape, Bar.shape)

sys.stdout.flush()

# Create instruction and timing tensors
instructions, timings = make_instructions(DEPTH)

print(f"Instruction and timing tensors created, of shapes {instructions.shape} and {timings.shape}")

# Run the matvec kernel
matvec(instructions, timings, W, A, O, Bar)

print("Kernel launched")

torch.cuda.synchronize()


print('Starting timing loop...')

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for i in range(5):
    matvec(instructions, timings, W, A, O, Bar)
torch.cuda.synchronize()
end_event.record()
torch.cuda.synchronize()
elapsed_time = start_event.elapsed_time(end_event)
t1 = time.time()  # Keep this for compatibility with the time_per_iter calculation
t0 = t1 - (elapsed_time / 1000)  # Convert ms to seconds
time_per_iter = ((t1-t0)*1e6)/5
print(f'Time per iter: {time_per_iter} us')
print(f'GB/s: {(2*2048*2048*1e-9)/(time_per_iter*1e-6)}')


print("Test completed successfully!")

O = O.to(torch.float32).cpu().numpy()
print(O.shape)
print(O)

O2 = (A@W.T).to(torch.float32).cpu().numpy()
print(O2.shape)
print(O2)

print('TIMINGS')
for i in range(128):
    print(f'event {i:3d}: {", ".join([f"{timings[0,j,i]:6d}" for j in range(DEPTH)])}')

# Create histogram of differences
import matplotlib.pyplot as plt
differences = (O - O2).flatten()
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
