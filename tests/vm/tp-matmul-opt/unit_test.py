import torch
from matmul import matmul, can_access_peer, enable_p2p_access
from time import time


###
#   Global Parameters
###
NUM_DEVICES = 8
NUM_ITERS = 5
OPCODE = 725
# M, K, N = 3072, 4096, 3072
# M, K, N = 512, 256, 256
M, K, N = 16384, 3072, 16384
# M, K, N = 3072, 16384*2, 3072
# M, K, N = 256, 4096, 256

if M%256 != 0: raise ValueError("M must be divisible by 256")
if K%128 != 0: raise ValueError("K must be divisible by 128")
if N%256 != 0: raise ValueError("N must be divisible by 256")
if M%NUM_DEVICES != 0: raise ValueError("M must be divisible by NUM_DEVICES")
if N%NUM_DEVICES != 0: raise ValueError("N must be divisible by NUM_DEVICES")
if M%(NUM_DEVICES*256) != 0: raise ValueError("For now, M must be divisible by (NUM_DEVICES*256))")
if N%(NUM_DEVICES*256) != 0: raise ValueError("For now, N must be divisible by (NUM_DEVICES*256))")


###
#   Prepare Inputs
###
print(f'Starting test with M={M}, K={K}, N={N}')
print('\nGenerating inputs...')
torch.manual_seed(42)
dev_ids = [i for i in range(NUM_DEVICES)]
torch_devices = [torch.device(f"cuda:{dev_id}") for dev_id in dev_ids]
A = (torch.randn((M, K), device='cpu', dtype=torch.float32) / K**.25).to(dtype=torch.float8_e4m3fn)
B = (torch.randn((N, K), device='cpu', dtype=torch.float32) / K**.25).to(dtype=torch.float8_e4m3fn)
C =  torch.zeros((M, N), device='cpu', dtype=torch.float8_e4m3fn)

# Shard the inputs
A0s = [tensor.to(torch_devices[i]) for i, tensor in enumerate(A.chunk(len(torch_devices), dim=0))]
A1s = [torch.zeros_like(A0s[i]) for i in range(len(A0s))]
Bs = [tensor.to(torch_devices[i]) for i, tensor in enumerate(B.chunk(len(torch_devices), dim=0))] # B is transposed in the kernel
Cs = [tensor.to(torch_devices[i]) for i, tensor in enumerate(C.chunk(len(torch_devices), dim=1))]


###
#   Prepare Instructions, Timings, and Barriers
###
print('\nGenerating instructions and timings...')
instructions = []
barriers = []
timings = []
num_rows = M // 256
num_local_cols = (N // NUM_DEVICES) // 256
num_iters = K // 128
rows_per_dev = num_rows // NUM_DEVICES

for torch_device in torch_devices:

    # Instructions
    dev_instructions = [[] for _ in range(148)]
    instruction_idx = 0
    row_start = torch_device.index * rows_per_dev
    dev_idx = torch_device.index
    next_dev_idx = (dev_idx + 1) % NUM_DEVICES
    for _row in range(num_rows):
        row = (row_start + _row) % num_rows
        local_row = row % rows_per_dev
        phasebit = (_row // rows_per_dev) % 2
        for local_col in range(num_local_cols):
            dev_instructions[instruction_idx%148].append(
                [OPCODE, 2*row, 2*local_row, 2*local_col, num_iters, dev_idx, next_dev_idx, phasebit] + [0]*24
            )
            instruction_idx += 1
    while instruction_idx%148 != 0:
        dev_instructions[instruction_idx%148].append([0]*32)
        instruction_idx += 1
    dev_instructions = torch.tensor(dev_instructions, dtype=torch.int32, device=torch_device)
    instructions.append(dev_instructions)

    # Barriers
    _barriers = torch.ones((2*num_rows, num_iters), dtype=torch.uint32, device=torch_device)
    # _barriers[:, :] = 1
    barriers.append(_barriers)

    # Timings
    timings.append(torch.zeros((148, instruction_idx // 148, 128), dtype=torch.int32, device=torch_device))

print(f'Instructions shape: {instructions[0].shape}')
print(f'Barriers shape: {barriers[0].shape}')
print(f'Timings shape: {timings[0].shape}')


###
#  Enable P2P access
###
print('\nEnabling cross-device access...')
for i in dev_ids:
    for j in dev_ids:
        if i != j:
            assert can_access_peer(i, j), f'Device {i} cannot access device {j}'
            enable_p2p_access(i, j)


###
#   Launch the kernel and benchmark
###
print("\nLaunching the kernel...")
for dev_id in dev_ids:
    torch.cuda.synchronize(dev_id)
matmul(instructions, barriers, timings, A0s, A1s, Bs, Cs)
for dev_id in dev_ids:
    torch.cuda.synchronize(dev_id)

print('\nKernel finished, now benchmarking...')
times = []
for i in range(NUM_ITERS):
    start_time = time()
    matmul(instructions, barriers, timings, A0s, A1s, Bs, Cs)
    for dev_id in dev_ids: # can't use cudaEvent (which is device-specific)
        torch.cuda.synchronize(dev_id)
    end_time = time()
    times.append(end_time - start_time)
avg_time_ms = sum(times) / NUM_ITERS
total_tflop = 2 * M * N * K * 1e-12
print(f'Average time per iter: {avg_time_ms * 1e6} us')
print(f'Total TFLOP/s: {total_tflop / avg_time_ms}')
print(f'Per-device TFLOP/s: {(total_tflop / NUM_DEVICES) / avg_time_ms}')


###
#   Check for correctness (do matmul on GPU for speed)
###
if True:
    print('Skipping correctness check')
    quit()
print("\nChecking for correctness...")
C = torch.cat([tensor.to(dtype=torch.float32, device='cpu') for tensor in Cs], dim=1)
C_ref = (A.to(dtype=torch.float16, device='cuda:0') @ 
         B.to(dtype=torch.float16, device='cuda:0').T).to(torch.float8_e4m3fn).to(dtype=torch.float32, device='cpu') # simulate precision loss

print(C.dtype, C.shape)
print(C_ref.dtype, C_ref.shape)
print('Max abs diff:', abs(C-C_ref).max())
print('Mean abs diff:', abs(C-C_ref).mean())
