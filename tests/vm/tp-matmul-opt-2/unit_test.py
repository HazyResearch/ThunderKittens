import torch
from matmul import matmul, can_access_peer, enable_p2p_access
from time import time


###
#   Global Parameters
###
NUM_DEVICES = 8
NUM_COMMS = 8 # this is the magic number that works the best
NUM_ITERS = 5
MATMUL_OPCODE = 725
COMM_OPCODE = 97
# M, K, N = 3072, 4096, 3072
# M, K, N = 512, 256, 256
M, K, N = 16384*8, 3072, 16384*8
# M, K, N = 3072, 16384*2, 3072 
# M, K, N = 256, 4096, 256

if M%256 != 0: raise ValueError("M must be divisible by 256")
if K%128 != 0: raise ValueError("K must be divisible by 128")
if N%256 != 0: raise ValueError("N must be divisible by 256")
if M%NUM_DEVICES != 0: raise ValueError("M must be divisible by NUM_DEVICES")
if N%NUM_DEVICES != 0: raise ValueError("N must be divisible by NUM_DEVICES")
if (M//NUM_DEVICES)%256 != 0: raise ValueError("M//NUM_DEVICES must be divisible by 256")
if (N//NUM_DEVICES)%256 != 0: raise ValueError("N//NUM_DEVICES must be divisible by 256")


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
#   Prepare Instructions
###
instructions = []
M_per_dev = M // NUM_DEVICES
N_per_dev = N // NUM_DEVICES
num_ring_stages = NUM_DEVICES
num_rows = M_per_dev // 256
num_cols = N_per_dev // 256
num_iters = K // 128
for torch_device in torch_devices:
    dev_idx = torch_device.index
    prev_dev_idx = (dev_idx + NUM_DEVICES - 1) % NUM_DEVICES
    next_dev_idx = (dev_idx + 1) % NUM_DEVICES
    dev_instructions = [[] for _ in range(148)]

    # Comm Ops
    num_chunks = (M_per_dev * K) // 16384
    num_chunk_cols = K // 128
    comm_size = num_chunks // NUM_COMMS
    num_comps = num_rows * num_cols
    for comm_idx in range(NUM_COMMS):
        dev_instructions[comm_idx].append([COMM_OPCODE, comm_size, comm_idx, NUM_COMMS, num_comps, num_chunk_cols, dev_idx, prev_dev_idx, next_dev_idx] + [0]*23)

    # Compute Ops
    instruction_idx = 0
    for ring_stage in range(num_ring_stages):
        for row in range(num_rows):
            row_global_start = num_rows * dev_idx
            row_global_total = num_rows * num_ring_stages
            row_global = (row_global_start + row_global_total - num_rows * ring_stage + row) % row_global_total
            for col in range(num_cols):
                dev_instructions[NUM_COMMS+(instruction_idx%(148-NUM_COMMS))].append([MATMUL_OPCODE, 2*row, 2*col, 2*row_global, num_iters, ring_stage, NUM_COMMS, num_comps, dev_idx] + [0]*23)
                instruction_idx += 1

    # Paddings
    max_instruction_len = 0
    for sm_instructions in dev_instructions:
        max_instruction_len = max(max_instruction_len, len(sm_instructions))
    for sm_instructions in dev_instructions:
        while len(sm_instructions) < max_instruction_len:
            sm_instructions.append([0]*32)

    # Append
    instructions.append(torch.tensor(dev_instructions, dtype=torch.int32, device=torch_device))

print(f'Instructions shape: {instructions[0].shape}')


###
#   Prepare Timings and Barriers
###
print('\nGenerating barriers and timings...')
barriers = []
timings = []
for torch_device in torch_devices:
    barriers.append(torch.zeros((NUM_DEVICES,), dtype=torch.uint32, device=torch_device))
    timings.append(torch.zeros((148, instructions[0].shape[1], 128), dtype=torch.int32, device=torch_device))

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
    barriers[dev_id].zero_()
    torch.cuda.synchronize(dev_id)
matmul(instructions, barriers, timings, A0s, A1s, Bs, Cs)
for dev_id in dev_ids:
    torch.cuda.synchronize(dev_id)

print('\nKernel finished, now benchmarking...')
times = []
for i in range(NUM_ITERS):
    for dev_id in dev_ids:
        barriers[dev_id].zero_()
        torch.cuda.synchronize(dev_id)
    start_time = time()
    matmul(instructions, barriers, timings, A0s, A1s, Bs, Cs)
    for dev_id in dev_ids: # can't use cudaEvent (which is device-specific)
        torch.cuda.synchronize(dev_id)
    end_time = time()
    times.append(end_time - start_time)
avg_time = sum(times) / NUM_ITERS
total_tflop = 2 * M * N * K * 1e-12
print(f'Average time per iter: {avg_time * 1e6} us')
print(f'Total TFLOP/s: {total_tflop / avg_time}')
print(f'Per-device TFLOP/s: {(total_tflop / NUM_DEVICES) / avg_time}')
print(f'Per-unidirectional-NVLink GB/s: {M_per_dev * 7 * K * 1e-9 / avg_time}')


###
#   Check for correctness
###
if True: # note that running the kernel more than once will cause the results to be incorrect
    print('\nSkipping correctness check')
    quit()

def check_diff(x, y):
    x = x.to(dtype=torch.float32, device='cpu')
    y = y.to(dtype=torch.float32, device='cpu')
    abs_diff = abs(x - y)
    print('Max abs diff:', abs_diff.max())
    print('Mean abs diff:', abs_diff.mean())

print("\nChecking for correctness...")
C = torch.cat([tensor.to(dtype=torch.float32, device='cpu') for tensor in Cs], dim=1)
C_ref = (A.to(dtype=torch.float16, device='cuda:0') @ 
         B.to(dtype=torch.float16, device='cuda:0').T).to(torch.float8_e4m3fn).to(dtype=torch.float32, device='cpu') # simulate precision loss
check_diff(C, C_ref)

# Sanity check if comms are correct
As = A.chunk(len(torch_devices), dim=0)
for i in range(NUM_DEVICES):
    check_diff(A0s[i], As[(i + 2) % NUM_DEVICES])
    check_diff(A1s[i], As[(i + 1) % NUM_DEVICES])
