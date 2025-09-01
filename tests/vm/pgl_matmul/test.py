import torch
from matmul import matmul, enable_all_p2p_access, KittensClub, make_globals

M, K, N = 8192, 16384, 8192

NUM_DEVICES = 4

OPCODE = 1
SM_COUNT = 132
INSTRUCTION_WIDTH = 32
TIMING_WIDTH = 128
NUM_ITERS = 10
NUM_WARMUP_ITERS = 2

M_BLOCK = 128
K_BLOCK = 128
N_BLOCK = 256

print('Starting test...')

# Create input and output tensors
torch.manual_seed(1)

# A is broadcast to all devices
A = (torch.randn((M, K), device=0, dtype=torch.float32) / K**.25).to(torch.float8_e4m3fn)
As = [A.to(i) for i in range(NUM_DEVICES)]

# B is column-wise sharded
Bs = [(torch.randn((N//NUM_DEVICES, K), device=0, dtype=torch.float32) / K**.25).to(dtype=torch.float8_e4m3fn, device=i) for i in range(NUM_DEVICES)]
B = torch.cat([_B.to(0) for _B in Bs], dim=0)

# Every device has the full C
Cs = [torch.zeros((M, N), dtype=torch.float8_e4m3fn, device=i) for i in range(NUM_DEVICES)]

# Generate instructions
instructions = [[] for _ in range(SM_COUNT)]
num_iters = K // K_BLOCK
instruction_idx = 0
for row in range(M // M_BLOCK):
    for col in range((N // NUM_DEVICES) // N_BLOCK):
        instructions[instruction_idx%SM_COUNT].append([OPCODE, row, col, num_iters] + [0]*(INSTRUCTION_WIDTH-4))
        instruction_idx += 1

# Pad instructions
max_instructions = -1
for i in range(SM_COUNT):
    max_instructions = max(max_instructions, len(instructions[i]))
for i in range(SM_COUNT):
    while len(instructions[i]) < max_instructions:
        instructions[i].append([0] * INSTRUCTION_WIDTH)

# Instructions and timings are identical in all devices for this test
instructions = [torch.tensor(instructions, dtype=torch.int32, device=i) for i in range(NUM_DEVICES)]
timings = [torch.zeros((SM_COUNT, instructions[i].shape[1], TIMING_WIDTH), dtype=torch.int32, device=i) for i in range(NUM_DEVICES)]

# Prepare for multigpu kernel
device_ids = [i for i in range(NUM_DEVICES)]
club = KittensClub(device_ids)
kernel_globals = make_globals(instructions, timings, As, Bs, Cs)

# Run the matmul kernel
print('Launching kernel...')
matmul(club, *kernel_globals)
for i in range(NUM_DEVICES): 
    torch.cuda.synchronize(i)

# Check the results
C_ref = (A.to(torch.float16) @ B.to(torch.float16).T).to(torch.float8_e4m3fn).to(torch.float32).cpu().numpy()
for i, C in enumerate(Cs):
    print(f'Device {i}:')
    assert C.shape == C_ref.shape, f'Device {i} has shape {C.shape} but expected {C_ref.shape}'
    C = C.to(torch.float32).cpu().numpy()
    print('abs diff max:', abs(C - C_ref).max())
    print('abs diff mean:', abs(C - C_ref).mean())
