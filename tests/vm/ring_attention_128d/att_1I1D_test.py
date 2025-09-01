import torch
from ring_attention import ring_attention, can_access_peer, enable_p2p_access
from time import time
import numpy as np


###
#   Global Parameters
###
SM_COUNT = 148
NUM_DEVICES = 4
NUM_COMMS = 8 # this is the magic number that works the best
ATTN_OPCODE = 725
COMM_OPCODE = 97
B, H, N, D_h = 1, 1, 512*NUM_DEVICES, 128

assert N%NUM_DEVICES==0, "N must be divisible by NUM_DEVICES"
assert (N//NUM_DEVICES)%512==0, "N_per_dev must be divisible by 512 (QO Block Size * NUM_CONSUMERS * CTA Cluster Size)"
assert D_h==128, "D_h must be 128"

N_per_dev = N // NUM_DEVICES
num_qo_blocks = N_per_dev // 512 # 128 * NUM_CONSUMERS * CTA Cluster Size
num_kv_blocks = N_per_dev // 128
num_comps = B * H * num_qo_blocks
num_ring_stages = 1 # for unit testing


###
#   Prepare Inputs
###
print(f'Starting test with B={B}, H={H}, N={N}, D_h={D_h}, NUM_DEVICES={NUM_DEVICES}')
print('\nGenerating inputs...')
torch.manual_seed(42)
dev_ids = [i for i in range(NUM_DEVICES)]
torch_devices = [torch.device(f"cuda:{dev_id}") for dev_id in dev_ids]
Qs, K0s, K1s, V0s, V1s, Os, Ls, Ms = [], [], [], [], [], [], [], []
Ks, Vs = [], [] # for correctness check
for torch_device in torch_devices:
    torch.manual_seed(42 + torch_device.index)
    Qs.append(torch.randn((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    K0s.append(torch.randn((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    K1s.append(torch.zeros((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    V0s.append(torch.randn((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    V1s.append(torch.zeros((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    Os.append(torch.zeros((B, H, N_per_dev, D_h), device=torch_device, dtype=torch.bfloat16))
    Ls.append(torch.zeros((B, H, N_per_dev), device=torch_device, dtype=torch.float32))
    Ms.append(torch.zeros((B, H, N_per_dev), device=torch_device, dtype=torch.float32))
    Ks.append(K0s[-1].to('cpu')) # for correctness check
    Vs.append(V0s[-1].to('cpu'))


###
#   Prepare Instructions
###
print('\nGenerating instructions...')
instructions = []
for torch_device in torch_devices:
    dev_idx = torch_device.index
    prev_dev_idx = (dev_idx + NUM_DEVICES - 1) % NUM_DEVICES
    next_dev_idx = (dev_idx + 1) % NUM_DEVICES
    dev_instructions = [[] for _ in range(SM_COUNT)]

    # Compute Ops
    instruction_idx = 0
    assert instruction_idx%2 == 0, "instruction_idx must be even at start for CTA cluster matching"
    assert NUM_COMMS%2 == 0, "NUM_COMMS must be even for CTA cluster matching"
    assert SM_COUNT%2 == 0, "SM_COUNT must be even for CTA cluster matching"
    for ring_stage in range(num_ring_stages):
        for batch_idx in range(B):
            for head_idx in range(H):
                for qo_idx in range(num_qo_blocks):
                    for __cta_rank in range(2):
                        dev_instructions[NUM_COMMS+(instruction_idx%(SM_COUNT-NUM_COMMS))].append(
                            [ATTN_OPCODE, batch_idx, head_idx, qo_idx*4, num_kv_blocks, ring_stage, NUM_COMMS, num_comps, dev_idx]
                            + [0]*23
                        )
                        instruction_idx += 1
    print('Number of QO blocks per device:', num_qo_blocks)
    print('Number of KV blocks per device:', num_kv_blocks)
    print('Number of compute instructions per device:', instruction_idx)

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
    timings.append(torch.zeros((SM_COUNT, instructions[torch_device.index].shape[1], 128), dtype=torch.int32, device=torch_device))

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
#  Launch the kernel
###
print('\nLaunching kernel...')
for dev_id in dev_ids:
    barriers[dev_id].zero_()
    torch.cuda.synchronize(dev_id)
ring_attention(
    instructions, barriers, timings,
    Qs, K0s, K1s, V0s, V1s, Os, Ls, Ms
)
for dev_id in dev_ids:
    torch.cuda.synchronize(dev_id)


###
#  Verify correctness
###
def pytorch_blk_attn(q, k, v):
    QK = torch.matmul(q.float(), k.float().transpose(-2, -1))
    max_vec = torch.max(QK, dim=-1, keepdim=True).values
    QK *= 0.08838834764831843 * 1.44269504089
    QK -= max_vec * 0.08838834764831843 * 1.44269504089
    QK = torch.exp2(QK)
    norm_vec = torch.sum(QK, dim=-1, keepdim=False)
    out = torch.matmul(QK.bfloat16(), v)
    return out, norm_vec, max_vec

def check_diff(x, y):
    x = x.to(dtype=torch.float32, device='cpu')
    y = y.to(dtype=torch.float32, device='cpu')
    abs_diff = abs(x - y)
    print('Max abs diff:', abs_diff.max())
    print('Mean abs diff:', abs_diff.mean())

for dev_id in dev_ids:
    O_ref, L_ref, M_ref = pytorch_blk_attn(Qs[dev_id].cpu(), Ks[dev_id], Vs[dev_id])
    O_ref = O_ref[0, 0]
    O = Os[dev_id].cpu()[0, 0]
    check_diff(O, O_ref)
