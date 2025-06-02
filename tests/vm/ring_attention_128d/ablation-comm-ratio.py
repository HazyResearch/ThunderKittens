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
NUM_ITERS = 5
NUM_WARMUPS = 2
ATTN_OPCODE = 725
COMM_OPCODE = 97
B, H, N, D_h = 1, 16, 16384*NUM_DEVICES//2, 128

assert N%NUM_DEVICES==0, "N must be divisible by NUM_DEVICES"
assert (N//NUM_DEVICES)%512==0, "N_per_dev must be divisible by 512 (QO Block Size * NUM_CONSUMERS * CTA Cluster Size)"
assert D_h==128, "D_h must be 128"

N_per_dev = N // NUM_DEVICES
num_qo_blocks = N_per_dev // 512 # 128 * NUM_CONSUMERS * CTA Cluster Size
num_kv_blocks = N_per_dev // 128
num_comps = B * H * num_qo_blocks * 2 # 2 CTAs per cluster
num_ring_stages = NUM_DEVICES


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
def generate_instructions(is_comm, ring_stage, torch_device):
    dev_idx = torch_device.index
    prev_dev_idx = (dev_idx + NUM_DEVICES - 1) % NUM_DEVICES
    next_dev_idx = (dev_idx + 1) % NUM_DEVICES
    dev_instructions = [[] for _ in range(SM_COUNT)]
    num_comps = B * H * num_qo_blocks * 2 # 2 CTAs per cluster

    # Comm Ops
    if is_comm:
        num_comms_per_kv = NUM_COMMS // 2
        num_chunks_N = N_per_dev // 64 # comm tile height
        total_chunks = B * H * num_chunks_N
        num_chunks_per_comm = total_chunks // (num_comms_per_kv)
        assert NUM_COMMS % 2 == 0, "NUM_COMMS must be even"
        assert total_chunks % (num_comms_per_kv) == 0, "total_chunks must be divisible by NUM_COMMS / 2"
        for i in range(NUM_COMMS):
            k_or_v = i // (num_comms_per_kv)
            comm_idx = i % (num_comms_per_kv)
            dev_instructions[i].append( # CHANGED (just wait for comms to finish)
                [COMM_OPCODE, k_or_v, num_chunks_per_comm, comm_idx, NUM_COMMS, 0, num_chunks_N, H, dev_idx, prev_dev_idx, next_dev_idx, ring_stage] 
                + [0]*20
            )

    # Compute Ops
    else:
        instruction_idx = 0
        assert instruction_idx%2 == 0, "instruction_idx must be even at start for CTA cluster matching"
        assert NUM_COMMS%2 == 0, "NUM_COMMS must be even for CTA cluster matching"
        assert SM_COUNT%2 == 0, "SM_COUNT must be even for CTA cluster matching"
        for batch_idx in range(B):
            for head_idx in range(H):
                for qo_idx in range(num_qo_blocks):
                    for __cta_rank in range(2):
                        dev_instructions[instruction_idx%SM_COUNT].append( # CHANGED (just wait for comps to finish)
                            [ATTN_OPCODE, batch_idx, head_idx, qo_idx*4, num_kv_blocks, ring_stage, 0, num_comps, dev_idx]
                            + [0]*23
                        )
                        instruction_idx += 1

    # Paddings
    max_instruction_len = 0
    for sm_instructions in dev_instructions:
        max_instruction_len = max(max_instruction_len, len(sm_instructions))
    for sm_instructions in dev_instructions:
        while len(sm_instructions) < max_instruction_len:
            sm_instructions.append([0]*32)

    return torch.tensor(dev_instructions, dtype=torch.int32, device=torch_device)

comp_instructions = []
comp_timings = []
comm_instructions = []
comm_timings = []

for torch_device in torch_devices:
    dev_instructions = []
    for ring_stage in range(num_ring_stages):
        dev_instructions.append(generate_instructions(False, ring_stage, torch_device))
    comp_instructions.append(torch.cat(dev_instructions, dim=1))
    comp_timings.append(torch.zeros((SM_COUNT, comp_instructions[-1].shape[1], 128), dtype=torch.int32, device=torch_device))

    dev_instructions = []
    for ring_stage in range(num_ring_stages - 1):
        dev_instructions.append(generate_instructions(True, ring_stage, torch_device))
    comm_instructions.append(torch.cat(dev_instructions, dim=1))
    comm_timings.append(torch.zeros((SM_COUNT, comm_instructions[-1].shape[1], 128), dtype=torch.int32, device=torch_device))


###
#   Prepare Timings and Barriers
###
print('\nGenerating barriers and timings...')
barriers = []
for torch_device in torch_devices:
    barriers.append(torch.zeros((NUM_DEVICES,), dtype=torch.uint32, device=torch_device))


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
#  Check speed
###
print('\nNow benchmarking...')
for i in range(NUM_WARMUPS):
    for dev_id in dev_ids:
        barriers[dev_id].zero_()
        torch.cuda.synchronize(dev_id)
    ring_attention(
        comm_instructions, barriers, comm_timings,
        Qs, K0s, K1s, V0s, V1s, Os, Ls, Ms
    )
    for dev_id in dev_ids:
        torch.cuda.synchronize(dev_id)
for i in range(NUM_WARMUPS):
    for dev_id in dev_ids:
        barriers[dev_id].zero_()
        torch.cuda.synchronize(dev_id)
    ring_attention(
        comp_instructions, barriers, comp_timings,
        Qs, K0s, K1s, V0s, V1s, Os, Ls, Ms
    )
    for dev_id in dev_ids:
        torch.cuda.synchronize(dev_id)
comp_times = []
comm_times = []
for i in range(NUM_ITERS):
    for dev_id in dev_ids:
        barriers[dev_id].zero_()
        torch.cuda.synchronize(dev_id)
    start_time = time()
    ring_attention(
        comm_instructions, barriers, comm_timings,
        Qs, K0s, K1s, V0s, V1s, Os, Ls, Ms
    )
    for dev_id in dev_ids:
        torch.cuda.synchronize(dev_id)
    end_time = time()
    comm_times.append(end_time - start_time)

    for dev_id in dev_ids:
        barriers[dev_id].zero_()
        torch.cuda.synchronize(dev_id)
    start_time = time()
    ring_attention(
        comp_instructions, barriers, comp_timings,
        Qs, K0s, K1s, V0s, V1s, Os, Ls, Ms
    )
    for dev_id in dev_ids:
        torch.cuda.synchronize(dev_id)
    end_time = time()
    comp_times.append(end_time - start_time)

comm_avg_time = sum(comm_times) / NUM_ITERS
comp_avg_time = sum(comp_times) / NUM_ITERS
print(f'Compute avg time per iter: {comp_avg_time * 1e6} us')
print(f'Communication avg time per iter: {comm_avg_time * 1e6} us')
print(f'Compute/Communication ratio: {comp_avg_time / comm_avg_time}')
