from functools import partial
import os
from time import perf_counter
import torch
import torch.distributed

# Baseline repo: https://github.com/xdit-project/xDiT
from yunchang.kernels import AttnType
from xfuser.core.long_ctx_attention.ring.ring_flash_attn import xdit_ring_flash_attn_func
from xfuser.core.distributed import init_distributed_environment


###
#   Global Parameters
###
NUM_DEVICES = int(os.environ['WORLD_SIZE'])
CURRENT_DEVICE = int(os.environ['RANK'])
NUM_ITERS = 2
NUM_WARMUPS = 5
B, H, N, D_h = 1, 16, 16384*NUM_DEVICES*1, 128

assert os.environ['RANK']==os.environ['LOCAL_RANK'], 'Must be run on single node'
assert NUM_DEVICES>=1, 'NUM_DEVICES must be >= 1'
assert N%NUM_DEVICES==0, 'N must be divisible by NUM_DEVICES'


###
#   Prepare CUDA environment
###
world_size = NUM_DEVICES
rank = CURRENT_DEVICE
print(f'Process started with rank (device ID) {rank} and world size {world_size}.')
torch_device = torch.device(f'cuda:{rank}')
torch.cuda.set_device(torch_device)
init_distributed_environment(rank=rank, world_size=world_size, local_rank=-1, backend='nccl')


###
#   Prepare Inputs
###
print(f'Rank {rank}: Starting test with B={B}, H={H}, N={N}, D_h={D_h}, NUM_DEVICES={NUM_DEVICES}')
print(f'Rank {rank}: Generating inputs...')
N_per_dev = N // NUM_DEVICES
torch.manual_seed(42+rank)
local_Q = torch.randn((B, N_per_dev, H, D_h), device=torch_device, dtype=torch.bfloat16, requires_grad=False)
local_K = torch.randn((B, N_per_dev, H, D_h), device=torch_device, dtype=torch.bfloat16, requires_grad=False)
local_V = torch.randn((B, N_per_dev, H, D_h), device=torch_device, dtype=torch.bfloat16, requires_grad=False)


###
#  Check speed
###
print(f'Rank {rank}: Benchmarking...')
ring_attn = partial(
    xdit_ring_flash_attn_func,
    q=local_Q,
    k=local_K,
    v=local_V,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),
    group=torch.distributed.group.WORLD,
    alibi_slopes=None,
    deterministic=True,
    return_attn_probs=False,
    attn_type=AttnType.FA,
    attn_processor=None,
    attn_layer=None,
    joint_tensor_key=None,
    joint_tensor_value=None,
    joint_strategy='none'
)

for i in range(NUM_WARMUPS):
    _ = ring_attn()
    torch.cuda.synchronize()
torch.distributed.barrier()

times = []
for i in range(NUM_ITERS):
    torch.distributed.barrier()
    start_time = perf_counter()
    _ = ring_attn()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    end_time = perf_counter()
    times.append(end_time - start_time)
times_sum = torch.tensor([sum(times)], device=torch_device)
torch.distributed.reduce(times_sum, dst=0, op=torch.distributed.ReduceOp.SUM)
if rank == 0:
    print(times_sum.item())
    avg_time = times_sum.item() / (NUM_ITERS * NUM_DEVICES)
    total_tflops = (4 * B * H * N * N * D_h + 4 * B * H * N * N) * 1e-12
    print(f'Average time per iter: {avg_time * 1e6} us')
    print(f'Total TFLOP/s: {total_tflops / avg_time}')
    print(f'Per-device TFLOP/s: {(total_tflops / NUM_DEVICES) / avg_time}')
torch.distributed.barrier() # for clean print


###
#  Clean up
###
print(f'Rank {rank}: Test complete.')
torch.distributed.destroy_process_group()
