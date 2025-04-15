from functools import partial
from time import time
import gc

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec, NamedSharding
import torch

from implementations import generate_mha_inputs
from original_ring_attention import ring_attention # original authors' public implementation
from tk_ring_attention import ring_mha_forward


# Base Parameters
NUM_DEVICES = 8
B = 4
H = 8
N = 768 * NUM_DEVICES # sequence length (must be 768 * i)
D_h = 64 # head dimension (do not change)
dtype = 'bf16'
causal = False
num_iters = 10

assert NUM_DEVICES >= 1, 'NUM_DEVICES must be >= 1'
assert N % NUM_DEVICES == 0, 'N must be divisible by NUM_DEVICES'


def benchmark_orig_ring_attention(NUM_DEVICES, B, H, N, D_h, dtype='bf16', causal=False):

    # Generate inputs
    Q, K, V, dL = generate_mha_inputs(B, H, N, D_h, dtype=dtype, target='ring_mha_orig', num_devices=NUM_DEVICES)

    # Prepare the JIT'ed function and shard inputs
    mesh = jax.make_mesh((1, 1, NUM_DEVICES, 1), ("dp", "fsdp", "sp", "tp"))
    QKVO_ps = PartitionSpec(("dp", "fsdp"), "sp", "tp", None)
    bias_ps = PartitionSpec(("dp", "fsdp"), None, None, None)
    seg_ids_ps = PartitionSpec(("dp", "fsdp"), None)
    ring_attn_sharded = shard_map( # shard_map automatically JITs the function
        partial(
            ring_attention,
            axis_name="sp",
            float32_logits=False,
            cache_idx=None,
            blockwise_kwargs=dict(
                causal_block_size=None, # no causal mask
                deterministic=True, # or false
                dropout_rng=None, # or other value
                attn_pdrop=0.0, # or other value
                query_chunk_size=N//8, # or other value
                key_chunk_size=N//8, # or other value
                dtype=jax.numpy.bfloat16, # or other value
                policy=jax.checkpoint_policies.nothing_saveable,
                precision=None, # or other value
                prevent_cse=True, # or other value
            )
        ),
        mesh=mesh,
        in_specs=(
            QKVO_ps,
            QKVO_ps,
            QKVO_ps,
            bias_ps,
            seg_ids_ps,
        ),
        out_specs=QKVO_ps,
        check_rep=False
    )
    Q = jax.device_put(Q.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))
    K = jax.device_put(K.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))
    V = jax.device_put(V.transpose(0, 2, 1, 3), NamedSharding(mesh, QKVO_ps))
    attn_bias = jax.device_put(jnp.zeros((B, 1, 1, N)), NamedSharding(mesh, bias_ps))
    seg_ids = jax.device_put(jnp.zeros((B, N), dtype=jnp.int32), NamedSharding(mesh, seg_ids_ps))

    # Warmup
    for _ in range(2):
        _out_orig_ring = ring_attn_sharded(Q, K, V, attn_bias, seg_ids)
    times = []

    # Benchmark
    for _ in range(num_iters):
        start = time()
        _out_orig_ring = ring_attn_sharded(Q, K, V, attn_bias, seg_ids)
        end = time()
        times.append(end - start)
    avg_ms = (sum(times) / num_iters) * 1000
    print(f'    Average time for original ring attention: {avg_ms:.4f} ms')

    # Clean up
    del Q, K, V, dL
    del attn_bias, seg_ids, _out_orig_ring
    gc.collect()


def benchmark_tk_ring_attention(NUM_DEVICES, B, H, N, D_h, dtype='bf16', causal=False):

    # Generate inputs
    Qs, Ks, Vs, dLs = generate_mha_inputs(B, H, N, D_h, dtype=dtype, target='ring_mha_tk', num_devices=NUM_DEVICES)

    # Warmup
    for _ in range(2):
        _out_tk_ring = ring_mha_forward(Qs, Ks, Vs, causal)

    # Benchmark
    times = []
    for _ in range(num_iters):
        start = time()
        _out_tk_ring = ring_mha_forward(Qs, Ks, Vs, causal)
        end = time()
        times.append(end - start)
    avg_ms = (sum(times) / num_iters) * 1000
    print(f'    Average time for TK ring attention: {avg_ms:.4f} ms')

    # Clean up
    for i in range(NUM_DEVICES - 1, -1, -1):
        torch.cuda.set_device(i)
        torch.cuda.synchronize(i)
        torch.cuda.empty_cache()
        del Qs[i], Ks[i], Vs[i], dLs[i], _out_tk_ring[i]
    del Qs, Ks, Vs, dLs, _out_tk_ring
    gc.collect()


def benchmark(NUM_DEVICES, B, H, N, D_h, dtype='bf16', causal=False):

    print(f'\nBatch {B} X Head {H} X Seqlen {N} X Dim {D_h} (num_elem: {B * H * N * D_h}, Q size_per_dev: {B*H*N*D_h*2/(NUM_DEVICES*1024**3):.2f} GB)')

    # Benchmark original ring attention
    benchmark_orig_ring_attention(NUM_DEVICES, B, H, N, D_h, dtype=dtype, causal=causal)

    # Benchmark TK ring attention
    benchmark_tk_ring_attention(NUM_DEVICES, B, H, N, D_h, dtype=dtype, causal=causal)


if __name__ == "__main__":

    # Experiment results (run on 8xH100):
    #
    # Dim should be fixed at 64 for now, and changes in batch, head, seqlen all result in
    # changes in the number of CUDA thread blocks
    #
    # Batch 16 X Head 16 X Seqlen 6144 X Dim 64 (num_elem: 100663296, Q size_per_dev: 0.02 GB)
    #     Average time for TK ring attention: 355.8739 ms
    #     Average time for original ring attention: 1185.9777 ms
    #     Average time for TK attention: 5.4 ms
    # Batch 16 X Head 32 X Seqlen 6144 X Dim 64 (num_elem: 201326592, Q size_per_dev: 0.05 GB)
    #     Average time for TK ring attention: 350.0439 ms
    #     Average time for original ring attention: 1094.8989 ms
    #     Average time for TK attention: 10.78 ms
    # Batch 16 X Head 64 X Seqlen 6144 X Dim 64 (num_elem: 402653184, Q size_per_dev: 0.09 GB)
    #     Average time for TK ring attention: 357.2028 ms
    #     Average time for original ring attention: 1126.7457 ms
    #     Average time for TK attention: 21.81 ms
    # Batch 16 X Head 128 X Seqlen 6144 X Dim 64 (num_elem: 805306368, Q size_per_dev: 0.19 GB)
    #     Average time for TK ring attention: 395.7962 ms
    #     Average time for original ring attention: 1217.9673 ms
    #     Average time for TK attention: 45.33 ms
    # Batch 1 X Head 128 X Seqlen 12288 X Dim 64 (num_elem: 100663296, Q size_per_dev: 0.02 GB)
    #     Average time for TK ring attention: 350.0902 ms
    #     Average time for original ring attention: 1085.0306 ms
    #     Average time for TK attention: 10.58 ms
    # Batch 1 X Head 128 X Seqlen 24576 X Dim 64 (num_elem: 201326592, Q size_per_dev: 0.05 GB)
    #     Average time for TK ring attention: 385.1247 ms
    #     Average time for original ring attention: 1078.6474 ms
    #     Average time for TK attention: 44.11 ms
    # Batch 1 X Head 128 X Seqlen 49152 X Dim 64 (num_elem: 402653184, Q size_per_dev: 0.09 GB)
    #     Average time for TK ring attention: 501.7830 ms
    #     Average time for original ring attention: 1295.1739 ms
    #     Average time for TK attention: 175.42 ms
    # Batch 1 X Head 128 X Seqlen 98304 X Dim 64 (num_elem: 805306368, Q size_per_dev: 0.19 GB)
    #     Average time for TK ring attention: 976.3296 ms
    #     Original ring attention dies with OOM
    #     Average time for TK attention: 702.29 ms
    # Batch 1 X Head 128 X Seqlen 196608 X Dim 64 (num_elem: 1610612736, Q size_per_dev: 0.38 GB)
    #     Average time for TK ring attention: 2778.5540 ms
    #     Original ring attention dies with OOM
    #     Average time for TK attention: 2819.95 ms
    # Batch 1 X Head 128 X Seqlen 393216 X Dim 64 (num_elem: 3221225472, Q size_per_dev: 0.75 GB)
    #     Average time for TK ring attention: 10052.4999 ms
    #     Original ring attention dies with OOM
    #     Average time for TK attention: 11311.1251 ms

    # Experiment results on optimized ring attention (no pgl init overhead)
    # 
    # Batch 16 X Head 16 X Seqlen 6144 X Dim 64 (num_elem: 100663296, Q size_per_dev: 0.02 GB)
    #     Average time for TK ring attention: 6.2441 ms
    # Batch 16 X Head 32 X Seqlen 6144 X Dim 64 (num_elem: 201326592, Q size_per_dev: 0.05 GB)
    #     Average time for TK ring attention: 10.9082 ms
    # Batch 16 X Head 64 X Seqlen 6144 X Dim 64 (num_elem: 402653184, Q size_per_dev: 0.09 GB)
    #     Average time for TK ring attention: 19.5219 ms
    # Batch 16 X Head 128 X Seqlen 6144 X Dim 64 (num_elem: 805306368, Q size_per_dev: 0.19 GB)
    #     Average time for TK ring attention: 37.4881 ms
    # Batch 1 X Head 128 X Seqlen 12288 X Dim 64 (num_elem: 100663296, Q size_per_dev: 0.02 GB)
    #     Average time for TK ring attention: 11.7452 ms
    # Batch 1 X Head 128 X Seqlen 24576 X Dim 64 (num_elem: 201326592, Q size_per_dev: 0.05 GB)
    #     Average time for TK ring attention: 47.1947 ms
    # Batch 1 X Head 128 X Seqlen 49152 X Dim 64 (num_elem: 402653184, Q size_per_dev: 0.09 GB)
    #     Average time for TK ring attention: 166.0664 ms
    # Batch 1 X Head 128 X Seqlen 98304 X Dim 64 (num_elem: 805306368, Q size_per_dev: 0.19 GB)
    #     Average time for TK ring attention: 621.9198 ms
    # Batch 1 X Head 128 X Seqlen 196608 X Dim 64 (num_elem: 1610612736, Q size_per_dev: 0.38 GB)
    #     Average time for TK ring attention: 2425.7548 ms
    # Batch 1 X Head 128 X Seqlen 393216 X Dim 64 (num_elem: 3221225472, Q size_per_dev: 0.75 GB)
    #     Average time for TK ring attention: 9771.9243 ms

    benchmark(NUM_DEVICES, B=B, H=16, N=N, D_h=D_h)
    benchmark(NUM_DEVICES, B=B, H=32, N=N, D_h=D_h)
    benchmark(NUM_DEVICES, B=B, H=64, N=N, D_h=D_h)
    benchmark(NUM_DEVICES, B=B, H=128, N=N, D_h=D_h)
    benchmark(NUM_DEVICES, B=1, H=128, N=N*2, D_h=D_h)
    benchmark(NUM_DEVICES, B=1, H=128, N=N*4, D_h=D_h)
    benchmark(NUM_DEVICES, B=1, H=128, N=N*8, D_h=D_h)
    benchmark(NUM_DEVICES, B=1, H=128, N=N*16, D_h=D_h)
    benchmark(NUM_DEVICES, B=1, H=128, N=N*32, D_h=D_h)
    benchmark(NUM_DEVICES, B=1, H=128, N=N*64, D_h=D_h)
