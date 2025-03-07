# this is benchmarked on FA3 at this commit: https://github.com/Dao-AILab/flash-attention/tree/4f0640d534888c579a448fd89c2d4e064905d798

import torch

from triton.testing import do_bench, do_bench_cudagraph

from einops import rearrange

from flash_attn_interface import flash_attn_with_kvcache

try:
    from flash_attn.utils.benchmark import pytorch_profiler
except ImportError:
    pytorch_profiler = None

device = "cuda"
dtype = torch.bfloat16
seqlen = 64 * 1024
nheads = 16
nheads_kv = 16
headdim = 128
headdim_v = 128
has_qv = False

def run_benchmark(seqlens, seqlen_q, num_splits=0):
    print(f' ----------- starting seq_lengths: {seqlens} seqlen_q: {seqlen_q} num_splits: {num_splits} -----------')
    # page_size = None
    page_size = 1

    batch_size = len(seqlens)

    torch.manual_seed(0)

    # batch_size = 
    # cache_seqlens = torch.tensor([seqlen - 1] * batch_size, device=device, dtype=torch.int)
    cache_seqlens = torch.tensor([seqlens[i] for i in range(batch_size)], device=device, dtype=torch.int)
    # cache_seqlens = torch.tensor([53185//4, 53185//4, 53185//4, 53185//4], device=device, dtype=torch.int)
    # cache_seqlens = torch.tensor([seqlen - 1, 1024, 1024, 1024], device=device, dtype=torch.int32)
    # cache_seqlens = torch.tensor([1024] * batch_size, device=device, dtype=torch.int)
    # cache_seqlens = torch.tensor([seqlen - 1, 1024, 1024, 1024], device=device, dtype=torch.int)
    # cache_seqlens = torch.tensor([4500, 45000, 1800, 1800], dtype=torch.int32, device=device)

    # num_splits = 32
    # num_splits = 0
    q = torch.randn(batch_size, seqlen_q, nheads, headdim, dtype=dtype, device=device)
    v_cache = torch.randn(batch_size, seqlen, nheads_kv, headdim_v, dtype=dtype, device=device)
    k_cache = torch.randn(batch_size, seqlen, nheads_kv, headdim, dtype=dtype, device=device)
    if page_size is not None:
        assert seqlen % page_size == 0
        k_cache, v_cache = [rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size) for x in [k_cache, v_cache]]
        page_table = rearrange(torch.arange(batch_size * seqlen // page_size, device=device, dtype=torch.int32),
                            "(b s) -> b s", s=seqlen // page_size)
    else:
        page_table = None
    qv = torch.randn(batch_size, seqlen_q, nheads, headdim_v, dtype=dtype, device=device) if has_qv else None

    # Time in ms
    fn = lambda: flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=cache_seqlens, num_splits=num_splits, qv=qv, page_table=page_table, causal=True)
    t0 = do_bench(fn, warmup=1, rep=10)

    mem_io = cache_seqlens.sum().item() * nheads_kv * (headdim + headdim_v) * 2
    flops = seqlen_q * cache_seqlens.float().sum().item() * nheads * (headdim + headdim_v * 2) * 2
    print(f"Time: {t0 * 1e3:.1f} us, {mem_io * 1e-9 / (t0 * 1e-3):.0f} GB/s, {flops * 1e-12 / (t0 * 1e-3):.0f} TFLOPS/s")


if __name__ == "__main__":
    run_benchmark([4641,45118,1730,1696], 4)
    run_benchmark([65536], 1)
    # run_benchmark([871,568,711,329,617,1015,348,978,543,837,650,1020,924,679,560,497,650,406,381,423,511,423,569,943,645,820,829,883,937,765,711,847,722,546,519,279,516,315,664,845,850,546,670,871,527,329,446,764,582,1011,453,655,532,985,1019,810,317,305,949,317,669,768,530,349], 4)
    # run_benchmark([512]*64, 2)
    # run_benchmark([4096]*132, 4)
