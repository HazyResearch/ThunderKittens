import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial

try:
    from _C import fused_rotary
    print(f"Successfully imported thunderkittens")
except:
    print("Could not import thunderkittens")

try:
    from rotary.baselines.rotary import RotaryEmbedding
    from rotary.baselines.rotary import apply_rotary_emb
    print(f"Successfully imported RotaryEmbedding")
except:
    RotaryEmbedding = None
    print("Could not import RotaryEmbedding. Please obtain from FlashAttention repo.")
    

#################### Rotary ########################

def get_flops(batch, seqlen, headdim, nheads):
    f = batch*seqlen*nheads*(headdim//2) # mult with cos 
    f += batch*seqlen*nheads*(headdim//2) # mult with sin
    f += batch*seqlen*nheads*(headdim//2) # add rotated values
    return f * 2 # for q and k 


def get_rotary_inputs(b, h, n, dv, dt):
    d_model = h * dv
    head_dim = dv
    rotary_emb_fraction = 1
    rotary_emb_dim = head_dim * rotary_emb_fraction
    rotary_emb_base = 10000
    rotary_max_seqlen = None
    rotary_emb_scale_base=None
    rotary_emb_interleaved=False

    assert RotaryEmbedding is not None, "RotaryEmbedding not imported"

    rotary_emb = RotaryEmbedding(
        rotary_emb_dim,
        base=rotary_emb_base,
        scale_base=rotary_emb_scale_base,
        interleaved=rotary_emb_interleaved,
        device='cuda',
    )

    h = d_model // head_dim
    qkv = torch.randn((b, n, 3, h, head_dim), dtype=dt, device='cuda')
    qkv_emb_flash = rotary_emb(
        qkv, seqlen_offset=0, max_seqlen=rotary_max_seqlen
    )
    sin = rotary_emb._sin_cached
    cos = rotary_emb._cos_cached
    return qkv, cos, sin, rotary_max_seqlen, rotary_emb

def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)

class PytorchRotary(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, q, k, cos, sin, o_dt, dt, ro_dim, interleaved):
        # for q
        cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
        sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
        y1 = torch.cat(
            [q[..., :ro_dim] * cos + rotate_half(q[..., :ro_dim], interleaved) * sin, q[..., ro_dim:]],
            dim=-1,
        )
        
        # for k
        y2 = torch.cat(
            [k[..., :ro_dim] * cos + rotate_half(k[..., :ro_dim], interleaved) * sin, k[..., ro_dim:]],
            dim=-1,
        )
        return y1, y2
        
def rotary_test(dt, b, h, n, dv, causal, is_forwards, method_str, num_iters=10, verbose=True, torch_compile=False, **kwargs):
    # Reference: https://github.com/Dao-AILab/flash-attention/blob/898dd4bbf237b24ed8fd2a3d13ee33bd156bfb23/flash_attn/modules/mha.py#L95
    
    pytorch_method = PytorchRotary()
    if torch_compile and method_str == "torch":
        try:
            pytorch_method = torch.compile(pytorch_method)
        except Exception as e:
            print(f"Could not compile pytorch_method: {e}")
            
    for stage in ['warmup', 'timed']:
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

        for i in range(num_iters):
            qkv, cos, sin, rotary_max_seqlen, rotary_emb = get_rotary_inputs(b, h, n, dv, dt)
        
            try:
                if method_str == "flash_triton":
                    q = qkv[:,:,0]
                    k = qkv[:,:,1]

                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = apply_rotary_emb(
                        q, cos, sin
                    )
                    y = apply_rotary_emb(
                        k, cos, sin
                    )
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "torch":
                    o_dt = qkv.dtype
                    ro_dim = cos.shape[-1] * 2
                    assert ro_dim <= qkv.shape[-1]
                    interleaved = False
                    q = qkv[:, :, 0]
                    k = qkv[:, :, 1]

                    torch.cuda.synchronize()
                    start_events[i].record()
                    
                    y1, y2 = pytorch_method(q, k, cos, sin, o_dt, dt, ro_dim, interleaved)
                    y = torch.cat((y1, y2), dim=-1)
                    
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "tk":
                    q = qkv[:,:,0].transpose(1,2).clone()   # torch.Size([b, n, 3, h, dv])
                    k = qkv[:,:,1].transpose(1,2).clone()
                    cos = cos.contiguous()
                    sin = sin.contiguous()
                    q = q.contiguous()
                    k = k.contiguous()

                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = fused_rotary(q, cos, sin)
                    y = fused_rotary(k, cos, sin)
                    end_events[i].record()
                    torch.cuda.synchronize()

                else:
                    assert 0, f"Unknown method: {method_str}"

            except Exception as e:
                if verbose:
                    print(f"Error in rotary_test: {e}")
                return None, -1

            torch.cuda.empty_cache()

    tot = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)])/num_iters
    assert not np.isnan(y.detach().float().cpu()).any(), "NaN values detected in output 'out'"
    assert not np.isinf(y.detach().float().cpu()).any(), "Inf values detected in output 'out'"
    return y, tot


IMPLEMENTATIONS = {
    "flash_triton": partial(rotary_test, causal=True, is_forwards=True, method_str="flash_triton"),
    "torch": partial(rotary_test, causal=True, is_forwards=True, method_str="torch"),
    "tk": partial(rotary_test, causal=True, is_forwards=True, method_str="tk"),
}

NAME = "ROTARY"

################## Common Benchmark Harness ######################

# TODO: this code is redundant on every benchmark currently

import pandas as pd

import warnings
warnings.filterwarnings("ignore", message=".*not a leaf Tensor is being accessed.*")
warnings.filterwarnings("ignore", message=".*no current CUDA context.*")

b = 16
h = 16
dv = 64

def efficiency(flops, time):
    tflops = flops / 1e12
    time_ms = time / 1e6
    return tflops / time_ms

def measure_efficiency(dt, n, method_name, method, verbose=False, torch_compile=True):
    if verbose:
        print(f"{b=}, {n=}, {h=}, {dv=}")

    if 'c=t' in method_name:
        causal = True
        flops = get_flops(b, n, dv, h, causal=('causal'), mode='bwd' if 'bwd' in method_name else 'fwd')
    elif 'c=f' in method_name:
        causal = False
        flops = get_flops(b, n, dv, h, causal=causal, mode='bwd' if 'bwd' in method_name else 'fwd')
    else:
        flops = get_flops(b, n, dv, h)

    outputs, times = method(dt, b, h, n, dv, verbose=verbose, torch_compile=torch_compile)
    times = times * 1000

    eff = efficiency(flops, times)
    if verbose:
        print(f"Method {method_name} -- Efficiency: {eff:.2f} TFLOPS, Time: {times:.4f} us and FLOPS: {flops:.2f}")
    torch.cuda.empty_cache()
    return eff, times

if __name__ == "__main__":
    print("Benchmarking the kernels...")

    verbose = True
    torch_compile = False

    implementations_list = []
    implementations_fwd = IMPLEMENTATIONS
    implementations_list.append(implementations_fwd)
    name = NAME
    print("============" * 4, name, "============" * 4)

    try:
        implementations_bwd = IMPLEMENTATIONS_BWD
        implementations_list.append(implementations_bwd)
    except:
        pass

    for implementations in implementations_list:
        method2tflops = {}
        method2timing = {}

        for m, method in implementations.items():
            flops_result, timing_result = {},  {}
            if verbose:
                print(f"Method: {m}")
            for n in [
                1024 if 'attn' not in m else 768, 
                2048 if 'attn' not in m else 1536, 
                4096 if 'attn' in m else 3072,
                8192 if 'attn' in m else 6144,
                16384 if 'attn' in m else 12288
            ]:
                if "conv" in m and n not in [1024, 4096]:
                    # restrict to sizes we have implemented
                    continue
                if "mamba2_triton" in m and n not in [1024, 2048, 4096, 8192]:
                    # the kernel results in DEVICE_SIDE_ASSERTS
                    continue
                if "layernorm" in m and dv*h != 1024:
                    # restrict to sizes we have implemented
                    print('skipping layernorm due to incompatible model dim')
                if verbose:
                    print(f"Sequence Length: {n}")
                tflops, timing = measure_efficiency(torch.bfloat16, n, m, method, verbose=verbose, torch_compile=torch_compile)
                if tflops > 0: 
                    flops_result[n] = tflops
                    timing_result[n] = timing
            method2tflops[m] = flops_result
            method2timing[m] = timing_result

        # print table
        df = pd.DataFrame(method2tflops).replace(np.nan, 'OOM', regex=True)
        print(df)
