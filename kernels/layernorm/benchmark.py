import os    
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial

try:
    from _C import fused_layernorm
    print(f"Successfully imported thunderkittens")
except:
    print("Could not import thunderkittens")

try:
    from baselines.layer_norm_triton import layer_norm_fn, RMSNorm
    print(f"Successfully imported layer_norm_fn")
except:
    layer_norm_fn = None
    print("Could not import layer_norm_fn. Please obtain from FlashAttention repo.")


################## Layer norm ######################

def get_flops(batch, seqlen, headdim, nheads):
    f = batch*seqlen*nheads*headdim # compute the mask for dropout 
    f += batch*seqlen*nheads*headdim # add dropout and residual
    f += batch*seqlen*nheads*headdim # compute the mean
    f += batch*seqlen*nheads*headdim # compute the variance
    f += batch*seqlen*nheads*headdim # subtract mean
    f += batch*seqlen*nheads*headdim # divide by variance
    f += batch*seqlen*nheads*headdim # multiply by norm weight 
    f += batch*seqlen*nheads*headdim # add norm bias
    return f


def get_layer_norm_inputs(b, h, n, dv, dt):
    d_model = h * dv
    p = 0.1

    norm = nn.LayerNorm(d_model).cuda()
    dropout = nn.Dropout(p)

    x = torch.randn(b, n, d_model).to(dtype=dt, device='cuda')
    residual = torch.randn(b, n, d_model).to(dtype=dt, device='cuda')
    norm_weight = norm.weight.to(dtype=torch.bfloat16)
    norm_bias = norm.bias.to(dtype=torch.bfloat16)
    out = torch.zeros_like(x)
    out_resid = torch.zeros_like(x)
    return x, residual, norm_weight, norm_bias, out, out_resid, norm, dropout

class pytorch_layernorm(torch.nn.Module):
    def __init__(self, d_model, p):
        super().__init__()
        self.dropout = nn.Dropout(p).cuda()
        self.norm = nn.LayerNorm(d_model).cuda()
        
    def forward(self, x, residual, norm_weight, norm_bias, dropout_p):
        with torch.no_grad():
            dropped = self.dropout(x) 
            residual = (residual + dropped ) if residual is not None else dropped
            y = self.norm(residual.to(dtype=self.norm.weight.dtype))
            residual = residual.to(torch.float32)
        return y, residual
        
def layernorm_test(dt, b, h, n, dv, causal, is_forwards, method_str, num_iters=10, verbose=True, torch_compile=True, **kwargs):
    
    pytorch_method = pytorch_layernorm(d_model=h*dv, p=0.1)
    if torch_compile and method_str == "pytorch_layernorm":
        try:
            pytorch_method = torch.compile(pytorch_method)
        except Exception as e:
            print(f"Could not compile pytorch_layernorm: {e}")
                     
    for stage in ['warmup', 'timed']:
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    
        for i in range(num_iters):
            (
                x, residual, norm_weight, norm_bias, 
                out, out_resid, norm, dropout 
            ) = get_layer_norm_inputs(b, h, n, dv, dt)
            rowscale = torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype, )
            residual_in_fp32 = True
            dtype = dt
            dropout_p = dropout.p
                
            if True:
                if method_str == 'pytorch_layernorm':
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y, residual = pytorch_method(x, residual, norm_weight, norm_bias, dropout_p)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == 'triton_layernorm':
                    torch.cuda.synchronize()
                    start_events[i].record()
                    with torch.no_grad():
                        y, residual = layer_norm_fn(
                            x,
                            norm.weight,
                            norm.bias,
                            residual=residual,
                            eps=norm.eps,
                            dropout_p=dropout.p,
                            rowscale=rowscale,
                            prenorm=True,
                            residual_in_fp32=residual_in_fp32,
                            is_rms_norm=False
                        )
                    end_events[i].record()
                    torch.cuda.synchronize()
                
                elif method_str == 'tk_layernorm': 
                    torch.cuda.synchronize()
                    start_events[i].record()
                    with torch.no_grad():
                        y, residual = fused_layernorm(
                            x, residual, 
                            norm_weight, norm_bias, 
                            dropout.p,
                        )
                    end_events[i].record()
                    torch.cuda.synchronize()

                else:
                    raise ValueError(f"Unknown method: {method_str}")

            # except Exception as e:
            #     if verbose:
            #         print(f"Error: {e}")
            #     return None, -1

            torch.cuda.empty_cache()

    assert not np.isnan(y.detach().float().cpu()).any(), "NaN values detected in output 'o'"
    assert not np.isinf(y.detach().float().cpu()).any(), "Inf values detected in output 'o'"
    tot = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)])/num_iters
    return y, tot


IMPLEMENTATIONS = {
    'pytorch_layernorm': partial(layernorm_test, causal=True, is_forwards=True, method_str='pytorch_layernorm'),
    'triton_layernorm': partial(layernorm_test, causal=True, is_forwards=True, method_str='triton_layernorm'),
    'tk_layernorm': partial(layernorm_test, causal=True, is_forwards=True, method_str='tk_layernorm'),
}

NAME = "LAYER NORM"

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
