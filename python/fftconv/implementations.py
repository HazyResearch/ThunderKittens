import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial

try:
    import thunderkittens as tk
    print(f"Successfully imported thunderkittens")
except:
    print("Could not import thunderkittens")

try:
    from flashfftconv import FlashFFTConv
    print(f"Successfully imported FlashFFTConv")
except:
    FlashFFTConv = None
    print("Could not import FlashFFTConv. Please clone and install.")

try:
    from fftconv.baselines.tk_fftconv import TKFFTConv
    from fftconv.baselines.tk_fftconv import ref_fftconv
    print(f"Successfully imported TKFFTConv")
except:
    TKFFTConv = None
    print("Could not import TKFFTConv. Please install from TK.")


################ FFT Convolution ################


def get_flops(b: int, n: int, dv: int, h: int, causal: bool = False):
    flops = 2 * (10 * n * math.log(n, 2) * (h * dv) * b)
    return flops 


def get_fft_inputs(dt, b, h, n, dv):
    d_model = (h * dv)
    u = torch.randn((b, d_model, n), dtype=dt).to('cuda')
    k = torch.randn((d_model, n), dtype=torch.float32).to('cuda')
    return u, k

class PytorchFFTConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, k, n):
        return ref_fftconv(x, k, n)

def fftconv_tk_test(dt, b, h, n, dv, causal, is_forwards, method_str, num_iters=10, verbose=True, torch_compile=False, **kwargs):
    
    pytorch_method = PytorchFFTConv()
    if torch_compile and method_str == "conv_torch":
        print("Torchinductor does not support code generation for complex operators. Performance may be worse than eager.")
        print("Skipping torch compile for this method.")
        # try:
        #     pytorch_method = torch.compile(pytorch_method)
        # except Exception as e:
        #     print(f"Could not compile pytorch_method: {e}")
            
    for stage in ['warmup', 'timed']:
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        
        for i in range(num_iters):
            x, k = get_fft_inputs(dt, b, h, n, dv)
            
            try:

                if method_str == "conv_torch":
                    torch.cuda.synchronize()
                    start_events[i].record() 
                    y = pytorch_method(x, k, n)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "conv_tk":
                    tk_fft = TKFFTConv(k, seqlen=n, H=(h*dv), dtype=x.dtype).to(x.device)
                    torch.cuda.synchronize()
                    start_events[i].record()
                    with torch.no_grad():
                        y = tk_fft(x, k)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "conv_flashfft":
                    conv_flashfft = FlashFFTConv(n, dtype=u.dtype).to(u.device)
                    torch.cuda.synchronize()
                    start_events[i].record() 
                    with torch.no_grad():
                        y = conv_flashfft(x, k)
                    end_events[i].record()
                    torch.cuda.synchronize()

                else:
                    assert 0, f"Unknown method: {method_str}"

            except Exception as e:
                return None, -1

            torch.cuda.empty_cache()

    tot = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)])/num_iters
    assert not np.isnan(y.float().cpu()).any(), "NaN values detected in output 'y'"
    assert not np.isinf(y.float().cpu()).any(), "Inf values detected in output 'y'"
    return y, tot


IMPLEMENTATIONS = {
    "conv_torch": partial(fftconv_tk_test, causal=True, is_forwards=True, method_str="conv_torch"),
    "conv_tk": partial(fftconv_tk_test, causal=True, is_forwards=True, method_str="conv_tk"),
    "conv_flashfft": partial(fftconv_tk_test, causal=True, is_forwards=True, method_str="conv_flashfft"),
}

NAME = "FFTCONV"

