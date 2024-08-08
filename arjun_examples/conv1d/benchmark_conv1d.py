# Compare torch, FlashFFTConv, and TK Conv1d kernels

import torch 
import time
from torch import nn

from prettytable import PrettyTable

import sys
sys.path.append("/module")
import TKDepthWiseConv1d

#correctness test
def test_correctness(x, y, atol=1e-1):
    assert torch.allclose(x, y, atol=atol), f"Expected {x} to equal {y}"

torch.manual_seed(42)

dtype = torch.bfloat16
nbytes = 2
device = "cuda"

torch.set_default_device(device)
torch.set_default_dtype(dtype)
   

repeats = 100


results = PrettyTable()
results.field_names = ["B", "L", "D", "K", "torch time (ms)", "cudatime (ms)", "speedup", "Effective bandwidth (GB/s)", "TFLOPS"]
#===================================================================================================
#                            BHL
#===================================================================================================
print("======================================BHL======================================")
for b in [16]:
    for k in [5]:
        for l in [1024, 2048, 4096, 8192]:
            for d in [768, 1024, 2048, 8192]:
                padding =  (k -1)//2
                
                x = torch.randn([b, d, l])
                
                conv1d_torch = nn.Conv1d(
                    in_channels = d,
                    out_channels = d,
                    kernel_size = k,
                    groups = d,
                    padding = padding,
                    dtype = dtype
                )
                
                conv1d_cuda = FlashDepthWiseConv1d(channels = d,
                                                kernel_size=k,
                                                padding=padding,
                                                weights=conv1d_torch.weight,
                                                bias=conv1d_torch.bias,
                                                dtype = dtype
                                                )

                conv1d_tk = TKDepthWiseConv1d(channels = d,
                                            kernel_size=k,
                                            padding=padding,
                                            weights=conv1d_torch.weight,
                                            bias=conv1d_torch.bias,
                                            dtype = dtype
                                            )
                
                
                y_torch = conv1d_torch(x)
                
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeats):
                    y_torch = conv1d_torch(x)
                torch.cuda.synchronize()
                torch_time = (time.time() - start)*1000/repeats
                
                y_cuda = conv1d_cuda(x)
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeats):
                    y_cuda = conv1d_cuda(x)
                torch.cuda.synchronize()
                cuda_time = (time.time() - start)*1000/repeats

                y_tk = conv1d_tk(x)
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(repeats):
                    y_tk = conv1d_tk(x)
                torch.cuda.synchronize()
                tk_time = (time.time() - start)*1000/repeats

                test_correctness(y_torch, y_tk)
                speedup = torch_time / tk_time
                effective_bandwidth = (b * l * d * 2 * nbytes + k * d * nbytes) / (tk_time * 1e-3) / (2**30)
                l_out = l + 2 * padding - k + 1
                tera_flops = (b * l_out * d * 2 * k) / (tk_time * 1e-3) / (2**40)
                results.add_row([b, l, d, k, torch_time, tk_time, speedup, effective_bandwidth, tera_flops])
    
    results.float_format = "0.2"
    print(results)