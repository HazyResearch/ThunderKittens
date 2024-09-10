import torch
from flashfftconv import FlashFFTConv
import time
from prettytable import PrettyTable, PLAIN_COLUMNS, MSWORD_FRIENDLY

import sys
sys.path.append("./module")
from fftconv import TKFFTConv
torch.set_grad_enabled(False)

def ref_fftconv(u, k, N):
    L = u.shape[-1]
    u_f = torch.fft.fft(u.float(), n = N)
    k_f = torch.fft.fft(k.float(), n = N)
    y_f = u_f * k_f
    y = torch.fft.ifft(y_f, n = N).real[..., :L].to(u.dtype).contiguous()
    return y


def test_correctness(x, y, atol=1e0):
    assert torch.allclose(x, y, atol=atol), f"Expected {x} to equal {y}"

B = 64
H = 768
repeats = 10
torch.random.manual_seed(42)
dtype = torch.bfloat16
device = 'cuda'

results = PrettyTable()
results.set_style(PLAIN_COLUMNS)
results.field_names = ["B", "H", "L", "Torch (ms)", "FlashFFT (ms)", "TK (ms)", "FlashFFT Speedup", "TK Speedup"]

for b in [B // 2, B]:
    for h in [H // 4, H // 2, H]:
        # Only implemented seqlen 1024 in TK for now
        for seqlen in [1024]:
            torch.cuda.empty_cache()

            N = seqlen
            u = torch.randn((b, h, N), dtype=dtype).to(device)
            # Needs to be fp32 so we can take its FFT
            k = torch.randn((h, N), dtype=torch.float32).to(device)
            conv_tk = TKFFTConv(seqlen, dtype=dtype).to(device)
            conv_flashfft = FlashFFTConv(seqlen, dtype=dtype).to(device)
            
            y_torch = ref_fftconv(u, k, N)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeats):
                y_torch = ref_fftconv(u, k, N)
            torch.cuda.synchronize()
            torch_time = (time.time() - start)*1e3/repeats

            y_flashfft = conv_flashfft(u, k)
            #test_correctness(y_flashfft, y_torch)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeats):
                y_cuda = conv_flashfft(u, k)
            torch.cuda.synchronize()
            flashfft_time = (time.time() - start)*1e3/repeats

            y_tk = conv_tk(u, k)
            #test_correctness(y_tk, y_torch)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeats):
                y_tk = conv_tk(u, k)
            torch.cuda.synchronize()
            tk_time = (time.time() - start)*1e3/repeats

            print(y_tk[0,0,:4])

            flashfft_speedup = torch_time / flashfft_time
            tk_speedup = torch_time / tk_time

            results.add_row([b, h, seqlen, torch_time, flashfft_time, tk_time, flashfft_speedup, tk_speedup])
            #print(f"{b}\t{h}\t{seqlen}\t{torch_time:.2f}\t{flashfft_time:.2f}\t{tk_time:.2f}\t{flashfft_speedup:,.2f}x\t{tk_speedup:,.2f}x")
            #print(f"B: {b}, H: {h}, L: {seqlen}\n\tTorch: {torch_time:.2f} us, FlashFFT: {flashfft_time:.2f} us, TK: {tk_time:.2f} us\n\tFlashFFT Speedup: {flashfft_speedup:,.2f}x, TK Speedup: {tk_speedup:,.2f}x")

results.float_format = "0.2"
print(results)

