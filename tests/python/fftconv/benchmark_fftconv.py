import torch
import math
import sys
from flashfftconv import FlashFFTConv
import time
from prettytable import PrettyTable, PLAIN_COLUMNS, MSWORD_FRIENDLY

torch.set_grad_enabled(False)

def fft_matrix(N):
    n = torch.arange(N)
    k = n.view(-1, 1)
    M = torch.exp(-2j * torch.pi * n * k / N)
    return M

def compute_twiddle_factors_fft(n, m):
    """Compute the twiddle factors of size n x m"""
    n_a = torch.arange(n).view(-1, 1)
    m_a = torch.arange(m)
    N = n * m
    M = torch.exp(-2j * torch.pi * n_a * m_a / N)
    return M

def ifft_matrix(N):
    n = torch.arange(N)
    k = n.view(-1, 1)
    M = torch.exp(2j * torch.pi * n * k / N)
    return M

def compute_twiddle_factors_ifft(n, m):
    """Compute the twiddle factors of size n x m"""
    n_a = torch.arange(n).view(-1, 1)
    m_a = torch.arange(m)
    N = n * m
    M = torch.exp(2j * torch.pi * n_a * m_a / N)
    return M

class TKFFTConv(torch.nn.Module):
    def __init__(self, seqlen=1024, dtype=torch.bfloat16, use_h100=True):
        super().__init__()
        assert dtype == torch.bfloat16

        self.N = seqlen
        self.N1 = int(math.sqrt(seqlen))
        assert self.N1 == 32
        self.dtype = dtype
        
        f_mat = fft_matrix(self.N1)
        f_real = f_mat.real.to(dtype).contiguous()
        f_imag = f_mat.imag.to(dtype).contiguous()
        self.register_buffer('f_real', f_real)
        self.register_buffer('f_imag', f_imag)

        tw = compute_twiddle_factors_fft(self.N1, self.N1) / self.N
        tw_real = tw.real.to(dtype).contiguous()
        tw_imag = tw.imag.to(dtype).contiguous()
        self.register_buffer('tw_real', tw_real)
        self.register_buffer('tw_imag', tw_imag)

        finv_mat = ifft_matrix(self.N1)
        finv_real = finv_mat.real.to(dtype).contiguous()
        finv_imag = finv_mat.imag.to(dtype).contiguous()
        self.register_buffer('finv_real', finv_real)
        self.register_buffer('finv_imag', finv_imag)

        twinv = compute_twiddle_factors_ifft(self.N1, self.N1)
        twinv_real = twinv.real.to(dtype).contiguous()
        twinv_imag = twinv.imag.to(dtype).contiguous()
        self.register_buffer('twinv_real', twinv_real)
        self.register_buffer('twinv_imag', twinv_imag)
    
    def forward(self, u, k):
        B, H, L = u.shape

        if (H < 768 and L < 1024):
            B_TILE = 1
            H_TILE = 1
        else:
            B_TILE = 8
            H_TILE = 8

        assert L == self.N

        k_f = torch.fft.fft(k.float(), n=self.N)
        u_real = u.real.reshape(B, H, self.N1, self.N1).to(self.dtype).contiguous()
        u_imag = torch.zeros_like(u_real, dtype=self.dtype)
        k_fT = k_f.reshape(H, self.N1, self.N1).transpose(-1, -2).contiguous()
        kfT_real = k_fT.real.to(self.dtype).contiguous()
        kfT_imag = k_fT.imag.to(self.dtype).contiguous()

        out = mod.fftconv_tk(
            u_real, u_imag, kfT_real, kfT_imag, 
            self.f_real, self.f_imag, self.finv_real, self.finv_imag,
            self.tw_real, self.tw_imag, self.twinv_real, self.twinv_imag,
            B, H, self.N, self.N1, B_TILE, H_TILE
        )
        return out.reshape(B, H, self.N).contiguous()


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
repeats = 1
torch.random.manual_seed(42)
dtype = torch.bfloat16
device = 'cuda'

results = PrettyTable()
results.set_style(PLAIN_COLUMNS)
results.field_names = [
    "B", "H", "L", 
    "Torch (ms)", "FlashFFT (ms)", 
    # "TK (ms)", 
    "TK 4090 (ms)", 
    "FlashFFT x", 
    # "TK x", 
    "TK 4090 x"
]
print(f"{len(results.field_names)}")

for b in [B // 2, B]:
    for h in [H // 4, H // 2, H]:
        # Only implemented seqlen 1024 in TK for now
        for seqlen in [1024]:
            torch.cuda.empty_cache()

            N = seqlen
            u = torch.randn((b, h, N), dtype=dtype).to(device)
            # Needs to be fp32 so we can take its FFT
            k = torch.randn((h, N), dtype=torch.float32).to(device)
            # conv_tk_H100 = TKFFTConv(seqlen, dtype=dtype).to(device)
            conv_tk_4090 = TKFFTConv(seqlen, dtype=dtype, use_h100=False).to(device)
            conv_flashfft = FlashFFTConv(seqlen, dtype=dtype).to(device)
            
            y_torch = ref_fftconv(u, k, N)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeats):
                y_torch = ref_fftconv(u, k, N)
            torch.cuda.synchronize()
            torch_time = (time.time() - start)*1e3/repeats

            y_flashfft = conv_flashfft(u, k)
            # test_correctness(y_flashfft, y_torch)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeats):
                y_cuda = conv_flashfft(u, k)
            torch.cuda.synchronize()
            flashfft_time = (time.time() - start)*1e3/repeats

            # y_tk = conv_tk_H100(u, k)
            # test_correctness(y_tk, y_torch)
            # torch.cuda.synchronize()
            # start = time.time()
            # for _ in range(repeats):
            #     y_tk = conv_tk_H100(u, k)
            # torch.cuda.synchronize()
            # tk_time = (time.time() - start)*1e3/repeats


            y_tk_4090 = conv_tk_4090(u, k)
            # test_correctness(y_tk_4090, y_torch)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(repeats):
                y_tk_4090 = conv_tk_4090(u, k)
            torch.cuda.synchronize()
            tk_time_4090 = (time.time() - start)*1e3/repeats

            # breakpoint()
            print(f"{y_tk_4090[0, 0, :4]}")
            # # print(f"{y_tk[0, 0, :4]}")
            print(f"{y_cuda[0, 0, :4]}")
            print(f"{y_torch[0, 0, :4]}")

            flashfft_speedup = torch_time / flashfft_time
            # tk_speedup = torch_time / tk_time
            tk_speedup_4090 = torch_time / tk_time_4090

            results.add_row([
                b, h, seqlen, torch_time, flashfft_time, 
                # tk_time, 
                tk_time_4090, 
                flashfft_speedup, 
                # tk_speedup, 
                tk_speedup_4090
            ])
        # break
    # break

results.float_format = "0.2"
print(results)