import torch
import math
import sys

sys.path.append("../kernel")
import fftconv_tk as mod
# import fftconv_tk_h100 as mod_h100

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

        self.use_h100 = use_h100
    
    def forward(self, u, k):
        B, H, L = u.shape

        if (H < 768 and L < 1024):
            B_TILE = 1
            H_TILE = 1
        else:
            B_TILE = 4
            H_TILE = 4

        # Right now only working with N = 1024
        assert L == self.N

        k_f = torch.fft.fft(k.float(), n=self.N)
        u_real = u.real.reshape(B, H, self.N1, self.N1).to(self.dtype).contiguous()
        u_imag = torch.zeros_like(u_real, dtype=self.dtype)
        k_fT = k_f.reshape(H, self.N1, self.N1).transpose(-1, -2).contiguous()
        kfT_real = k_fT.real.to(self.dtype).contiguous()
        kfT_imag = k_fT.imag.to(self.dtype).contiguous()

        if self.use_h100:
            assert False, print("Not implemented")
            # out = mod_h100.fftconv_tk_h100(
            #     u_real, u_imag, kfT_real, kfT_imag, 
            #     self.f_real, self.f_imag, self.finv_real, self.finv_imag,
            #     self.tw_real, self.tw_imag, self.twinv_real, self.twinv_imag,
            #     B, H, self.N, self.N1
            # )
        else:
            # # check for nans in any of the inputs
            # if torch.isnan(u_real).any() or torch.isnan(u_imag).any() or torch.isnan(kfT_real).any() or torch.isnan(kfT_imag).any():
            #     print("Nans in inputs")
            #     print(u_real)
            #     print(u_imag)
            #     print(kfT_real)
            #     print(kfT_imag)
            #     sys.exit(1)
            out = mod.fftconv_tk(
                u_real, u_imag, kfT_real, kfT_imag, 
                self.f_real, self.f_imag, self.finv_real, self.finv_imag,
                self.tw_real, self.tw_imag, self.twinv_real, self.twinv_imag,
                B, H, self.N, self.N1, B_TILE, H_TILE
            )

        # print(out[0, 0, 0, :1])

        return out.reshape(B, H, self.N).contiguous()

