import torch
import math
import sys

try:
    from _C import fftconv
except:
    print("Could not import thunderkittens")


def ref_fftconv(u, k, N):
    L = u.shape[-1]
    u_f = torch.fft.fft(u.float(), n = N)
    k_f = torch.fft.fft(k.float(), n = N)
    y_f = u_f * k_f
    y = torch.fft.ifft(y_f, n = N).real[..., :L].to(u.dtype).contiguous()
    return y

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

def generate_1024(
    f_real, f_imag,
    finv_real, finv_imag,
    tw_real, tw_imag,
    twinv_real, twinv_imag,
    kfT_real, kfT_imag,
    H, N1
):
    # fft matrix
    f_sqrt_64x64_real = torch.zeros(64, 64, device=f_real.device).to(f_real.dtype)
    f_sqrt_64x64_real[:N1, :N1] = f_real
    f_sqrt_64x64_real[N1:, N1:] = f_real

    f_sqrt_64x64_imag = torch.zeros(64, 64, device=f_imag.device).to(f_imag.dtype)
    f_sqrt_64x64_imag[N1:, N1:] = f_imag
    f_sqrt_64x64_imag[:N1, :N1] = f_imag

    # ifft matrix
    f_inv_sqrt_64x64_real = torch.zeros(64, 64, device=finv_real.device).to(finv_real.dtype)
    f_inv_sqrt_64x64_real[:N1, :N1] = finv_real
    f_inv_sqrt_64x64_real[N1:, N1:] = finv_real

    f_inv_sqrt_64x64_imag = torch.zeros(64, 64, device=finv_imag.device).to(finv_imag.dtype)
    f_inv_sqrt_64x64_imag[N1:, N1:] = finv_imag
    f_inv_sqrt_64x64_imag[:N1, :N1] = finv_imag

    # Normalization factor to make IFFT exact inverse of FFT
    tw_real_64x64_real = torch.zeros(64, 64, device=tw_real.device).to(tw_real.dtype)
    tw_real_64x64_real[:N1, :N1] = tw_real
    tw_real_64x64_real[N1:, N1:] = tw_real
    tw_real_64x64_real[:N1, N1:] = tw_real
    tw_real_64x64_real[N1:, :N1] = tw_real

    tw_imag_64x64_imag = torch.zeros(64, 64, device=tw_imag.device).to(tw_imag.dtype)
    tw_imag_64x64_imag[:N1, :N1] = tw_imag
    tw_imag_64x64_imag[N1:, N1:] = tw_imag
    tw_imag_64x64_imag[:N1, N1:] = tw_imag
    tw_imag_64x64_imag[N1:, :N1] = tw_imag

    # twiddle inverse
    twinv_real_64x64_real = torch.zeros(64, 64, device=twinv_real.device).to(twinv_real.dtype)
    twinv_real_64x64_real[:N1, :N1] = twinv_real
    twinv_real_64x64_real[N1:, N1:] = twinv_real
    twinv_real_64x64_real[:N1, N1:] = twinv_real
    twinv_real_64x64_real[N1:, :N1] = twinv_real

    twinv_imag_64x64_imag = torch.zeros(64, 64, device=twinv_imag.device).to(twinv_imag.dtype)
    twinv_imag_64x64_imag[:N1, :N1] = twinv_imag
    twinv_imag_64x64_imag[N1:, N1:] = twinv_imag
    twinv_imag_64x64_imag[:N1, N1:] = twinv_imag
    twinv_imag_64x64_imag[N1:, :N1] = twinv_imag

    # Compute the regular FFT if the seq len isn't 512 or 2048
    k_fT_64x64_real = torch.zeros(H, 64, 64, device=kfT_real.device).to(torch.bfloat16)
    k_fT_64x64_real[:, :N1, :N1] = kfT_real.to(k_fT_64x64_real.dtype)
    k_fT_64x64_real[:, N1:, N1:] = kfT_real.to(k_fT_64x64_real.dtype)
    k_fT_64x64_real[:, :N1, N1:] = kfT_real.to(k_fT_64x64_real.dtype)
    k_fT_64x64_real[:, N1:, :N1] = kfT_real.to(k_fT_64x64_real.dtype)

    k_fT_64x64_imag = torch.zeros(H, 64, 64, device=kfT_imag.device).to(torch.bfloat16)
    k_fT_64x64_imag[:, :N1, :N1] = kfT_imag.to(k_fT_64x64_imag.dtype)
    k_fT_64x64_imag[:, N1:, N1:] = kfT_imag.to(k_fT_64x64_imag.dtype)
    k_fT_64x64_imag[:, :N1, N1:] = kfT_imag.to(k_fT_64x64_imag.dtype)
    k_fT_64x64_imag[:, N1:, :N1] = kfT_imag.to(k_fT_64x64_imag.dtype)
    
    return f_sqrt_64x64_real, f_sqrt_64x64_imag, f_inv_sqrt_64x64_real, f_inv_sqrt_64x64_imag, tw_real_64x64_real, tw_imag_64x64_imag, twinv_real_64x64_real, twinv_imag_64x64_imag, k_fT_64x64_real, k_fT_64x64_imag


class TKFFTConv(torch.nn.Module):
    def __init__(self, k, seqlen=1024, H=128, dtype=torch.bfloat16, use_h100=True):
        super().__init__()
        assert dtype == torch.bfloat16

        self.N = seqlen
        self.H = H
        self.N1 = int(math.sqrt(seqlen))
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

        k_f = torch.fft.fft(k.float(), n=self.N)
        k_fT = k_f.reshape(H, self.N1, self.N1).transpose(-1, -2).contiguous()
        kfT_real = k_fT.real.to(self.dtype).contiguous()
        kfT_imag = k_fT.imag.to(self.dtype).contiguous()
        self.register_buffer('kfT_real', kfT_real)
        self.register_buffer('kfT_imag', kfT_imag)

        if self.N == 1024:
            (
                self.f_real, self.f_imag, 
                self.finv_real, self.finv_imag, 
                self.tw_real, self.tw_imag, 
                self.twinv_real, self.twinv_imag, 
                self.kfT_real, self.kfT_imag
            ) = generate_1024(
                f_real, f_imag,
                finv_real, finv_imag,
                tw_real, tw_imag,
                twinv_real, twinv_imag,
                self.kfT_real, self.kfT_imag,
                H, self.N1
            )
    
    def forward(self, u, k):
        B, H, L = u.shape
        assert L == self.N
        u_real = u.real.reshape(B, H, self.N1, self.N1).to(self.dtype).contiguous()

        out = fftconv(
            u_real, 
            self.kfT_real, self.kfT_imag, 
            self.f_real, self.f_imag, 
            self.finv_real, self.finv_imag,
            self.tw_real, self.tw_imag, 
            self.twinv_real, self.twinv_imag,
            B, self.H, self.N, self.N1
        )
        
        return out.reshape(B, self.H, self.N).contiguous()

        