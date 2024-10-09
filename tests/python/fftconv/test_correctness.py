import numpy as np
import torch

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

def pytorch_test(u, k, TESTNAME='all'):
    u_real = u.to(torch.bfloat16)
    u_imag = torch.zeros_like(u, dtype=torch.bfloat16)
    f_mat = fft_matrix(N1)
    f_real = f_mat.real.to(torch.bfloat16).contiguous()
    f_imag = f_mat.imag.to(torch.bfloat16).contiguous()
    finv_mat = ifft_matrix(N1)
    finv_real = finv_mat.real.to(torch.bfloat16).contiguous()
    finv_imag = finv_mat.imag.to(torch.bfloat16).contiguous()
    # Normalization factor to make IFFT exact inverse of FFT
    tw = compute_twiddle_factors_fft(N1, N1) / N
    tw_real = tw.real.to(torch.bfloat16).contiguous()
    tw_imag = tw.imag.to(torch.bfloat16).contiguous()
    twinv = compute_twiddle_factors_ifft(N1, N1)
    twinv_real = twinv.real.to(torch.bfloat16).contiguous()
    twinv_imag = twinv.imag.to(torch.bfloat16).contiguous()

    # Compute the regular FFT if the seq len isn't 512 or 2048
    k_f = torch.fft.fft(k.float(), n = N)
    k_fT = k_f.reshape(H, N1, N1).transpose(-1, -2)
    kfT_real = k_fT.real.to(torch.bfloat16).contiguous()
    kfT_imag = k_fT.imag.to(torch.bfloat16).contiguous()

    o_real = ref_fftconv(u, k, N)
    o_real = o_real.reshape(B, H, N1, N1).to(torch.bfloat16).contiguous()

    u_real = u_real.reshape(B, H, N1, N1).to(torch.bfloat16).contiguous()
    u_imag = u_imag.reshape(B, H, N1, N1).to(torch.bfloat16).contiguous()

    return u_real, u_imag, kfT_real, kfT_imag, f_real, f_imag, finv_real, finv_imag, tw_real, tw_imag, twinv_real, twinv_imag, o_real


# run test 

N = 4096
B = 16
H = 64
N1 = int(np.sqrt(N))

torch.random.manual_seed(42)
u = (torch.randn((B, H, N), dtype=torch.bfloat16, device='cpu')).to(torch.float32) / H
k = (torch.randn((H, N), dtype=torch.bfloat16, device='cpu')).to(torch.float32) / H

u_real, u_imag, kfT_real, kfT_imag, f_real, f_imag, finv_real, finv_imag, tw_real, tw_imag, twinv_real, twinv_imag, o_real = pytorch_test(u, k)

# put all the inputs on the device
u_real = u_real.to('cuda').contiguous()
u_imag = u_imag.to('cuda').contiguous()
kfT_real = kfT_real.to('cuda').contiguous()
kfT_imag = kfT_imag.to('cuda').contiguous()
f_real = f_real.to('cuda').contiguous()
f_imag = f_imag.to('cuda').contiguous()
finv_real = finv_real.to('cuda').contiguous()
finv_imag = finv_imag.to('cuda').contiguous()
tw_real = tw_real.to('cuda').contiguous()
tw_imag = tw_imag.to('cuda').contiguous()
twinv_real = twinv_real.to('cuda').contiguous()
twinv_imag = twinv_imag.to('cuda').contiguous()

import thunderkittens as tk

B_TILE = 4
H_TILE = 4
out = tk.fftconv(
    u_real, 
    kfT_real, kfT_imag, 
    f_real, f_imag, 
    finv_real, finv_imag,
    tw_real, tw_imag, 
    twinv_real, twinv_imag,
    B, H, N, N1, B_TILE, H_TILE
)
breakpoint()
out = out.reshape(B, H, N)
ref = ref_fftconv(u, k, N).to('cuda')

# out is all zero? 
is_all_zero = torch.allclose(out, torch.zeros_like(out), atol=1e-3)
print(f"{is_all_zero=}")    

print(out[0, 40, :10])
print(ref[0, 40, :10])

correct = torch.allclose(out, ref.to(out.dtype), atol=0.001)
print(correct)




