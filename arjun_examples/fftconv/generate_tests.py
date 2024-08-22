import torch
from tqdm import trange
import numpy as np
import scipy
import sys
torch.set_grad_enabled(False)

N = 1024
B = 1
H = 1
N1 = int(np.sqrt(N))

TESTNAME = sys.argv[1]

if TESTNAME in ['ones_all']:
    u = (torch.ones((B, H, N), dtype=torch.bfloat16, device='cpu')).to(torch.float32) 
    k = (torch.ones((H, N), dtype=torch.bfloat16, device='cpu')).to(torch.float32)
    
elif TESTNAME in ['randn_all']:
    torch.random.manual_seed(42)
    u = (torch.randn((B, H, N), dtype=torch.bfloat16, device='cpu')).to(torch.float32) 
    k = (torch.randn((H, N), dtype=torch.bfloat16, device='cpu')).to(torch.float32)

else:
    print('Invalid test name')
    sys.exit(0)

def ref_fftconv(u, k, N):
    L = u.shape[-1]
    # First element in the batch for u
    u_f = torch.fft.fft(u[0].float(), n = N)
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
    k_fT = k_f.reshape(H, N1, N1).transpose(-1, -2)[0, :]
    kfT_real = k_fT.real.to(torch.bfloat16).contiguous()
    kfT_imag = k_fT.imag.to(torch.bfloat16).contiguous()

    o_real = ref_fftconv(u, k, N)
    o_real = o_real.unsqueeze(0).reshape(B, H, N1, N1).to(torch.bfloat16)[0, 0, ...].contiguous()

    u_real = u_real.reshape(B, H, N1, N1).to(torch.bfloat16)[0, 0, ...].contiguous()
    u_imag = u_imag.reshape(B, H, N1, N1).to(torch.bfloat16)[0, 0, ...].contiguous()

    return u_real, u_imag, kfT_real, kfT_imag, f_real, f_imag, finv_real, finv_imag, tw_real, tw_imag, twinv_real, twinv_imag, o_real


u_real, u_imag, kfT_real, kfT_imag, f_real, f_imag, finv_real, finv_imag, tw_real, tw_imag, twinv_real, twinv_imag, o_real = pytorch_test(u, k, TESTNAME=TESTNAME)

with open(f'{TESTNAME}.txt', 'w') as f:
    u_real_f = u_real.to(torch.float32).flatten().cpu().numpy()
    u_imag_f = u_imag.to(torch.float32).flatten().cpu().numpy()
    kfT_real_f = kfT_real.to(torch.float32).flatten().cpu().numpy()
    kfT_imag_f = kfT_imag.to(torch.float32).flatten().cpu().numpy()
    f_real_f = f_real.to(torch.float32).flatten().cpu().numpy()
    f_imag_f = f_imag.to(torch.float32).flatten().cpu().numpy()
    finv_real_f = finv_real.to(torch.float32).flatten().cpu().numpy()
    finv_imag_f = finv_imag.to(torch.float32).flatten().cpu().numpy()
    tw_real_f = tw_real.to(torch.float32).flatten().cpu().numpy()
    tw_imag_f = tw_imag.to(torch.float32).flatten().cpu().numpy()
    twinv_real_f = twinv_real.to(torch.float32).flatten().cpu().numpy()
    twinv_imag_f = twinv_imag.to(torch.float32).flatten().cpu().numpy()
    o_real_f = o_real.to(torch.float32).flatten().cpu().numpy()

    for i in trange(u_real_f.shape[0]):
        f.write(repr(u_real_f[i]))
        f.write(' ')
    for i in trange(u_imag_f.shape[0]):
        f.write(repr(u_imag_f[i]))
        f.write(' ')
    for i in trange(kfT_real_f.shape[0]):
        f.write(repr(kfT_real_f[i]))
        f.write(' ')
    for i in trange(kfT_imag_f.shape[0]):
        f.write(repr(kfT_imag_f[i]))
        f.write(' ')
    for i in trange(f_real_f.shape[0]):
        f.write(repr(f_real_f[i]))
        f.write(' ')
    for i in trange(f_imag_f.shape[0]):
        f.write(repr(f_imag_f[i]))
        f.write(' ')
    for i in trange(finv_real_f.shape[0]):
        f.write(repr(finv_real_f[i]))
        f.write(' ')
    for i in trange(finv_imag_f.shape[0]):
        f.write(repr(finv_imag_f[i]))
        f.write(' ')
    for i in trange(tw_real_f.shape[0]):
        f.write(repr(tw_real_f[i]))
        f.write(' ')
    for i in trange(tw_imag_f.shape[0]):
        f.write(repr(tw_imag_f[i]))
        f.write(' ')
    for i in trange(twinv_real_f.shape[0]):
        f.write(repr(twinv_real_f[i]))
        f.write(' ')
    for i in trange(twinv_imag_f.shape[0]):
        f.write(repr(twinv_imag_f[i]))
        f.write(' ')
    for i in trange(o_real_f.shape[0]):
        f.write(repr(o_real_f[i]))
        f.write(' ')
    