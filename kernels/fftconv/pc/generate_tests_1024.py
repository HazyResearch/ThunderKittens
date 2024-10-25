import torch
from tqdm import trange
import numpy as np
import sys
torch.set_grad_enabled(False)

N = 1024
B = 4
H = 1024
N1 = int(np.sqrt(N))
N_64 = 64

TESTNAME = sys.argv[1]

if TESTNAME in ['ones']:
    TESTNAME = f'ones_f{N}_H{H}_B{B}'
    u = (torch.ones((B, H, N), dtype=torch.bfloat16, device='cpu')).to(torch.float32) 
    k = (torch.ones((H, N), dtype=torch.bfloat16, device='cpu')).to(torch.float32)
    
elif TESTNAME in ['randn']:
    TESTNAME = f'randn_f{N}_H{H}_B{B}'
    torch.random.manual_seed(42)
    u = (torch.randn((B, H, N), dtype=torch.bfloat16, device='cpu')).to(torch.float32) 
    k = (torch.randn((H, N), dtype=torch.bfloat16, device='cpu')).to(torch.float32)

elif TESTNAME in ['arange']:
    TESTNAME = f'arange_f{N}_H{H}_B{B}'
    u = (torch.arange(B*H*N, dtype=torch.bfloat16, device='cpu')).to(torch.float32).reshape(B, H, N)
    k = (torch.arange(H*N, dtype=torch.bfloat16, device='cpu')).to(torch.float32).reshape(H, N)

else:
    print('Invalid test name')
    sys.exit(0)

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


def our_reference_logic(u_reals, u_imags, k_fT_64x64_real, k_fT_64x64_imag, f_sqrt_64x64_real, f_sqrt_64x64_imag, f_inv_sqrt_64x64_real, f_inv_sqrt_64x64_imag, tw_real_64x64_real, tw_imag_64x64_imag, twinv_real_64x64_real, twinv_imag_64x64_imag):
    # breakpoint()
    B, H, N1, N1 = u_reals[0].shape

    # get the inputs
    x = torch.zeros(B, H, 64, 64).to(u_reals[0].device).to(torch.cfloat)
    x[:, :, :N1, :N1] = u_reals[0] + 1j * u_imags[0]
    x[:, :, :N1, N1:] = u_reals[1] + 1j * u_imags[1]
    x[:, :, N1:, :N1] = u_reals[2] + 1j * u_imags[2]
    x[:, :, N1:, N1:] = u_reals[3] + 1j * u_imags[3]

    f_sqrt_N_fft = torch.zeros(64, 64, device=f_sqrt_64x64_real.device).to(f_sqrt_64x64_real.dtype).to(torch.cfloat)
    f_sqrt_N_fft.real = f_sqrt_64x64_real
    f_sqrt_N_fft.imag = f_sqrt_64x64_imag

    f_sqrt_N_ifft = torch.zeros(64, 64, device=f_inv_sqrt_64x64_real.device).to(f_inv_sqrt_64x64_real.dtype).to(torch.cfloat)
    f_sqrt_N_ifft.real = f_inv_sqrt_64x64_real
    f_sqrt_N_ifft.imag = f_inv_sqrt_64x64_imag

    twiddle_factors_fft = torch.zeros(64, 64, device=tw_real_64x64_real.device).to(tw_real_64x64_real.dtype).to(torch.cfloat)
    twiddle_factors_fft.real = tw_real_64x64_real
    twiddle_factors_fft.imag = tw_imag_64x64_imag

    twiddle_factors_ifft = torch.zeros(64, 64, device=twinv_real_64x64_real.device).to(twinv_real_64x64_real.dtype).to(torch.cfloat)
    twiddle_factors_ifft.real = twinv_real_64x64_real
    twiddle_factors_ifft.imag = twinv_imag_64x64_imag

    k_f = torch.zeros(H, 64, 64, device=k_fT_64x64_real.device).to(k_fT_64x64_real.dtype).to(torch.cfloat)
    k_f.real = k_fT_64x64_real
    k_f.imag = k_fT_64x64_imag

    # run op
    x = x.reshape(B, H, 64, 64)
    x = f_sqrt_N_fft @ x
    
    x = x * twiddle_factors_fft
    x = x @ f_sqrt_N_fft

    k_f = k_f.reshape(H, 64, 64) 
    x = x * k_f

    x = x @ f_sqrt_N_ifft   
    x = x.transpose(-1, -2)
    x = x * twiddle_factors_ifft 

    x = x @ f_sqrt_N_ifft 
    x = x.transpose(-1, -2) 

    our_outputs = []
    out1 = x[:, :, :N1, :N1].reshape(B, H, N1, N1).real.to(torch.bfloat16).contiguous()
    out2 = x[:, :, :N1, N1:].reshape(B, H, N1, N1).real.to(torch.bfloat16).contiguous()
    out3 = x[:, :, N1:, :N1].reshape(B, H, N1, N1).real.to(torch.bfloat16).contiguous()
    out4 = x[:, :, N1:, N1:].reshape(B, H, N1, N1).real.to(torch.bfloat16).contiguous()
    our_outputs.append(out1)
    our_outputs.append(out2)
    our_outputs.append(out3)
    our_outputs.append(out4)
    return our_outputs


def pytorch_test(u, k, TESTNAME='all'):
    u_reals, u_imags = [], []
    u_real = u.to(torch.bfloat16)
    u_imag = torch.zeros_like(u, dtype=torch.bfloat16)
    u_reals.append(u_real)
    u_imags.append(u_imag)
    for i in range(3):
        u_new = torch.randn_like(u)*(i+2)
        u_real = u_new.to(torch.bfloat16)
        u_imag = torch.zeros_like(u, dtype=torch.bfloat16)
        u_reals.append(u_real)
        u_imags.append(u_imag)

    # fft matrix
    f_mat = fft_matrix(N1)
    f_real = f_mat.real.to(torch.bfloat16).contiguous()
    f_imag = f_mat.imag.to(torch.bfloat16).contiguous()

    f_sqrt_64x64_real = torch.zeros(64, 64, device=f_real.device).to(f_real.dtype)
    f_sqrt_64x64_real[:N1, :N1] = f_real
    f_sqrt_64x64_real[N1:, N1:] = f_real

    f_sqrt_64x64_imag = torch.zeros(64, 64, device=f_imag.device).to(f_imag.dtype)
    f_sqrt_64x64_imag[N1:, N1:] = f_imag
    f_sqrt_64x64_imag[:N1, :N1] = f_imag

    # ifft matrix
    finv_mat = ifft_matrix(N1)
    finv_real = finv_mat.real.to(torch.bfloat16).contiguous()
    finv_imag = finv_mat.imag.to(torch.bfloat16).contiguous()

    f_inv_sqrt_64x64_real = torch.zeros(64, 64, device=finv_real.device).to(finv_real.dtype)
    f_inv_sqrt_64x64_real[:N1, :N1] = finv_real
    f_inv_sqrt_64x64_real[N1:, N1:] = finv_real

    f_inv_sqrt_64x64_imag = torch.zeros(64, 64, device=finv_imag.device).to(finv_imag.dtype)
    f_inv_sqrt_64x64_imag[N1:, N1:] = finv_imag
    f_inv_sqrt_64x64_imag[:N1, :N1] = finv_imag

    # Normalization factor to make IFFT exact inverse of FFT
    tw = compute_twiddle_factors_fft(N1, N1) / N
    tw_real = tw.real.to(torch.bfloat16).contiguous()
    tw_imag = tw.imag.to(torch.bfloat16).contiguous()

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
    twinv = compute_twiddle_factors_ifft(N1, N1)
    twinv_real = twinv.real.to(torch.bfloat16).contiguous()
    twinv_imag = twinv.imag.to(torch.bfloat16).contiguous()

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
    k_f = torch.fft.fft(k.float(), n = N)
    k_fT = k_f.reshape(H, N1, N1).transpose(-1, -2)
    kfT_real = k_fT.real.to(torch.bfloat16).contiguous()
    kfT_imag = k_fT.imag.to(torch.bfloat16).contiguous()

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

    o_reals = []
    for i in range(4):
        o_real = ref_fftconv(u_reals[i], k, N)
        o_real = o_real.reshape(B, H, N1, N1).to(torch.bfloat16).contiguous()
        o_reals.append(o_real)

    u_reals_reshaped, u_imags_reshaped = [], []
    for i in range(4):
        u_reals_reshaped.append(u_reals[i].reshape(B, H, N1, N1).to(torch.bfloat16).contiguous())
        u_imags_reshaped.append(u_imags[i].reshape(B, H, N1, N1).to(torch.bfloat16).contiguous())
    
    return u_reals_reshaped, u_imags_reshaped, k_fT_64x64_real, k_fT_64x64_imag, f_sqrt_64x64_real, f_sqrt_64x64_imag, f_inv_sqrt_64x64_real, f_inv_sqrt_64x64_imag, tw_real_64x64_real, tw_imag_64x64_imag, twinv_real_64x64_real, twinv_imag_64x64_imag, o_reals


def check_our_outputs(our_outputs, o_reals):
    for i in range(4):
        all_close = torch.allclose(our_outputs[i], o_reals[i], atol=1e-3, rtol=1e-3)
        max_diff = torch.max(torch.abs(our_outputs[i] - o_reals[i]))
        mean_diff = torch.mean(torch.abs(our_outputs[i] - o_reals[i]))
        avg_value = torch.mean(torch.abs(o_reals[i]))
        print(f'Output {i+1} all close: {all_close}, max diff: {max_diff}, mean diff: {mean_diff}, avg value: {avg_value}')

u_reals_reshaped, u_imags_reshaped, k_fT_64x64_real, k_fT_64x64_imag, f_sqrt_64x64_real, f_sqrt_64x64_imag, f_inv_sqrt_64x64_real, f_inv_sqrt_64x64_imag, tw_real_64x64_real, tw_imag_64x64_imag, twinv_real_64x64_real, twinv_imag_64x64_imag, o_reals = pytorch_test(u, k, TESTNAME)

# run a check with these values
our_outputs = our_reference_logic(u_reals_reshaped, u_imags_reshaped, k_fT_64x64_real, k_fT_64x64_imag, f_sqrt_64x64_real, f_sqrt_64x64_imag, f_inv_sqrt_64x64_real, f_inv_sqrt_64x64_imag, tw_real_64x64_real, tw_imag_64x64_imag, twinv_real_64x64_real, twinv_imag_64x64_imag)
check_our_outputs(our_outputs, o_reals)


with open(f'{TESTNAME}.txt', 'w') as f:
    # u reals 
    for i in range(len(u_reals_reshaped)):
        u_real_f = u_reals_reshaped[i].to(torch.float32).flatten().cpu().numpy()
        for j in trange(u_real_f.shape[0]):
            f.write(repr(u_real_f[j]))
            f.write(' ')
    for i in range(len(u_reals_reshaped)):
        u_imag_f = u_imags_reshaped[i].to(torch.float32).flatten().cpu().numpy()
        for j in trange(u_imag_f.shape[0]):
            f.write(repr(u_imag_f[j]))
            f.write(' ')

    kfT_real_f = k_fT_64x64_real.to(torch.float32).flatten().cpu().numpy()
    kfT_imag_f = k_fT_64x64_imag.to(torch.float32).flatten().cpu().numpy()
    for i in trange(kfT_real_f.shape[0]):
        f.write(repr(kfT_real_f[i]))
        f.write(' ')
    for i in trange(kfT_imag_f.shape[0]):
        f.write(repr(kfT_imag_f[i]))
        f.write(' ')

    f_real_f = f_sqrt_64x64_real.to(torch.float32).flatten().cpu().numpy()
    f_imag_f = f_sqrt_64x64_imag.to(torch.float32).flatten().cpu().numpy()
    for i in trange(f_real_f.shape[0]):
        f.write(repr(f_real_f[i]))
        f.write(' ')
    for i in trange(f_imag_f.shape[0]):
        f.write(repr(f_imag_f[i]))
        f.write(' ')


    finv_real_f = f_inv_sqrt_64x64_real.to(torch.float32).flatten().cpu().numpy()
    finv_imag_f = f_inv_sqrt_64x64_imag.to(torch.float32).flatten().cpu().numpy()
    for i in trange(finv_real_f.shape[0]):
        f.write(repr(finv_real_f[i]))
        f.write(' ')
    for i in trange(finv_imag_f.shape[0]):
        f.write(repr(finv_imag_f[i]))
        f.write(' ')

    tw_real_f = tw_real_64x64_real.to(torch.float32).flatten().cpu().numpy()
    tw_imag_f = tw_imag_64x64_imag.to(torch.float32).flatten().cpu().numpy()
    for i in trange(tw_real_f.shape[0]):
        f.write(repr(tw_real_f[i]))
        f.write(' ')
    for i in trange(tw_imag_f.shape[0]):
        f.write(repr(tw_imag_f[i]))
        f.write(' ')
    
    twinv_real_f = twinv_real_64x64_real.to(torch.float32).flatten().cpu().numpy()
    twinv_imag_f = twinv_imag_64x64_imag.to(torch.float32).flatten().cpu().numpy()
    for i in trange(twinv_real_f.shape[0]):
        f.write(repr(twinv_real_f[i]))
        f.write(' ')
    for i in trange(twinv_imag_f.shape[0]):
        f.write(repr(twinv_imag_f[i]))
        f.write(' ')

    for i in range(len(o_reals)):
        o_real = o_reals[i]
        o_real_f = o_real.to(torch.float32).flatten().cpu().numpy()
        for j in trange(o_real_f.shape[0]):
            f.write(repr(o_real_f[j]))
            f.write(' ')
    