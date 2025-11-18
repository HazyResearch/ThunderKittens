import torch
import numpy as np
import thunderkittens as tk


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

def pytorch_test_1024(u, k, TESTNAME='all'):
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
    
    u_real = torch.cat(u_reals_reshaped, dim=0)
    u_imag = torch.cat(u_imags_reshaped, dim=0)

    return u_real, u_imag, k_fT_64x64_real, k_fT_64x64_imag, f_sqrt_64x64_real, f_sqrt_64x64_imag, f_inv_sqrt_64x64_real, f_inv_sqrt_64x64_imag, tw_real_64x64_real, tw_imag_64x64_imag, twinv_real_64x64_real, twinv_imag_64x64_imag, o_reals


def pytorch_test_4096(u, k, TESTNAME='all'):
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

N = 1024
B = 2
H = 4
N1 = int(np.sqrt(N))

if N == 1024:
    pytorch_test = pytorch_test_1024
elif N == 4096:
    pytorch_test = pytorch_test_4096
else:
    raise ValueError("N must be 1024 or 4096")

u = (torch.randn((B, H, N), dtype=torch.bfloat16, device='cuda')).to(torch.float32) / H
k = (torch.randn((H, N), dtype=torch.bfloat16, device='cuda')).to(torch.float32) / H

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

if N == 1024:
    B = B * 4
    u = torch.complex(u_real.float(), u_imag.float()).reshape(B, H, N1*N1).to('cuda')

out = tk.fftconv(
    u_real, 
    kfT_real, kfT_imag, 
    f_real, f_imag, 
    finv_real, finv_imag,
    tw_real, tw_imag, 
    twinv_real, twinv_imag,
    B, H, N, N1
)
out = out.reshape(B, H, N)
print(f"{u.shape=}, {k.shape=}, {out.shape=}")
ref = ref_fftconv(u, k, N).real.to('cuda')
print(f"{out.shape=}, {ref.shape=}")

# # out is all zero? 
is_all_zero = torch.allclose(out, torch.zeros_like(out), atol=1e-3)
print(f"{is_all_zero=}")    

batch = torch.randint(0, B, (1,))   
channel = torch.randint(0, H, (1,))
print(out[batch, channel, 760:780])
print(ref[batch, channel, 760:780])

diff = out - ref.to(out.dtype)
max_diff = torch.abs(diff).max()
max_diff_pos = torch.argmax(torch.abs(diff))
print(f"max diff: {max_diff}")
print(f"out[{max_diff_pos}] = {out.flatten()[max_diff_pos]}")
print(f"ref[{max_diff_pos}] = {ref.flatten()[max_diff_pos]}")

# write diff to file
diff = diff.float().cpu().numpy()
with open(f'diff_{N}.txt', 'w') as f:
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            for k in range(diff.shape[2]):
                f.write(repr(diff[i, j, k]))
                f.write(' ')


