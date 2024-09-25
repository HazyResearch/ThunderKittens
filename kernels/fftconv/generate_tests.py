import torch
from tqdm import trange
import numpy as np
import scipy
import sys
torch.set_grad_enabled(False)

N = 32 * 32 * 32
B = 1
H = 1
N1 = 32
N2 = 32

TESTNAME = sys.argv[1]

if TESTNAME in ['ones_all']:
    u = (torch.ones((B, H, N), dtype=torch.cfloat, device='cpu'))
    k = (torch.ones((H, N), dtype=torch.cfloat, device='cpu'))
elif TESTNAME in ['randn_all']:
    torch.random.manual_seed(41)
    u = (torch.randn((B, H, N), dtype=torch.cfloat, device='cpu')) 
    k = (torch.randn((H, N), dtype=torch.cfloat, device='cpu')) 
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


############ GENERATE KERNEL INPUTS ############


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


############## REFERENCE FOR KERNEL LOGIC ##############


def monarch_conv(x, k_f, f_sqrt_N_fft, twiddle_factors_fft, f_sqrt_N_ifft, twiddle_factors_ifft, N, sqrt_N):
    '''
    x: (B, H, N)
    k_f: (H, N)
    f_sqrt_N_fft: (sqrt_N, sqrt_N)
    twiddle_factors_fft: (sqrt_N, sqrt_N)
    f_sqrt_N_ifft: (sqrt_N, sqrt_N)
    twiddle_factors_ifft: (sqrt_N, sqrt_N)
    N: 16K
    sqrt_N: 32
    '''
    B, H, _, N = x.shape

    # compute the FFT
    x = x.reshape(B, H, _, sqrt_N, sqrt_N)
    x = x.transpose(-1, -2)
    x = x @ f_sqrt_N_fft
    x = x.transpose(-1, -2)
    x = x * twiddle_factors_fft # (H, sqrt_N, sqrt_N) * (sqrt_N, sqrt_N), pointwise
    x = x @ f_sqrt_N_fft
    # x = x.transpose(-1, -2)

    # pointwise multiplication 
    k_f = k_f.reshape(H, _, sqrt_N, sqrt_N) # to match the shape of x
    x = x * k_f

    # compute the IFFT
    # x = x.transpose(-1, -2)
    x = x @ f_sqrt_N_ifft
    x = x.transpose(-1, -2)
    x = x * twiddle_factors_ifft # (H, sqrt_N, sqrt_N) * (sqrt_N, sqrt_N), pointwise
    x = x @ f_sqrt_N_ifft
    x = x.transpose(-1, -2) # necessary to complete the ifft

    x = x.reshape(B, H, _, N)
    return x

def monarch_conv_full(
    x, k_f_permuted, 
    f_32_fft,  
    twiddle_factors_fft_32_1K, twiddle_factors_fft_32_32,
    f_32_ifft, 
    twiddle_factors_ifft_32_1K, twiddle_factors_ifft_32_32,
    N, sqrt_N_32, N_1024
):
    x = x.reshape(B, H, sqrt_N_32, N_1024)
    x = x.transpose(-1, -2) # ... 1K, 16
    x = x @ f_32_fft    # ... 1K, 16
    x = x.transpose(-1, -2) # ... 16, 1K
    x = x * twiddle_factors_fft_32_1K # (H, sqrt_N, 1K) * (16, 1K), pointwise
    # x = x @ f_256_fft       # ... 16, 1K

    x = monarch_conv(
            x, k_f_permuted, f_32_fft, twiddle_factors_fft_32_32,
            f_32_ifft, twiddle_factors_ifft_32_32, N_1024, 32)

    x = x * twiddle_factors_ifft_32_1K # ... 16, 1K
    x = x.transpose(-1, -2) # ... 1K, 16
    x = x @ f_32_ifft   # ... 1K, 16
    x = x.transpose(-1, -2) # ... 16, 1K
    return x


################## GENERATE OUTPUTS ##################


def pytorch_test(u, k, TESTNAME='all'):
    u = u.reshape(B, H, 32, 1024)
    for i in range(1024):
        u[:, :, :, i] = torch.arange(32) + 1
    # for i in range(32):
    #     u[:, :, i,:] = torch.arange(1024)
    # u = u.reshape(B, H, N).to(torch.cfloat)

    # input
    u_real = u.to(torch.bfloat16).contiguous()
    u_imag = torch.zeros_like(u, dtype=torch.bfloat16).contiguous()

    # fft matrix
    f_mat = fft_matrix(N1)
    f_real = f_mat.real.to(torch.bfloat16).contiguous()
    f_imag = f_mat.imag.to(torch.bfloat16).contiguous()

    # ifft matrix
    finv_mat = ifft_matrix(N1)
    finv_real = finv_mat.real.to(torch.bfloat16).contiguous()
    finv_imag = finv_mat.imag.to(torch.bfloat16).contiguous()

    # twiddles stage 1
    tw_32_1k = compute_twiddle_factors_fft(N1, 1024)
    tw_32_1k_real = tw_32_1k.real.to(torch.bfloat16).contiguous()
    tw_32_1k_imag = tw_32_1k.imag.to(torch.bfloat16).contiguous()
    
    tw_32_1k_inv = compute_twiddle_factors_ifft(N1, 1024) / N
    tw_32_1k_inv_real = tw_32_1k_inv.real.to(torch.bfloat16).contiguous()
    tw_32_1k_inv_imag = tw_32_1k_inv.imag.to(torch.bfloat16).contiguous()

    # twiddles stage 2
    tw_32_32 = compute_twiddle_factors_fft(N2, N2)
    tw_32_32_real = tw_32_32.real.to(torch.bfloat16).contiguous()
    tw_32_32_imag = tw_32_32.imag.to(torch.bfloat16).contiguous()

    tw_32_32_inv = compute_twiddle_factors_ifft(N2, N2)
    tw_32_32_inv_real = tw_32_32_inv.real.to(torch.bfloat16).contiguous()
    tw_32_32_inv_imag = tw_32_32_inv.imag.to(torch.bfloat16).contiguous()

    # filter
    k_f = torch.fft.fft(k.float(), n = N)
    k_f_permuted = k_f.reshape(H, 1024, N1).transpose(-1, -2).reshape(H, N1, N2, N2).transpose(-1, -2).reshape(H, N).contiguous()
    kfT_real = k_f_permuted.real.to(torch.bfloat16).contiguous()
    kfT_imag = k_f_permuted.imag.to(torch.bfloat16).contiguous()

    print(f"{k_f_permuted.shape=}")

    # # check that our inputs to the kernel will be good
    # out_with_kernel_inputs = monarch_conv_full(
    #     u, k_f_permuted, 
    #     f_mat,  
    #     tw_32_1k, tw_32_32,
    #     finv_mat, 
    #     tw_32_1k_inv, tw_32_32_inv,
    #     N, 32, 1024
    # )   # B, H, 32, 1024
    # out_with_kernel_inputs = out_with_kernel_inputs.real.to(torch.bfloat16).contiguous()

    # input reshaped
    # u_real = u_real.reshape(B, H, 32, 1024).to(torch.bfloat16).contiguous()
    # u_imag = u_imag.reshape(B, H, 32, 1024).to(torch.bfloat16).contiguous()

    # verify that the kernel inputs are correct
    # print("\nVerifying:")
    # o_real_ref = ref_fftconv(u, k, N)   # B, H, N 
    # o_real_ref = o_real_ref.reshape(B, H, N1, 1024).to(torch.bfloat16).contiguous()
    # print(torch.allclose(out_with_kernel_inputs, o_real_ref, atol=2))
    # print(f"out_with_kernel_inputs\n{out_with_kernel_inputs[0, 0, 6, 56:60]}")
    # print(f"o_real_ref\n{o_real_ref[0, 0, 6, 56:60]}\n")


    ############# KERNEL INPUTS GENERATED #############

    chunk_size = 32 
    chunks = 32 
    x2 = u #.clone().reshape(B, H, 32, 1024).to(torch.cfloat)
    for i in range(chunks):
        block = x2[:, :, :, i*chunk_size:(i+1)*chunk_size] 
        if i == 0: print(block.real[0][0])
        # block = block.transpose(-1, -2) 
        block = torch.einsum('bhij,jk->bhik', block, f_mat)
        # block = block * f_mat
        # block @ f_mat 
        # block = block.transpose(-1, -2)
        # block = block * tw_32_1k[:, i*chunk_size:(i+1)*chunk_size]
        # if i > 1: 
            # block = torch.zeros_like(block)
        x2[:, :, :, i*chunk_size:(i+1)*chunk_size] = block
        # break

    # print(f_mat.real)


    # print(f"{tw_32_1k[:, i*chunk_size:(i+1)*chunk_size].shape=}")
    # x2 = x2.reshape(B, H, 32, 32, 32)
    # k_f = k_f.reshape(H, 32, 32, 32)
    # chunk_size = 4 
    # chunks = x2.shape[2] // chunk_size
    # for i in range(chunks):
    #     block = x2[:, :, i*chunk_size:(i+1)*chunk_size]
    #     block = block.transpose(-1, -2)
    #     # block = block @ f_mat  # Apply FFT
    #     block = block.transpose(-1, -2)
    #     block = block * tw_32_32
    #     # block = block @ f_mat  # Apply FFT again
    #     x2[:, :, i*chunk_size:(i+1)*chunk_size] = block
        
    #     # breakpoint()
    #     # pointwise multiplication
    #     # x2[:, :, i*chunk_size:(i+1)*chunk_size] = x2[:, :, i*chunk_size:(i+1)*chunk_size] * k_f[:, i*chunk_size:(i+1)*chunk_size]

    #     # Now apply the IFFT
    #     block = x2[:, :, i*chunk_size:(i+1)*chunk_size]
    #     # block = block @ finv_mat  # Apply iFFT
    #     block = block.transpose(-1, -2)
    #     block = block * tw_32_32_inv 
    #     # block = block @ finv_mat
    #     block = block.transpose(-1, -2) 
    #     x2[:, :, i*chunk_size:(i+1)*chunk_size] = block

    # x2 = x2.reshape(B, H, 32, 1024)

    # # compute ifft
    # chunk_size = 32 
    # chunks = 32 
    # for i in range(chunks):
    #     block = x2[:, :, :, i*chunk_size:(i+1)*chunk_size] 
    # #     block = block * tw_32_1k_inv[:, i*chunk_size:(i+1)*chunk_size] 
    #     block = block.transpose(-1, -2) 
    # #     block = block @ finv_mat  
    #     block = block.transpose(-1, -2)
    #     x2[:, :, :, i*chunk_size:(i+1)*chunk_size] = block

    # # print(f"{kfT_real.shape=}")
    # # # try reshape
    kfT_real = kfT_real.reshape(H, 1024, 32)
    kfT_imag = kfT_imag.reshape(H, 1024, 32)

    o_real = x2.real.to(torch.bfloat16).contiguous()

    print(f"\nVerifying chunked code:")
    # print(torch.allclose(o_real, o_real_ref, atol=2))

    return (
        # input and filter
        u_real, u_imag, 
        # kfT_real, kfT_imag, 

        # fft and ifft matrices
        f_real, f_imag,
        # , 
        # finv_real, finv_imag,

        # # twiddle factors stage 1
        # tw_32_1k_real, tw_32_1k_imag,
        # tw_32_1k_inv_real, tw_32_1k_inv_imag,

        # # twiddle factors stage 2
        # tw_32_32_real, tw_32_32_imag,
        # tw_32_32_inv_real, tw_32_32_inv_imag,

        # output
        o_real
    )

(   

    # input and filter
    u_real, u_imag, 
    # kfT_real, kfT_imag, 

    # fft and ifft matrices
    f_real, f_imag, 
    # finv_real, finv_imag, 

    # # twiddle factors stage 1
    # tw_32_1k_real, tw_32_1k_imag,
    # tw_32_1k_inv_real, tw_32_1k_inv_imag,

    # # twiddle factors stage 2
    # tw_32_32_real, tw_32_32_imag,
    # tw_32_32_inv_real, tw_32_32_inv_imag,

    # output
    o_real 

) = pytorch_test(u, k, TESTNAME=TESTNAME)

# print shapes
print("\nPrinting shapes:")
print(f"{u_real.shape=}")
print(f"{u_imag.shape=}")
# print(f"{kfT_real.shape=}")
# print(f"{kfT_imag.shape=}")
print(f"{f_real.shape=}")
print(f"{f_imag.shape=}")
# print(f"{finv_real.shape=}")
# print(f"{finv_imag.shape=}")
# print(f"{tw_32_1k_real.shape=}")
# print(f"{tw_32_1k_imag.shape=}")
# print(f"{tw_32_1k_inv_real.shape=}")
# print(f"{tw_32_1k_inv_imag.shape=}")
# print(f"{tw_32_32_real.shape=}")
# print(f"{tw_32_32_imag.shape=}")
# print(f"{tw_32_32_inv_real.shape=}")
# print(f"{tw_32_32_inv_imag.shape=}")
print(f"{o_real.shape=}")

with open(f'{TESTNAME}.txt', 'w') as f:

    # inputs and filters
    u_real_f = u_real.to(torch.float32).flatten().cpu().numpy()
    u_imag_f = u_imag.to(torch.float32).flatten().cpu().numpy()

    # kfT_real_f = kfT_real.to(torch.float32).flatten().cpu().numpy()
    # kfT_imag_f = kfT_imag.to(torch.float32).flatten().cpu().numpy()

    for i in trange(u_real_f.shape[0]):
        f.write(repr(u_real_f[i]))
        f.write(' ')
    for i in trange(u_imag_f.shape[0]):
        f.write(repr(u_imag_f[i]))
        f.write(' ')

    # for i in trange(kfT_real_f.shape[0]):
    #     f.write(repr(kfT_real_f[i]))
    #     f.write(' ')
    # for i in trange(kfT_imag_f.shape[0]):
    #     f.write(repr(kfT_imag_f[i]))
    #     f.write(' ')

    # fft and ifft matrices
    f_real_f = f_real.to(torch.float32).flatten().cpu().numpy()
    f_imag_f = f_imag.to(torch.float32).flatten().cpu().numpy()

    # finv_real_f = finv_real.to(torch.float32).flatten().cpu().numpy()
    # finv_imag_f = finv_imag.to(torch.float32).flatten().cpu().numpy()

    for i in trange(f_real_f.shape[0]):
        f.write(repr(f_real_f[i]))
        f.write(' ')
    for i in trange(f_imag_f.shape[0]):
        f.write(repr(f_imag_f[i]))
        f.write(' ')

    # for i in trange(finv_real_f.shape[0]):
    #     f.write(repr(finv_real_f[i]))
    #     f.write(' ')
    # for i in trange(finv_imag_f.shape[0]):
    #     f.write(repr(finv_imag_f[i]))
    #     f.write(' ')

    # # twiddle factors stage 1
    # tw_32_1k_real_f = tw_32_1k_real.to(torch.float32).flatten().cpu().numpy()
    # tw_32_1k_imag_f = tw_32_1k_imag.to(torch.float32).flatten().cpu().numpy()

    # tw_32_1k_inv_real_f = tw_32_1k_inv_real.to(torch.float32).flatten().cpu().numpy()
    # tw_32_1k_inv_imag_f = tw_32_1k_inv_imag.to(torch.float32).flatten().cpu().numpy()

    # for i in trange(tw_32_1k_real_f.shape[0]):
    #     f.write(repr(tw_32_1k_real_f[i]))
    #     f.write(' ')
    # for i in trange(tw_32_1k_imag_f.shape[0]):
    #     f.write(repr(tw_32_1k_imag_f[i]))
    #     f.write(' ')

    # for i in trange(tw_32_1k_inv_real_f.shape[0]):
    #     f.write(repr(tw_32_1k_inv_real_f[i]))
    #     f.write(' ')
    # for i in trange(tw_32_1k_inv_imag_f.shape[0]):
    #     f.write(repr(tw_32_1k_inv_imag_f[i]))
    #     f.write(' ')

    # # twiddle factors stage 2
    # tw_32_32_real_f = tw_32_32_real.to(torch.float32).flatten().cpu().numpy()
    # tw_32_32_imag_f = tw_32_32_imag.to(torch.float32).flatten().cpu().numpy()

    # tw_32_32_inv_real_f = tw_32_32_inv_real.to(torch.float32).flatten().cpu().numpy()
    # tw_32_32_inv_imag_f = tw_32_32_inv_imag.to(torch.float32).flatten().cpu().numpy()

    # for i in trange(tw_32_32_real_f.shape[0]):
    #     f.write(repr(tw_32_32_real_f[i]))
    #     f.write(' ')
    # for i in trange(tw_32_32_imag_f.shape[0]):
    #     f.write(repr(tw_32_32_imag_f[i]))
    #     f.write(' ')
        
    # for i in trange(tw_32_32_inv_real_f.shape[0]):
    #     f.write(repr(tw_32_32_inv_real_f[i]))
    #     f.write(' ')
    # for i in trange(tw_32_32_inv_imag_f.shape[0]):
    #     f.write(repr(tw_32_32_inv_imag_f[i]))
    #     f.write(' ')

    # outputs
    o_real_f = o_real.to(torch.float32).flatten().cpu().numpy()
    
    for i in trange(o_real_f.shape[0]):
        f.write(repr(o_real_f[i]))
        f.write(' ')

