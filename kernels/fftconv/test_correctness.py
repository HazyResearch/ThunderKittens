def ref_fftconv(u, k, N):
    L = u.shape[-1]
    u_f = torch.fft.fft(u.float(), n = N)
    k_f = torch.fft.fft(k.float(), n = N)
    y_f = u_f * k_f
    y = torch.fft.ifft(y_f, n = N).real[..., :L].to(u.dtype).contiguous()
    return y

def pytorch_test(u, k):
    ############# GET THE INPUTS #############
    f_mat = fft_matrix(N1)
    finv_mat = ifft_matrix(N1)
    
    # Normalization factor to make IFFT exact inverse of FFT
    twiddle_factors_fft = compute_twiddle_factors_fft(N1, N1).to(u.device)
    twiddle_factors_ifft = compute_twiddle_factors_ifft(N1, N1).to(u.device)

    k_f = torch.fft.fft(k, n = N).real

    ############# COMPUTE OUTPUT #############
    # step 1. FFT(U) using FFT matrix
    # compute the FFT
    x = u.reshape(B, H, N1, N1)
    x = x.transpose(-1, -2)
    x = torch.einsum('...i,ij->...j', x, f_mat)
    x = x.transpose(-1, -2)
    x = x * twiddle_factors_fft # (H, sqrt_N, sqrt_N) * (sqrt_N, sqrt_N), pointwise
    x = torch.einsum('...i,ij->...j', x, f_mat)
    # x = x.transpose(-1, -2)

    # pointwise multiplication 
    k_f = k_f.reshape(H, N1, N1).transpose(1,2) # to match the shape of x
    x = x * k_f

    # compute the IFFT
    # x = x.transpose(-1, -2)
    x = x @ finv_mat
    x = x.transpose(-1, -2)
    x = x * twiddle_factors_ifft # (H, sqrt_N, sqrt_N) * (sqrt_N, sqrt_N), pointwise
    x = x @ finv_mat
    x = x.transpose(-1, -2) # necessary to complete the ifft

    x = x.reshape(B, H, N)
    return x