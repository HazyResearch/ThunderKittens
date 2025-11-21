import torch
import torch.nn.functional as F
from tqdm import trange
import sys

D_QK = 128
D_VO = 128

def generate_inputs(B, H, N):
    q = torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda') / (D_QK ** 0.5)
    k = torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda') / (D_QK ** 0.5)
    v = torch.randn((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')
    s = torch.rand((H,), dtype=torch.float32, device='cuda')  # s stays float32
    return q, k, v, s

def get_mask(n, slope=1):
    mask = torch.triu(torch.zeros(n, n).float().fill_(float("-inf")), 1)
    for i in range(n):
        x = torch.arange(i + 1)
        y = slope * x
        mask[i, :i + 1] = -torch.flip(y, [0])
    return torch.exp(mask)

def get_full_mask(n, slopes):
    arr = []
    for slope in slopes:
        arr.append(get_mask(n, slope.item()))
    mask = torch.stack(arr, dim=0)
    return mask

def linear_attn(q, k, v, s):
    b, h, n, d = q.shape
    mask = get_full_mask(n, s).to(q.device).to(torch.float32)
    qk = torch.matmul(q, k.transpose(2, 3))
    qk = (qk.to(torch.float32) * mask).to(q.dtype)
    o = torch.matmul(qk, v)
    return o

def save_test_case(q, k, v, s, o, n):
    filename = f'randn_{n}.txt'
    with open(filename, 'w') as f:    
        sf = s.to(torch.float32).flatten().cpu().numpy().tolist()
        qf = q.to(torch.float32).flatten().cpu().numpy().tolist()
        kf = k.to(torch.float32).flatten().cpu().numpy().tolist()
        vf = v.to(torch.float32).flatten().cpu().numpy().tolist()
        of = o.to(torch.float32).flatten().cpu().numpy().tolist()

        for i in trange(len(sf)):
            f.write(repr(sf[i]))
            f.write(' ')

        for i in trange(len(qf)):
            f.write(repr(qf[i]))
            f.write(' ')
            
        for i in trange(len(kf)):
            f.write(repr(kf[i]))
            f.write(' ')

        for i in trange(len(vf)):
            f.write(repr(vf[i]))
            f.write(' ')

        for i in trange(len(of)):
            f.write(repr(of[i]))
            f.write(' ')

def main():
    torch.manual_seed(0)
    
    B, H = 1, 8
    sequence_lengths = [1024]
    
    for N in sequence_lengths:
        print(f"\nGenerating test case for sequence length {N}")
        q, k, v, s = generate_inputs(B, H, N)

        pytorch_out = linear_attn(q, k, v, s)
        # triton_out = lightning_attn(q, k, v, s)
        
        # avg_mag_pytorch = torch.mean(torch.abs(pytorch_out)).item()
        # avg_mag_triton = torch.mean(torch.abs(triton_out)).item()
        # max_diff = torch.max(torch.abs(pytorch_out - triton_out)).item()
        # avg_diff = torch.mean(torch.abs(pytorch_out - triton_out)).item()
        
        # print(f"PyTorch output magnitude: {avg_mag_pytorch}")
        # print(f"Triton  output magnitude: {avg_mag_triton}")
        # print(f"Max     difference between PyTorch and Triton: {max_diff}")
        # print(f"Average difference between PyTorch and Triton: {avg_diff}")
        
        save_test_case(q, k, v, s, pytorch_out, N)
        print(f"Generated random test case for N={N}")

if __name__ == "__main__":
    main()