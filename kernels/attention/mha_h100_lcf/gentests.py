import torch
import numpy as np
import sys
import math

print("Generating tests. This will take a few minutes")

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 16
H = 16
N = 3072  # Must be multiple of kv_tile rows (192 for D=64, 128 for D=128)
D = 128

softmax_scale = 1 / math.sqrt(D)

TESTNAME = sys.argv[1]

if TESTNAME == 'ones':
    q           = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k           = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v           = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    grad_output = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    q           = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k           = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v           = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    grad_output = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'qk_test':
    q = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    q = q.to(dtype=torch.bfloat16, device='cuda')
    
    k = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    k = k.to(dtype=torch.bfloat16, device='cuda')
    
    v = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    v = v.to(dtype=torch.bfloat16, device='cuda')
    
    grad_output = torch.eye(D).reshape((1,1,D,D)).repeat(B, H, N//D, 1)*10
    grad_output = grad_output.to(dtype=torch.bfloat16, device='cuda')
elif TESTNAME == 'v_orientation':
    q = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    k = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
    v = (torch.arange(D, dtype=torch.bfloat16, device='cuda')/D).reshape((1,1,1,-1)).repeat(B, H, N, 1)
    
    grad_output = torch.ones((B, H, N, D), dtype=torch.bfloat16, device='cuda')
else:
    print('Invalid test name')
    sys.exit(0)

o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

fn = f'{TESTNAME}_{N}_{D}.txt'

# Convert tensors to numpy arrays (BHND layout, no transpose needed)
qf = q.to(torch.float32).flatten().detach().cpu().numpy()
kf = k.to(torch.float32).flatten().detach().cpu().numpy()
vf = v.to(torch.float32).flatten().detach().cpu().numpy()
of = o.to(torch.float32).flatten().detach().cpu().numpy()

with open(fn, 'wb') as f:
    for name, arr in [('Q', qf), ('K', kf), ('V', vf), ('O', of)]:
        print(f"Writing {name}...")
        np.savetxt(f, arr.reshape(1, -1), fmt='%.8g', delimiter=' ', newline=' ')