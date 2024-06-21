import torch
from tqdm import trange
import numpy as np
import sys

import torch.nn.functional as F

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
D_QK = 128
D_VO = 128

# accept arg for N 
N = int(sys.argv[1])

testname = 'randn' if len(sys.argv) < 3 else sys.argv[2]

torch.random.manual_seed(32)
if testname == 'randn':
    q = (torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda'))
    k = (torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda'))
    v = (torch.randn((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')) / 5
    qmap = (torch.randn((D_QK, 64), dtype=torch.bfloat16, device='cuda'))
    kmap = (torch.randn((D_QK, 64), dtype=torch.bfloat16, device='cuda'))
elif testname == 'ones':
    q = (torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
    k = (torch.ones((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
    v = (torch.ones((B, H, N, D_VO), dtype=torch.bfloat16, device='cuda')) / 5
    qmap = (torch.ones((D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
    kmap = (torch.ones((D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
elif testname == 'qk_test':
    q = torch.eye(D_QK).reshape((1,1,D_QK,D_QK)).repeat(B, H, N//D_QK, 1)*10
    q = q.to(dtype=torch.bfloat16, device='cuda')
    k = torch.eye(D_QK).reshape((1,1,D_QK,D_QK)).repeat(B, H, N//D_QK, 1)*10
    k = k.to(dtype=torch.bfloat16, device='cuda')
    v = torch.eye(D_VO).reshape((1,1,D_VO,D_VO)).repeat(B, H, N//D_VO, 1)*10
    v = v.to(dtype=torch.bfloat16, device='cuda')
    qmap = (torch.ones((D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
    kmap = (torch.ones((D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
elif testname == 'v_or':
    q = (torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
    k = (torch.randn((B, H, N, D_QK), dtype=torch.bfloat16, device='cuda')) / 5
    v = (torch.arange(D_VO*2, dtype=torch.bfloat16, device='cuda')/D_VO).reshape((1,1,2,-1)).repeat(B, H, N//2, 1) / 5
    qmap = (torch.ones((D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
    kmap = (torch.ones((D_QK, 64), dtype=torch.bfloat16, device='cuda')) / 20
else:
    print('bad testname')
    sys.exit(0)


def pytorch_softmax_gt(x, map_mat, label=None):

    x = torch.einsum('bhmd,dn->bhmn', x, map_mat)

    if label is not None:
        with open(label, 'w') as f:
            for row in x[0,0,0:256]:
                f.write(' '.join(map(lambda k: f"{k:8.4f}", row[:256].tolist())) + '\n')
    
    x_pos = x
    x_neg = -x
    
    x_pos_max = torch.amax(x_pos, dim=-1, keepdim=True)
    x_neg_max = torch.amax(x_neg, dim=-1, keepdim=True)
    
    x_pos = x_pos - x_pos_max
    x_neg = x_neg - x_neg_max
    
    x_pos_num = torch.exp(x_pos)
    x_pos_den = torch.sum(torch.exp(x_pos), dim=-1, keepdim=True)
    
    x_neg_num = torch.exp(x_neg)
    x_neg_den = torch.sum(torch.exp(x_neg), dim=-1, keepdim=True)
    
    x_pos = x_pos_num / x_pos_den
    x_neg = x_neg_num / x_neg_den
    
    x = torch.cat([x_pos, x_neg], dim=-1).clamp(min=1e-6)
    
    return x

generator_mat = torch.block_diag(*[torch.ones((64,64), device='cuda')]*64) # 4096 x 4096 should be big enough for most porpoises
generator_mat += torch.roll(generator_mat, -64, -1) # this adds the terracing
lin_mask = torch.tril(1-generator_mat)
exp_mask = torch.tril(generator_mat)
if False: # if you want to debug / visualize these masks
    with open('lin_mask.txt', 'w') as f:
        for row in lin_mask[:384]:
            f.write(' '.join(map(lambda k: '#' if k else ',', row[:384].tolist())) + '\n')
    with open('exp_mask.txt', 'w') as f:
        for row in exp_mask[:384]:
            f.write(' '.join(map(lambda k: '#' if k else ',', row[:384].tolist())) + '\n')
exp_mask = 100*exp_mask - 100 # we actually want to effectively subtract infinity instead with this mask, since it should happen pre-exp

def pytorch_test(Q, K, V, Qmap, Kmap, alpha=1.4, beta=0.6):
    
    Qs = pytorch_softmax_gt(Q, Qmap)
    Ks = pytorch_softmax_gt(K, Kmap)
    print(Qs.shape, Q.shape)
    print(Ks.shape, K.shape)
    
    causal = True
    
    a_lin = torch.einsum('bhmd,bhnd->bhmn', Qs, Ks).to(torch.float32)
    a_exp = torch.einsum('bhmd,bhnd->bhmn', Q, K).to(torch.float32)
    # mask
    a_lin *= lin_mask[:a_lin.shape[2], :a_lin.shape[3]] * alpha # zero out unwanted entries
    a_exp += exp_mask[:a_exp.shape[2], :a_exp.shape[3]] # subtract infinity off of unwanted entries
    a_exp -= a_exp.max(dim=-1, keepdim=True).values
    a_exp = torch.exp(a_exp * 0.125) * beta
    with open('a_exp.txt', 'w') as f:
        for row in a_exp[0,0,0:512]:
            f.write(' '.join(map(lambda k: f"{k:8.4f}", row[:512].tolist())) + '\n')
    with open('a_lin.txt', 'w') as f:
        for row in a_lin[0,0,0:512]:
            f.write(' '.join(map(lambda k: f"{k:8.4f}", row[:512].tolist())) + '\n')

    a = a_lin # TODO a_exp + a_lin
    a = (a / (a.sum(dim=-1, keepdim=True)+1e-6)).to(torch.bfloat16) # normalize
    with open('a.txt', 'w') as f:
        for row in a[0,0,0:256]:
            f.write(' '.join(map(lambda k: f"{k:8.4f}", row[:256].tolist())) + '\n')
    
    out = torch.einsum('bhmn,bhnd->bhmd', a, V).to(torch.bfloat16)
    
    kv_state = torch.einsum('bhlf,bhld->bhfd', Ks, V).detach()
    k_state  = Ks.sum(dim=-2, keepdim=True).detach()
    
    return out, kv_state, k_state

# add padding to Q, K, V, and O to make multiple of 64
N = (N + 63) // 64 * 64

q = F.pad(q, (0, 0, 0, N - q.size(2)), value=0)
k = F.pad(k, (0, 0, 0, N - k.size(2)), value=0)
v = F.pad(v, (0, 0, 0, N - v.size(2)), value=0)

o, kv_state, k_state = pytorch_test(q, k, v, qmap, kmap)

print(o[:,:,:,:5])

avg_o = torch.mean(torch.abs(o.to(torch.float32)))
avg_kv = torch.mean(torch.abs(kv_state.to(torch.float32)))
avg_k  = torch.mean(torch.abs(k_state.to(torch.float32)))

print(f"1/100 of Avg mag of o: {avg_o.item()/100}")
print(f"1/100 of Avg mag of kv: {avg_kv.item()/100}")
print(f"1/100 of Avg mag of k: {avg_k.item()/100}")

print("-" * 80)
# print B, H, N, D_QK, D_VO
print(f'B = {B}')
print(f'H = {H}')
print(f'N = {N}')
print(f'D_QK = {D_QK}')
print(f'D_QK = {D_VO}')

print("-" * 80)
# print desc of inputs and outputs
print(f'q: {q.shape} {q.dtype}')
print(f'k: {k.shape} {k.dtype}')
print(f'v: {v.shape} {v.dtype}')
print(f'o: {o.shape} {o.dtype}')
print(f'kv_state: {kv_state.shape} {kv_state.dtype}')
print(f'k_state: {k_state.shape} {k_state.dtype}')

print("-" * 80)
with open(f'{testname}_{N}.txt', 'w') as f:
    qmap = qmap.to(torch.float32).flatten().cpu().numpy()
    kmap = kmap.to(torch.float32).flatten().cpu().numpy()
    qf = q.to(torch.float32).flatten().cpu().numpy()
    kf = k.to(torch.float32).flatten().cpu().numpy()
    vf = v.to(torch.float32).flatten().cpu().numpy()
    of = o.to(torch.float32).flatten().cpu().numpy()
    # kv = kv_state.to(torch.float32).flatten().cpu().numpy()
    # ksf = k_state.to(torch.float32).flatten().cpu().numpy()

    print('ALERT: ALPHA AND BETA ARE NOT 1 IN THIS DEMO')
    
    print('QMAP', qmap.shape, D_QK*64)
    for i in trange(D_QK*64):
        f.write(repr(qmap[i]))
        f.write(' ')
    f.write('\n')
    print('KMAP', kmap.shape, D_QK*64)
    for i in trange(D_QK*64):
        f.write(repr(kmap[i]))
        f.write(' ')
    f.write('\n')
    print('QF', qf.shape, B*H*N*D_QK)
    for i in trange(B*H*N*D_QK):
        f.write(repr(qf[i]))
        f.write(' ')
    f.write('\n')
    print('KF', kf.shape, B*H*N*D_QK)
    for i in trange(B*H*N*D_QK):
        f.write(repr(kf[i]))
        f.write(' ')
    f.write('\n')
    print('VF', vf.shape, B*H*N*D_VO)
    for i in trange(B*H*N*D_VO):
        f.write(repr(vf[i]))
        f.write(' ')
    f.write('\n')
    print('OF', of.shape, B*H*N*D_VO)
    for i in trange(B*H*N*D_VO):
        f.write(repr(of[i]))
        f.write(' ')
    f.write('\n')
    # for i in trange(B*H*2*D*D):
    #     f.write(repr(kv[i]))
    #     f.write(' ')
    # for i in trange(B*H*1*2*D):
    #     f.write(repr(ksf[i]))
    #     f.write(' ')

