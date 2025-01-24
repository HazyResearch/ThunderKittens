import torch
from tqdm import trange
import numpy as np
import sys
import math

# pip install "grouped-query-attention-pytorch @ git+ssh://git@github.com/fkodom/grouped-query-attention-pytorch.git"
from grouped_query_attention_pytorch.attention import scaled_dot_product_gqa

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
N = int(sys.argv[1])
D = int(sys.argv[2])

H_QO = int(sys.argv[3])
H_KV = int(sys.argv[4])

causal = False

torch.random.manual_seed(42)
q = (torch.randn((B, H_QO, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()
k = (torch.randn((B, H_KV, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()
v = (torch.randn((B, H_KV, N, D), dtype=torch.bfloat16, device='cuda')).requires_grad_()
grad_output = (torch.randn((B, H_QO, N, D), dtype=torch.bfloat16, device='cuda'))

# pad seqlen to multiple of 128
o, _ = scaled_dot_product_gqa(
    q.permute(0, 2, 1, 3).contiguous(),
    k.permute(0, 2, 1, 3).contiguous(),
    v.permute(0, 2, 1, 3).contiguous(),
    is_causal=causal,
    need_weights=False,
)
o = o.permute(0, 2, 1, 3).contiguous()

##########################################
### EXACT GQA COMPUTATION FROM LLAMA 3 ###
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


keys_l = repeat_kv(k.permute(0, 2, 1, 3), H_QO // H_KV).permute(0, 2, 1, 3)
values_l = repeat_kv(v.permute(0, 2, 1, 3), H_QO // H_KV).permute(0, 2, 1, 3)

scores = torch.matmul(q, keys_l.transpose(2, 3)) / math.sqrt(D)

mask = torch.full((N, N), float('-inf'), device=q.device, dtype=q.dtype)
mask = torch.triu(mask, diagonal=1)
if mask is not None and causal:
    scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
scores = torch.nn.functional.softmax(scores, dim=-1).type_as(q)
output = torch.matmul(scores, values_l)  # (bs, n_local_heads, seqlen, head_dim)
### EXACT GQA COMPUTATION FROM LLAMA 3 ###
##########################################

# now do backwards computations
o.backward(grad_output)

q_grad = q.grad
k_grad = k.grad
v_grad = v.grad
    
softmax_scale = 1 / math.sqrt(D)
l_vec = torch.empty((B, H_QO, N, N), dtype=torch.bfloat16, device=q.device)

for i in range(H_QO):
    group_idx = i // (H_QO // H_KV)
    l_vec[:, i] = torch.einsum("bnd,bmd->bnm", q[:, i], k[:, group_idx]) * softmax_scale

mask = torch.triu(torch.ones(N, N), diagonal=1).to('cuda').bool().unsqueeze(0).unsqueeze(0).expand(B, H_QO, -1, -1)
if causal:
    l_vec = l_vec.masked_fill(mask, float('-inf'))

max_vec = l_vec.max(dim=-1, keepdim=True).values
l_vec = l_vec - max_vec
l_vec = torch.exp(l_vec)
l_vec_sum = l_vec.sum(dim=-1, keepdim=True)
l_vec = max_vec + torch.log(l_vec_sum)

d_vec = torch.mul(o.to(torch.bfloat16), grad_output.to(torch.bfloat16))
d_vec = d_vec.to(torch.bfloat16).sum(dim=-1, keepdim=True)


print("--------------------------------------")
print("Q shape: ",      q.shape)
print("K shape: ",      k.shape)
print("V shape: ",      v.shape)
print("O shape: ",      o.shape)
print("Q grad shape: ", q_grad.shape)
print("K grad shape: ", k_grad.shape)
print("V grad shape: ", v_grad.shape)
print("L shape: ",      l_vec.shape)
print("D shape: ",      d_vec.shape)
print("--------------------------------------")

# print out avg magnitude of output tensor
print(f'Average magnitude of OUTPUT tensor: {o.abs().mean()}')
print(f'1/100 magnitude of OUTPUT tensor:   {o.abs().mean()/100}')
print(f'Average magnitude of Q_GRAD tensor: {q_grad.abs().mean()}')
print(f'1/100 magnitude of Q_GRAD tensor:   {q_grad.abs().mean()/100}')
print(f'Average magnitude of K_GRAD tensor: {k_grad.abs().mean()}')
print(f'1/100 magnitude of K_GRAD tensor:   {k_grad.abs().mean()/100}')
print(f'Average magnitude of V_GRAD tensor: {v_grad.abs().mean()}')
print(f'1/100 magnitude of V_GRAD tensor:   {v_grad.abs().mean()/100}')
print(f'Average magnitude of L tensor:      {l_vec.abs().mean()}')
print(f'1/100 magnitude of L tensor:        {l_vec.abs().mean()/100}')
print(f'Average magnitude of D tensor:      {d_vec.abs().mean()}')
print(f'1/100 magnitude of D tensor:        {d_vec.abs().mean()/100}')
print("--------------------------------------")

filename = f'randn_{N}N_{D}D_{H_QO}QO_{H_KV}KV'

if causal:
    filename += '_causal'
if H_QO != H_KV:
    filename += '_gqa'

filename += '.txt'

with open(filename, 'w') as f:
    # inputs
    qf = q.to(torch.float32).flatten().detach().cpu().numpy()
    kf = k.to(torch.float32).flatten().detach().cpu().numpy()
    vf = v.to(torch.float32).flatten().detach().cpu().numpy()
    of = o.to(torch.float32).flatten().detach().cpu().numpy()
    
    og_f = grad_output.to(torch.float32).flatten().detach().cpu().numpy()
    
    # intermediate
    l_vecf = l_vec.to(torch.float32).flatten().detach().cpu().numpy()
    d_vecf = d_vec.to(torch.float32).flatten().detach().cpu().numpy()
    
    qg_f = q_grad.to(torch.float32).flatten().detach().cpu().numpy()
    kg_f = k_grad.to(torch.float32).flatten().detach().cpu().numpy()
    vg_f = v_grad.to(torch.float32).flatten().detach().cpu().numpy()
    
    for i in trange(q.shape[0] * q.shape[1] * q.shape[2] * q.shape[3]):
        f.write(repr(float(qf[i])))
        f.write(' ')
    for i in trange(k.shape[0] * k.shape[1] * k.shape[2] * k.shape[3]):
        f.write(repr(float(kf[i])))
        f.write(' ')
    for i in trange(v.shape[0] * v.shape[1] * v.shape[2] * v.shape[3]):
        f.write(repr(float(vf[i])))
        f.write(' ')
    for i in trange(o.shape[0] * o.shape[1] * o.shape[2] * o.shape[3]):
        f.write(repr(float(of[i])))
        f.write(' ')
    for i in trange(l_vec.shape[0] * l_vec.shape[1] * l_vec.shape[2]):
        f.write(repr(float(l_vecf[i])))
        f.write(' ')
    for i in trange(d_vec.shape[0] * d_vec.shape[1] * d_vec.shape[2]):
        f.write(repr(float(d_vecf[i])))
        f.write(' ')
    for i in trange(grad_output.shape[0] * grad_output.shape[1] * grad_output.shape[2] * grad_output.shape[3]):
        f.write(repr(float(og_f[i])))
        f.write(' ')
    for i in trange(q_grad.shape[0] * q_grad.shape[1] * q_grad.shape[2] * q_grad.shape[3]):
        f.write(repr(float(qg_f[i])))
        f.write(' ')
    for i in trange(k_grad.shape[0] * k_grad.shape[1] * k_grad.shape[2] * k_grad.shape[3]):
        f.write(repr(float(kg_f[i])))
        f.write(' ')
    for i in trange(v_grad.shape[0] * v_grad.shape[1] * v_grad.shape[2] * v_grad.shape[3]):
        f.write(repr(float(vg_f[i])))
        f.write(' ')