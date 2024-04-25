import torch
from tqdm import trange
import numpy as np
import sys
from einops import rearrange, repeat
import math

# only generate a single batch/head of data, which makes file loading much faster.
# it does mean we'll have to check batch/head behavior separately later, but that should be much easier to debug.
B = 1
H = 1
N = 1024 if len(sys.argv) <= 2 else int(sys.argv[2])
D = 64 if len(sys.argv) <= 3 else int(sys.argv[3])

softmax_scale = 1 / math.sqrt(D)
def forward(Q, K, V):
    A0 = torch.einsum("bhnd,bhmd->bhnm",Q, K) * softmax_scale
    
    numerator  = torch.exp(A0) 
    denominator = torch.exp(A0).sum(dim=-1, keepdim=True)
    
    A = numerator / denominator
    
    y  = torch.einsum("bhnm,bhmd->bhnd",A,V)
    return y

def backward(Q, K, V, grad_output):
    running_Q_grad = torch.zeros_like(Q)
    running_K_grad = torch.zeros_like(K)
    running_V_grad = torch.zeros_like(V)
    
    A0 = torch.einsum("bhnd,bhmd->bhnm",Q, K) * softmax_scale
    numerator  = torch.exp(A0) 
    denominator = torch.exp(A0).sum(dim=-1, keepdim=True)
    A = numerator / denominator
        
    running_V_grad += torch.einsum("bhnm,bhnd->bhmd", A, grad_output)
    
    dL_dA = torch.einsum("bhnd,bhmd->bhnm", grad_output, V)
    dL_dA0 = A * (dL_dA - (dL_dA * A).sum(dim=-1, keepdim=True))
    dL_dA0 *= softmax_scale 
    
    running_Q_grad += torch.einsum("bhnm,bhmd->bhnd", dL_dA0, K)
    running_K_grad += torch.einsum("bhnm,bhnd->bhmd", dL_dA0, Q)

    return running_Q_grad, running_K_grad, running_V_grad

def backward_blocked(Q, K, V, dO, l_vec, d_vec):
    assert D == 64
    
    # initialize q_grad, k_grad, v_grad
    q_grad = torch.zeros_like(Q)
    k_grad = torch.zeros_like(K)
    v_grad = torch.zeros_like(V)
    
    # divide Q into 64 x 64 blocks
    Q = Q.reshape(B, H, N//64, 64, D)
    q_grad = q_grad.reshape(B, H, N//64, 64, D)
   
    # divide K and V into 16 x 64 blocks
    K = K.reshape(B, H, N//16, 16, D)
    k_grad = k_grad.reshape(B, H, N//16, 16, D)
    
    V = V.reshape(B, H, N//16, 16, D)
    v_grad = v_grad.reshape(B, H, N//16, 16, D)
    
    # divide grad_output into 64 x 64 blocks
    dO = dO.reshape(B, H, N//64, 64, D)
    # divide l_vec and d_vec into 64 x 1 blocks
    l_vec = l_vec.reshape(B, H, N//64, 64, 1) # D = 1, because we sum over D
    d_vec = d_vec.reshape(B, H, N//64, 64, 1) # D = 1, because we sum over D
    
    #### KV then Q
    # q_grads (one for each kv_blk)
    q_grads = torch.zeros((K.shape[2], B, H, N//64, 64, D), dtype=torch.bfloat16, device='cuda')
    
    for kv_blk in range(K.shape[2]):
        # loads
        k = K[:,:,kv_blk,:,:]
        v = V[:,:,kv_blk,:,:]
        
        # initialize k_grad and v_grad in SRAM/register
        k_g  = torch.zeros_like(k_grad[:,:,kv_blk,:,:])
        v_g  = torch.zeros_like(v_grad[:,:,kv_blk,:,:])
        
        for q_blk in range(Q.shape[2]):
            # loads
            q = Q[:,:,q_blk,:,:]
            do = dO[:,:,q_blk,:,:]
            l = l_vec[:,:,q_blk,:,:]
            d = d_vec[:,:,q_blk,:,:]
            
            s = torch.einsum("bhnd,bhmd->bhnm", q, k) * softmax_scale
            p = torch.exp(s - l)
            
            # accumulate in reg/smem
            v_g += torch.einsum("bhnm,bhnd->bhmd", p, do)
            
            dp = torch.einsum("bhnd,bhmd->bhnm", do, v)
            
            temp = dp - d
            ds = torch.mul(p, temp)
            
            # write to HBM
            q_grads[kv_blk, :, :, q_blk, :, :] = torch.einsum("bhnm,bhmd->bhnd", ds, k) * softmax_scale
            # accumulate in reg/smem
            k_g += torch.einsum("bhnm,bhnd->bhmd", ds, q) * softmax_scale
            
        # store
        k_grad[:,:,kv_blk,:,:] = k_g
        v_grad[:,:,kv_blk,:,:] = v_g
    
    # combine q_grads in a reduction-only kernel
    for kv_blk in range(K.shape[2]):
        q_grad += q_grads[kv_blk, :, :, :, :, :]
        
    return q_grad.reshape(B, H, N, D), k_grad.reshape(B, H, N, D), v_grad.reshape(B, H, N, D)
    
    # for q_blk in range(Q.shape[2]):
    #     q  = Q[:,:,q_blk,:,:] 
    #     do = dO[:,:,q_blk,:,:]
    #     l  = l_vec[:,:,q_blk,:,:]
    #     d  = d_vec[:,:,q_blk,:,:]
    #     for kv_blk in range(K.shape[2]):
    #         k = K[:,:,kv_blk,:,:]
    #         v = V[:,:,kv_blk,:,:]

    #         s = torch.einsum("bhnd,bhmd->bhnm", q, k) * softmax_scale
    #         p = torch.exp(s - l)

    #         v_grad[:,:,kv_blk,:,:] += torch.einsum("bhnm,bhnd->bhmd", p, do)

    #         dp = torch.einsum("bhnd,bhmd->bhnm", do, v)

    #         temp = dp - d
    #         ds = torch.mul(p, temp)

    #         q_grad[:,:,q_blk,:,:]  += torch.einsum("bhnm,bhmd->bhnd", ds, k) * softmax_scale
    #         k_grad[:,:,kv_blk,:,:] += torch.einsum("bhnm,bhnd->bhmd", ds, q) * softmax_scale
    
    
    ##### Q then KV
    # divide k_grads and v_grads into unique for each q_blk    
    # k_grads = torch.zeros((Q.shape[2], B, H, N//16, 16, D), dtype=torch.bfloat16, device='cuda')
    # v_grads = torch.zeros((Q.shape[2], B, H, N//16, 16, D), dtype=torch.bfloat16, device='cuda')
    
    # # this for loop is now parallel (wohoohoo!)
    # for q_blk in range(Q.shape[2]):
        
    #     # loads
    #     q  = Q[:,:,q_blk,:,:]
    #     do = dO[:,:,q_blk,:,:]
    #     l  = l_vec[:,:,q_blk,:,:]
    #     d  = d_vec[:,:,q_blk,:,:]
    #     ###
        
    #     # initialize q_grads
    #     qg = torch.zeros_like(q_grad[:,:,q_blk,:,:])
            
    #     for kv_blk in range(K.shape[2]):
    #         # loads
    #         k = K[:,:,kv_blk,:,:]
    #         v = V[:,:,kv_blk,:,:]
    #         ###
            
    #         s = torch.einsum("bhnd,bhmd->bhnm", q, k) * softmax_scale
    #         p = torch.exp(s - l)
            
    #         # write to HBM
    #         v_grads[q_blk, :, :, kv_blk, :, :] = torch.einsum("bhnm,bhnd->bhmd", p, do)
            
    #         dp = torch.einsum("bhnd,bhmd->bhnm", do, v)
            
    #         temp = dp - d
    #         ds = torch.mul(p, temp)
            
    #         qg = qg + torch.einsum("bhnm,bhmd->bhnd", ds, k) * softmax_scale
            
    #         # write to HBM
    #         k_grads[q_blk, :, :, kv_blk, :, :] = torch.einsum("bhnm,bhnd->bhmd", ds, q) * softmax_scale
            
    #     # store 
    #     q_grad[:,:,q_blk,:,:] = qg
    
    # # combine k_grads and v_grads
    # for q_blk in range(Q.shape[2]):
    #     k_grad += k_grads[q_blk, :, :, :, :, :]
    #     v_grad += v_grads[q_blk, :, :, :, :, :]
    
    # return q_grad.reshape(B, H, N, D), k_grad.reshape(B, H, N, D), v_grad.reshape(B, H, N, D)

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

l_vec = torch.einsum("bhnd,bhmd->bhnm", q.clone(), k.clone()) * softmax_scale
max_vec = l_vec.max(dim=-1, keepdim=True).values
l_vec = l_vec - max_vec
l_vec = torch.exp(l_vec)
l_vec = l_vec.sum(dim=-1, keepdim=True)

l_vec = max_vec + torch.log(l_vec)

o = forward(q, k, v)
q_grad, k_grad, v_grad = backward(q, k, v, grad_output)

d_vec = torch.mul(grad_output, o)
d_vec = d_vec.sum(dim=-1, keepdim=True)
q_grad_r, k_grad_r, v_grad_r = backward_blocked(q, k, v, grad_output, l_vec, d_vec)

# check if the results are the same
print(torch.max(torch.abs(q_grad - q_grad_r)))
print(torch.max(torch.abs(k_grad - k_grad_r)))
print(torch.max(torch.abs(v_grad - v_grad_r)))

