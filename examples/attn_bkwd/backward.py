import torch

def backward(q, k, v, o, do, dq, dk, dv):
    
    N = q.size(0)
    D = q.size(1)
    
    softmax_scale = 1.0/((D)**0.5)

    Br = 16
    Tr = (N + Br - 1) // Br
    
    Bc = 16
    Tc = (D + Bc - 1) // Bc
    
    # Q is N x D - divide it into blocks of size Br X D
    q_blk = q.reshape(Tr, Br, D)
    # K is N x D - divide it into blocks of size Bc X D
    k_blk = k.reshape(Tc, Bc, D)
    # V is N x D - divide it into blocks of size Bc X D
    v_blk = v.reshape(Tc, Bc, D)
    
    # O is N x D - divide it into blocks of size Br X D
    o_blk = o.reshape(Tr, Br, D)
    # dO is N x D - divide it into blocks of size Br X D
    do_blk = do.reshape(Tr, Br, D)
    # dq is N x D - divide it into blocks of size Br X D
    dq_blk = dq.reshape(Tr, Br, D)
    # dK is N x D - divide it into blocks of size Bc X D
    dk_blk = dk.reshape(Tc, Bc, D)
    # dV is N x D - divide it into blocks of size Bc X D
    dv_blk = dv.reshape(Tc, Bc, D)  
    
    for kv_idx in range(Tc):
        # load k and v
        k_idx = k_blk[kv_idx, :, :]
        v_idx = v_blk[kv_idx, :, :]
        
        # initialize dK and dV as 0
        dk_blk[kv_idx, :, :] = torch.zeros_like(dk_blk[kv_idx, :, :])
        dv_blk[kv_idx, :, :] = torch.zeros_like(dv_blk[kv_idx, :, :])
        
        for qo_idx in range(Tr):
            # load q, o, dO
            q_idx = q_blk[qo_idx, :, :]
            o_idx = o_blk[qo_idx, :, :]
            do_idx = do_blk[qo_idx, :, :]
            
            s_idx = torch.einsum("bd,cd->bc", q_idx, k_idx)
            numerator = torch.exp(s_idx)
            denominator = torch.exp(s_idx).sum(dim=-1, keepdim=True)
            A = numerator / denominator
            
            v_grad += torch.einsum("bc, bd->cd", A, do_idx)
        
            dl_da = torch.einsum("bd,cd->bc", do_idx, v_idx)
            dl_da = A * (dl_da - (dl_da * A).sum(dim=-1, keepdim=True))
            
            dl_da *= softmax_scale