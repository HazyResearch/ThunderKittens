
import torch
import torch.nn as nn


def get_flops_based(batch, seqlen, headdim, nheads):
    featuredim = 16
    expanded_dim = featuredim * featuredim + featuredim + 1
    f = 2 * batch * seqlen * nheads * expanded_dim # compute feature map on q and k
    f += batch * seqlen * nheads * headdim * expanded_dim # (k * v)
    f += batch * seqlen * nheads * headdim * expanded_dim # (cumsum)
    f += batch * seqlen * nheads * headdim * expanded_dim # (q * (k * v).cumsum)
    f += batch * seqlen * nheads * headdim * expanded_dim # .sum(dim=-1)
    return f


def get_flops_hedgehog(batch, seqlen, headdim, nheads):
    # https://docs.google.com/spreadsheets/d/1sKBZBHMX_ABfu4XNfGgHM8xD67mtnJY_9lAaTGHtKC0/edit?gid=0#gid=0

    featuredim = 64
    headdim = 128

    def get_masks(window_size: int, q_len: int, k_len: int, device ='cpu') -> tuple[torch.Tensor]:
        """
        Return masks for softmax and linear attention terms
        -> 1 is include, 0 is ignore
        """
        import math
        l = window_size
        m = math.ceil(max(q_len, k_len) / window_size)
        mask = torch.block_diag(*[torch.ones((l, l), )] * m)
        mask += torch.roll(mask, -l, -1) # this adds the terracing
        if mask.shape[0] > q_len:
            mask = mask[-q_len:]
        if mask.shape[1] > k_len:
            mask = mask[:, -k_len:]
        mask_swa, mask_lin = torch.tril(mask), torch.tril(1 - mask)

        num_mask_swa = mask_swa.sum().item()
        num_mask_lins = mask_lin.sum().item()
        return num_mask_swa, num_mask_lins
    
    # featurization 
    expanded_dim = featuredim * 2
    lin_flops = 2 * batch * seqlen * nheads * featuredim * headdim # compute feature map on q and k
    lin_flops += batch * seqlen * nheads * headdim * expanded_dim # apply a nonlinearity
    # numerator for linear
    lin_flops += batch * seqlen * nheads * headdim * expanded_dim # (k * v) pointwise
    lin_flops += batch * seqlen * nheads * headdim * expanded_dim # (cumsum) adds
    lin_flops += batch * seqlen * nheads * headdim * expanded_dim # (q * (k * v).cumsum) pointwise
    lin_flops += batch * seqlen * nheads * headdim * expanded_dim # .sum(dim=-1) sum 
    # denominator for linear 
    lin_flops += batch * seqlen * nheads * headdim * expanded_dim # (k) cumsum 
    lin_flops += batch * seqlen * nheads * headdim * expanded_dim # (q * k.cumsum) pointwise

    # attention 
    attn_flops = 4 * batch * seqlen**2 * nheads * headdim // 2 # 2 is for causality
    
    # take proportions
    terraced_window_size = 64
    num_mask_swa, num_mask_lin = get_masks(terraced_window_size, seqlen, seqlen)
    pct_swa = num_mask_swa / (num_mask_swa + num_mask_lin)
    pct_lin = num_mask_lin / (num_mask_swa + num_mask_lin)

    f = (attn_flops * pct_swa) + (lin_flops * pct_lin)
    return f


def get_flops_mamba2(batch_size, seqlen, head_dimension, nheads, chunk_length=64, state_dimension=128, ngroups=1):
    """
    Mamba-2 FLOPS Calculator 
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py 

    ngroups: default for multi-value attention 
    state_dimension: expansion factor for the recurrent state
    """

    flops = 0
    # seqlen = chunk_length * num_chunks
    num_chunks = seqlen // chunk_length

    # construct mask with decays
    # A_cumsum = torch.cumsum(A, dim=-1)
    # L = torch.exp(segsum(A))
    flops += 1 * ( seqlen * nheads ) # cumsum 
    flops += 1 * ( seqlen * nheads * chunk_length ) # segsum
    flops += 1 * ( seqlen * nheads * chunk_length ) # exp

    # compute center blocks 
    # Y_diag  = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, L, X)
    flops += 2 * num_chunks * ngroups * ( chunk_length * chunk_length * state_dimension )  # mult. QK^T
    flops += 1 * num_chunks * nheads  * ( chunk_length * chunk_length )                    # mask with L
    flops += 2 * num_chunks * nheads  * ( chunk_length * chunk_length * head_dimension)    # mult. by V

    #### low-rank factors ###
    # decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
    flops += 1 * ( num_chunks * nheads * chunk_length ) # subtract
    flops += 1 * ( num_chunks * nheads * chunk_length ) # exp

    # Compute the state for each intra-chunk
    # states = torch.einsum("bclhn,bhcl,bclhp->bchpn", B, decay_states, X)
    # -- mult1 = K * rearrange(decay_states, "b h c l -> b c l h").unsqueeze(-1)
    flops += 1 * ( num_chunks * chunk_length *  nheads )
    # -- mult2 = torch.einsum("bclhn,bclhp->bchpn", mult1, V)
    flops += 2 * ( num_chunks * nheads * chunk_length * head_dimension * state_dimension )

    # Compute the inter-chunk SSM recurrence
    # decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    flops += 1 * ( nheads * num_chunks * num_chunks ) # segsum
    flops += 1 * ( nheads * num_chunks * num_chunks ) # exp
    # new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    flops += 1 * ( num_chunks * nheads * head_dimension * state_dimension )

    # Compute state -> output conversion per chunk
    # state_decay_out = torch.exp(A_cumsum)
    flops += 1 * ( nheads * num_chunks * chunk_length ) # exp
    # Y_off = torch.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)
    # -- step 1: C_adjusted, states_adjusted = C.transpose(2,3), states.transpose(3,4) 
    #            intermediate = torch.matmul(C_adjusted, states_adjusted)
    flops += 2 * ( num_chunks * nheads * chunk_length * head_dimension * state_dimension )
    # -- step 2: state_decay_adjusted = state_decay_out.transpose(1, 2)..unsqueeze(-1)
    #            Y_check = (intermediate * state_decay_adjusted).transpose(2,3)
    flops += 1 * ( num_chunks * nheads * chunk_length * head_dimension )

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    # Y = Y_diag+Y_off
    flops += 1 * ( num_chunks * chunk_length * nheads * head_dimension )

    # batchify and return 
    flops = batch_size * flops
    return flops 

def get_flops_attn(batch, seqlen, headdim, nheads):
    return 4*batch*seqlen**2*nheads*headdim//2

def get_flops_fftconv(batch, seqlen, headdim, nheads):
    d_model = headdim * nheads
    import math
    flops = 2 * (10 * seqlen * math.log(seqlen, 2) * d_model * batch)
    return flops 

def get_layernorm_flops(batch, seqlen, headdim, nheads):
    f = batch*seqlen*nheads*headdim # compute the mask for dropout 
    f += batch*seqlen*nheads*headdim # add dropout and residual
    f += batch*seqlen*nheads*headdim # compute the mean
    f += batch*seqlen*nheads*headdim # compute the variance
    f += batch*seqlen*nheads*headdim # subtract mean
    f += batch*seqlen*nheads*headdim # divide by variance
    f += batch*seqlen*nheads*headdim # multiply by norm weight 
    f += batch*seqlen*nheads*headdim # add norm bias
    return f

def get_flops_rotary(batch, seqlen, headdim, nheads):
    f = batch*seqlen*nheads*(headdim//2) # mult with cos 
    f += batch*seqlen*nheads*(headdim//2) # mult with sin
    f += batch*seqlen*nheads*(headdim//2) # add rotated values
    return f * 2 # for q and k  

