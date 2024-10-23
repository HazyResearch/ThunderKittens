
import torch 
import torch.nn as nn
# from torchvision.ops import StochasticDepth

import thunderkittens as tk

def run_torch(x, residual, drop_path, dropout, norm, residual_in_fp32=False):
    dropped = dropout(x) #drop_path(dropout(x))
    residual = (residual + dropped ) if residual is not None else dropped
    out = norm(residual.to(dtype=norm.weight.dtype))
    residual = residual.to(torch.float32)
    return out, residual 


def run_tk(x, residual, drop_path, dropout, norm, residual_in_fp32=False):
    x = x.to(dtype=torch.bfloat16)
    residual = residual.to(dtype=torch.bfloat16)
    norm_weight = norm.weight.to(dtype=torch.bfloat16)
    norm_bias = norm.bias.to(dtype=torch.bfloat16)

    out, out_resid = tk.fused_layernorm(
        x, residual, 
        norm_weight, norm_bias, 
        float(dropout.p)
    )

    return out, out_resid 


def run_flash(x, residual, drop_path, dropout, norm, residual_in_fp32=True):
    from layer_norm_triton import layer_norm_fn, RMSNorm

    rowscale = torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype, )
    # drop_path()
    out, residual = layer_norm_fn(
        x,
        norm.weight,
        norm.bias,
        residual=residual,
        eps=norm.eps,
        dropout_p=dropout.p,
        rowscale=rowscale,
        prenorm=True,
        residual_in_fp32=residual_in_fp32,
        is_rms_norm=False
    )
    return out, residual


def run_naive(x, residual, drop_path, dropout, norm, residual_in_fp32=False):
    use_dropout = False
    
    # 1. dropout on x
    if use_dropout: 
        mask = torch.bernoulli(torch.full_like(x, 1 - dropout.p))
        dropped = x * mask / (1 - dropout.p) 
    else:
        dropped = x
    num_zeros = (dropped == 0).sum().item()

    # 2. drop_path 
    pass # ignore for now

    # 3. residual = dropped + residual
    residual_new = ( dropped + residual ) if residual is not None else dropped 

    # 4. norm
    mean = residual_new.mean(dim=-1, keepdim=True)
    var  = residual_new.var(dim=-1, keepdim=True, unbiased=False)
    residual_norm = (residual_new - mean) / torch.sqrt(var + norm.eps)
    x_norm = norm.weight * residual_norm + norm.bias 

    # compare
    if use_dropout:
        dropped_ref = dropout(x)
    else:
        dropped_ref = x
    num_zeros_ref = (dropped_ref == 0).sum().item()
    residual_ref = ( dropped_ref + residual ) if residual is not None else dropped_ref
    x_norm_ref = norm(residual_ref)

    diff = torch.norm(x_norm - x_norm_ref).item()
    print(x_norm[0,0,:16])
    print(x_norm_ref[0,0,:16])
    print(f"Diff custom: {diff}")
    
    print(f"Num zeros: {num_zeros_ref=}, {num_zeros=}")


if __name__ == "__main__":

    b, n, d = 16, 32, 1024
    p = 0.0
    p_path = 0.00

    torch.manual_seed(0)
    x = torch.randn((b, n, d), device='cuda')
    residual = torch.randn((b, n, d), device='cuda')

    # manual impl.
    norm = nn.LayerNorm(d).cuda()
    dropout = nn.Dropout(p)
    drop_path = None #StochasticDepth(p_path, mode="row")
    run_naive(x, residual, drop_path, dropout, norm)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    norm = nn.LayerNorm(d).cuda()
    dropout = nn.Dropout(p)
    drop_path = None #StochasticDepth(p_path, mode="row")
    out, resid = run_torch(x, residual, drop_path, dropout, norm)

    outs = []
    resids = []
    for fn in [run_tk, run_flash]:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        norm = nn.LayerNorm(d).cuda()
        dropout = nn.Dropout(p)
        drop_path = None #StochasticDepth(p_path, mode="row")
        fn_out, fn_resid = fn(x, residual, drop_path, dropout, norm)

        print("----"*10)
        diff = torch.norm(out - fn_out).item()
        print(out[2,4,:8])
        print(fn_out[2,4,:8])
        print(f"Out Diff: {diff}")

        diff = torch.norm(resid - fn_resid).item()
        print(resid[4,2,:8])
        print(fn_resid[4,2,:8])
        print(f"Resid Diff: {diff}")


