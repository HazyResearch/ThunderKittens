
import torch 
import torch.nn as nn
from torchvision.ops import StochasticDepth

def run_torch(x, drop_path, dropout, norm, residual_in_fp32=False):
    residual = x

    dropped = drop_path(dropout(x))
    residual = (residual + dropped ) if residual is not None else dropped
    x = norm(residual.to(dtype=norm.weight.dtype))
    residual = residual.to(torch.float32)


    return x, residual 


def run_flash(x, drop_path, dropout, norm, residual_in_fp32=True):
    from layer_norm_triton import layer_norm_fn, RMSNorm

    residual = x

    rowscale = drop_path(torch.ones(x.shape[:-1], device=x.device, dtype=x.dtype, ))
    x, residual = layer_norm_fn(
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

    return x, residual


def run_naive(x, drop_path, dropout, norm, residual_in_fp32=False):
    residual = x
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

    breakpoint()

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

    b, n, d = 2, 36, 64
    p = 0.1
    p_path = 0.00

    torch.manual_seed(0)
    x = torch.randn((b, n, d), device='cuda')

    # manual impl.
    norm = nn.LayerNorm(d).cuda()
    dropout = nn.Dropout(p)
    drop_path = StochasticDepth(p_path, mode="row")
    run_naive(x, drop_path, dropout, norm)

    # baselines
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    norm = nn.LayerNorm(d).cuda()
    dropout = nn.Dropout(p)
    drop_path = StochasticDepth(p_path, mode="row")
    torch_out, torch_resid = run_torch(x, drop_path, dropout, norm)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    norm = nn.LayerNorm(d).cuda()
    dropout = nn.Dropout(p)
    drop_path = StochasticDepth(p_path, mode="row")
    flash_out, flash_resid = run_flash(x, drop_path, dropout, norm)

    diff = torch.norm(torch_out - flash_out).item()
    print(torch_out[0,0,:16])
    print(flash_out[0,0,:16])
    print(f"Out Diff: {diff}")

    diff = torch.norm(torch_resid - flash_resid).item()
    print(torch_resid[0,0,:16])
    print(flash_resid[0,0,:16])
    print(f"Resid Diff: {diff}")


