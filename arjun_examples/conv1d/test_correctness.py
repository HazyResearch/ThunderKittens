import torch
import torch.nn as nn
import scipy

import sys
sys.path.append("kernel/")
import short_conv as mod

# Model dim
D_model = 1
# Kernel size
D_conv = 3
# Dimension of kittens tile - the vector needs to be this width minimum
TILE_DIM = 16

conv = nn.Conv1d(
    in_channels = D_model,
    out_channels=D_model,
    kernel_size=D_conv,
    # Depthwise conv
    groups=D_model,
    padding=D_conv - 1,
    bias=True
)

# h is the 1d filter, x is the input vector
def run_torch(h, x):
    # Torch uses cross-correlation (not actual convolution) so we need to reverse the filter
    conv.weight = torch.nn.Parameter(torch.flip(h, (0,)).reshape((1, D_model, D_conv)))
    # Batch dim is 1
    out = conv(x.reshape(1, D_model, x.shape[0]))
    out = out.to(dtype=torch.float32)
    return out

def run_tk(h, x):
    # Toeplitz matrix for the convolution
    # "Full" mode analogous to torch conv1d w/ input padding on both sides
    K = h.shape[0]
    N = x.shape[0]
    toeplitz = scipy.linalg.convolution_matrix(h, N, mode='full')
    H = torch.from_numpy(toeplitz)
    x.unsqueeze_(0).t_()
    # Pad x to be the same size as a tile of kittens
    x = torch.nn.functional.pad(x, (0, TILE_DIM-1), value=0)
    H = H.to(dtype=torch.bfloat16)
    x = x.to(dtype=torch.bfloat16)
    out = torch.zeros_like(x)
    mod.short_conv(
        h, x, out,
        K, N
    )
    # TODO do we convert output to float32?
    out = out.to(torch.float32)
    return out


if __name__ == "__main__":

    N = 1024
    p = 0.0
    p_path = 0.00

    torch.manual_seed(0)
    h = torch.randn((D_conv,), device='cuda')
    x = torch.randn((N,), device='cuda')

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    norm = nn.LayerNorm(d).cuda()
    dropout = nn.Dropout(p)
    drop_path = None #StochasticDepth(p_path, mode="row")
    out = run_torch(x, residual, drop_path, dropout, norm)

    outs = []
    resids = []

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    norm = nn.LayerNorm(d).cuda()
    dropout = nn.Dropout(p)
    drop_path = None #StochasticDepth(p_path, mode="row")
    fn_out = run_tk(x, residual, drop_path, dropout, norm)

    print("----"*10)
    diff = torch.norm(out - fn_out).item()
    print(out[2,4,:8])
    print(fn_out[2,4,:8])
    print(f"Out Diff: {diff}")
