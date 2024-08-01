import torch
import torch.nn as nn

import sys
sys.path.append("kernel/")
import layer_norm as mod

# Model dim
D_model = 3
# Kernel size
D_conv = 3


def run_torch(h, x):
    self.conv = nn.Conv1d(
        in_channels = D_model,
        out_channels=D_model,
        kernel_size=D_conv,
        groups=D_model,
        padding=D_conv - 1,
        bias=True
    )

def run_tk(h, x, K, N):
    h = h.to(dtype=torch.bfloat16)
    x = x.to(dtype=torch.bfloat16)

}

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

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    norm = nn.LayerNorm(d).cuda()
    dropout = nn.Dropout(p)
    drop_path = None #StochasticDepth(p_path, mode="row")
    out, resid = run_torch(x, residual, drop_path, dropout, norm)

    outs = []
    resids = []

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    norm = nn.LayerNorm(d).cuda()
    dropout = nn.Dropout(p)
    drop_path = None #StochasticDepth(p_path, mode="row")
    fn_out, fn_resid = run_tk(x, residual, drop_path, dropout, norm)

    print("----"*10)
    diff = torch.norm(out - fn_out).item()
    print(out[2,4,:8])
    print(fn_out[2,4,:8])
    print(f"Out Diff: {diff}")

    diff = torch.norm(resid - fn_resid).item()
    print(resid[4,2,:8])
    print(fn_resid[4,2,:8])
    print(f"Resid Diff: {diff}")
