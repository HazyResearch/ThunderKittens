import torch
import torch.nn.functional as F
import sys
from tqdm import tqdm
try:
    import thunderkittens
except ImportError:
    pass

size=(4096, 4096)

TESTNAME = sys.argv[1]

if TESTNAME == 'ones':
    x = 10 * torch.ones(size, dtype=torch.float32, device='cuda')
    w = 10 * torch.ones(size, dtype=torch.float32, device='cuda').t()   # Note: cuBLASLt float8 matmul requires column major for the second argument
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    x = torch.randn(size, dtype=torch.float32, device='cuda') * 0.2
    w = torch.randn(size, dtype=torch.float32, device='cuda').t() * 0.2
elif TESTNAME == 'custom':
    x = torch.arange(size[0] * size[1], dtype=torch.float32, device='cuda') / 100000.0
    w = x.clone()  # Since you want the same values
    x = x.reshape(size).contiguous()
    w = w.reshape(size).contiguous().t()
    print(f"x.shape: {x.shape}, w.shape: {w.shape}")
else:
    print('Invalid test name')
    sys.exit(0)

finfo = torch.finfo(torch.float8_e4m3fn)
FP8_e4m3_MAX = finfo.max
FP8_e4m3_MIN = finfo.min
print(f'FP8_e4m3_MAX: {FP8_e4m3_MAX}, FP8_e4m3_MIN: {FP8_e4m3_MIN}')

def to_float8_e4m3fn(x: torch.Tensor):
    scales = x.abs().amax(dim=-1).div(FP8_e4m3_MAX)
    x = x.div(scales.unsqueeze(-1)).clamp(min=FP8_e4m3_MIN, max=FP8_e4m3_MAX)
    x = x.to(torch.float8_e4m3fn)
    return x, scales.float()

def compare_f8_mm(dtype=torch.float8_e4m3fn) -> None:
    # do a scaled cast to float8 on the inputs
    x_f8, x_inv_s = to_float8_e4m3fn(x)
    w_f8, w_inv_s = to_float8_e4m3fn(w)

    print(f'x_inv_s: {x_inv_s[:10]}')
    print(f'w_inv_s: {w_inv_s[:10]}')

    y = torch._scaled_mm(
        x_f8, w_f8,      
        out_dtype=torch.bfloat16,
        scale_a=x_inv_s.unsqueeze(-1),     # (16, 1)
        scale_b=w_inv_s.unsqueeze(0),      # (1, 16)
        use_fast_accum=True
    ) # bias=bias

    # compare output of float8 matmul to the fp16 baseline
    cos_sim = F.cosine_similarity(torch.mm(x, w).reshape(-1), y.reshape(-1), dim=0)    
    print(f'cos_sim {cos_sim.item():.4f}')

    # average of x
    avg_x = x.abs().mean()
    print(f'avg_x {avg_x.item():.4f}')

    # average of w
    avg_w = w.abs().mean()
    print(f'avg_w {avg_w.item():.4f}')

    # average of y
    avg_y = y.abs().mean()
    print(f'avg_y {avg_y.item():.4f}')

    # max diff
    max_diff = (torch.mm(x, w) - y).abs().max()
    print(f'max_diff {max_diff.item():.4f}')

    # average diff
    avg_diff = (torch.mm(x, w) - y).abs().mean()
    print(f'avg_diff {avg_diff.item():.4f}')

def compare_tk_torch_mm() -> None:
    global x, w
    x_f8, x_inv_s = to_float8_e4m3fn(x)
    w_f8, w_inv_s = to_float8_e4m3fn(w)

    print(f'x_inv_s: {x_inv_s[:10]}')
    print(f'w_inv_s: {w_inv_s[:10]}')

    y = thunderkittens.scaled_matmul(x_f8, w_f8.t().contiguous(), x_inv_s, w_inv_s)

    # compare output of float8 matmul to the fp16 baseline
    cos_sim = F.cosine_similarity(torch.mm(x, w).reshape(-1), y.reshape(-1), dim=0)    
    print(f'cos_sim {cos_sim.item():.4f}')
    
    # average of x
    avg_x = x.abs().mean()
    print(f'avg_x {avg_x.item():.4f}')

    # average of w
    avg_w = w.abs().mean()
    print(f'avg_w {avg_w.item():.4f}')

    # average of y
    avg_y = y.abs().mean()
    print(f'avg_y {avg_y.item():.4f}')

    # max diff
    max_diff = (torch.mm(x, w) - y).abs().max()
    print(f'max_diff {max_diff.item():.4f}')

def compare_tk_torch_scaled_mm() -> None:
    global x, w
    x_f8, x_inv_s = to_float8_e4m3fn(x)
    w_f8, w_inv_s = to_float8_e4m3fn(w)

    print(f'x_inv_s: {x_inv_s[:10]}')
    print(f'w_inv_s: {w_inv_s[:10]}')

    y = thunderkittens.scaled_matmul(x_f8, w_f8.t().contiguous(), x_inv_s, w_inv_s)

    y2 = torch._scaled_mm(
        x_f8, w_f8,      
        out_dtype=torch.bfloat16,
        scale_a=x_inv_s.unsqueeze(-1),     # (16, 1)
        scale_b=w_inv_s.unsqueeze(0),      # (1, 16)
        use_fast_accum=True
    ) # bias=bias


    # compare output of float8 matmul to the fp16 baseline
    cos_sim = F.cosine_similarity(y2.reshape(-1), y.reshape(-1), dim=0)    
    print(f'cos_sim {cos_sim.item():.4f}')
    
    # average of x
    avg_x = x.abs().mean()
    print(f'avg_x {avg_x.item():.4f}')

    # average of w
    avg_w = w.abs().mean()
    print(f'avg_w {avg_w.item():.4f}')

    # average of y
    avg_y = y.abs().mean()
    print(f'avg_y {avg_y.item():.4f}')

    # max diff
    max_diff = (y2 - y).abs().max()
    print(f'max_diff {max_diff.item():.4f}')


    

if __name__ == "__main__":
    compare_f8_mm()
    compare_tk_torch_mm()
    compare_tk_torch_scaled_mm()


