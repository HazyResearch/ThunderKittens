import torch
import torch.nn.functional as F
import sys


size=(16, 16)

TESTNAME = sys.argv[1]

if TESTNAME == 'ones':
    x = torch.ones(size, dtype=torch.float16, device='cuda')
    w = torch.ones(size, dtype=torch.float16, device='cuda').t()   # Note: cuBLASLt float8 matmul requires column major for the second argument
elif TESTNAME == 'randn':
    torch.random.manual_seed(42)
    x = torch.randn(size, dtype=torch.float16, device='cuda')
    w = torch.randn(size, dtype=torch.float16, device='cuda').t() # Note: cuBLASLt float8 matmul requires column major for the second argument
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

    breakpoint()
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

o = compare_f8_mm()

if __name__ == "__main__":
    compare_f8_mm()
    



