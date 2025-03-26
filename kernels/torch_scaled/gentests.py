import torch
import torch.nn.functional as F
import sys
from tqdm import tqdm
try:
    import thunderkittens
except ImportError:
    pass

size=(4096, 4096)
# size = (128, 128)

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

def to_float8_e4m3fn(x: torch.Tensor, dim: int = -1):
    scales = x.abs().amax(dim=dim).div(FP8_e4m3_MAX)
    x = x.div(scales.unsqueeze(dim)).clamp(min=FP8_e4m3_MIN, max=FP8_e4m3_MAX)
    x = x.to(torch.float8_e4m3fn)
    return x, scales.float()

def compare_f8_mm(dtype=torch.float8_e4m3fn) -> None:
    # do a scaled cast to float8 on the inputs
    x_f8, x_inv_s = to_float8_e4m3fn(x, dim = -1)
    w_f8, w_inv_s = to_float8_e4m3fn(w, dim = 0)

    # print(f'x_inv_s: {x_inv_s[:10]}')
    # print(f'w_inv_s: {w_inv_s[:10]}')

    y = torch._scaled_mm(
        x_f8, w_f8,      
        out_dtype=torch.bfloat16,
        scale_a=x_inv_s.unsqueeze(-1),     # (16, 1)
        scale_b=w_inv_s.unsqueeze(0),      # (1, 16)
        use_fast_accum=True
    ) # bias=bias

    # compare output of float8 matmul to the fp16 baseline
    result = torch.mm(x, w)
    cos_sim = F.cosine_similarity(result.reshape(-1), y.reshape(-1), dim=0)    
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
    max_diff = (result - y).abs().max()
    print(f'max_diff {max_diff.item():.4f}')

    # Additional debugging
    # max_diff_idx = (result - y).abs().argmax().item()
    # max_diff_idx = (max_diff_idx // size[1], max_diff_idx % size[1])

    # print(f"max_diff_idx: {max_diff_idx}, result[max_diff_idx]: {result[max_diff_idx[0], max_diff_idx[1]]}, y[max_diff_idx]: {y[max_diff_idx[0], max_diff_idx[1]]}") # 1263, 1019. 
    # # result[1263, 1019] = 10.5930, y[1263, 1019] = 15.625
    # x_dquant = x_f8.float() * x_inv_s.unsqueeze(-1)
    # w_dquant = w_f8.float() * w_inv_s.unsqueeze(0)
    # i, j = max_diff_idx
    # print(f"x_dquant error at row {i}: { (x_dquant[i, :] - x[i, :]).abs().mean()}, w_dquant error at col {j}: {(w_dquant[:, j] - w[:, j]).abs().mean()}")
    # print(f"x_dquant error in general: {(x_dquant - x).abs().mean()} with std {(x_dquant - x).abs().mean(dim=-1).std()}, w_dquant error in general: {(w_dquant - w).abs().mean()} with std {(w_dquant - w).abs().mean(dim=0).std()}")

    # contribs_quant = (x_dquant[i, :] * w_dquant[:, j])
    # print(f"contribs_quant: {contribs_quant.sum()}")

    # contribs_fp32 = (x[i, :] * w[:, j])
    # print(f"contribs_fp32: {contribs_fp32.sum()}")

    # w_col_j_topkmax, w_col_j_topkmax_idx = w[:, j].topk(k=10, dim=-1)
    # print(f"w_col_j_topkmax: {round_list(w_col_j_topkmax.tolist())}, w_dquant_col_j_topkmax: {round_list(w_dquant[w_col_j_topkmax_idx, j].tolist())}, w_col_j_topkmax_idx: {w_col_j_topkmax_idx.tolist()}")

    # w_col_j_bottomkmax, w_col_j_bottomkmax_idx = w[:, j].topk(k=10, dim=-1, largest=False)
    # print(f"w_col_j_bottomkmax: {round_list(w_col_j_bottomkmax.tolist())}, w_dquant_col_j_bottomkmax: {round_list(w_dquant[w_col_j_bottomkmax_idx, j].tolist())}, w_col_j_bottomkmax_idx: {w_col_j_bottomkmax_idx.tolist()}")

    # j = 0
    # print("now j = 0")
    # w_col_j_topkmax, w_col_j_topkmax_idx = w[:, j].topk(k=10, dim=-1)
    # print(f"w_col_j_topkmax: {round_list(w_col_j_topkmax.tolist())}, w_dquant_col_j_topkmax: {round_list(w_dquant[w_col_j_topkmax_idx, j].tolist())}, w_col_j_topkmax_idx: {w_col_j_topkmax_idx.tolist()}")

    # w_col_j_bottomkmax, w_col_j_bottomkmax_idx = w[:, j].topk(k=10, dim=-1, largest=False)
    # print(f"w_col_j_bottomkmax: {round_list(w_col_j_bottomkmax.tolist())}, w_dquant_col_j_bottomkmax: {round_list(w_dquant[w_col_j_bottomkmax_idx, j].tolist())}, w_col_j_bottomkmax_idx: {w_col_j_bottomkmax_idx.tolist()}")


    # x_row_i_topkmax, x_row_i_topkmax_idx = x[i, :].topk(k=10, dim=-1)
    # print(f"x_row_i_topkmax: {round_list(x_row_i_topkmax.tolist())}, x_dquant_row_i_topkmax: {round_list(x_dquant[i, x_row_i_topkmax_idx].tolist())}, x_row_i_topkmax_idx: {x_row_i_topkmax_idx.tolist()}")

    # diff_contribs = (contribs_fp32 - contribs_quant)
    # print("\nIndividual term contributions (over index k):")
    # for k, (fp32_val, quant_val, diff_val) in enumerate(zip(contribs_fp32, contribs_quant, diff_contribs)):
    #     print(f"k = {k}: fp32 = {fp32_val:.4f}, quant = {quant_val:.4f}, diff = {diff_val:.4f}")


    # average diff
    avg_diff = (torch.mm(x, w) - y).abs().mean()
    print(f'avg_diff {avg_diff.item():.4f}')

def compare_tk_torch_mm() -> None:
    x_f8, x_inv_s = to_float8_e4m3fn(x, dim = -1)
    w_f8, w_inv_s = to_float8_e4m3fn(w, dim = 0)

    # print(f'x_inv_s: {x_inv_s[:10]}')
    # print(f'w_inv_s: {w_inv_s[:10]}')

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

    # average diff
    avg_diff = (torch.mm(x, w) - y).abs().mean()
    print(f'avg_diff {avg_diff.item():.4f}')

def compare_tk_torch_scaled_mm() -> None:
    x_f8, x_inv_s = to_float8_e4m3fn(x, dim = -1)
    w_f8, w_inv_s = to_float8_e4m3fn(w, dim = 0)

    # print(f'x_inv_s: {x_inv_s[:10]}')
    # print(f'w_inv_s: {w_inv_s[:10]}')

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

    # average diff
    avg_diff = (y2 - y).abs().mean()
    print(f'avg_diff {avg_diff.item():.4f}')

def compare_tk_torch_dequant_mm() -> None:
    x_f8, x_inv_s = to_float8_e4m3fn(x, dim = -1)
    w_f8, w_inv_s = to_float8_e4m3fn(w, dim = 0)

    # print(f'x_inv_s: {x_inv_s[:10]}')
    # print(f'w_inv_s: {w_inv_s[:10]}')

    y = thunderkittens.scaled_matmul(x_f8, w_f8.t().contiguous(), x_inv_s, w_inv_s)

    # y2 = torch._scaled_mm(
    #     x_f8, w_f8,      
    #     out_dtype=torch.bfloat16,
    #     scale_a=x_inv_s.unsqueeze(-1),     # (16, 1)
    #     scale_b=w_inv_s.unsqueeze(0),      # (1, 16)
    #     use_fast_accum=True
    # ) # bias=bias
    x2 = x_f8.float() * x_inv_s.unsqueeze(-1)
    w2 = w_f8.float() * w_inv_s.unsqueeze(0)
    y2 = x2 @ w2

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

    # average diff
    avg_diff = (y2 - y).abs().mean()
    print(f'avg_diff {avg_diff.item():.4f}')

    

if __name__ == "__main__":
    compare_f8_mm()
    compare_tk_torch_mm()
    compare_tk_torch_scaled_mm()
    compare_tk_torch_dequant_mm()


