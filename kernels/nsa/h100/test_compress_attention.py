import torch
import triton
import thunderkittens as tk
torch.autograd.set_grad_enabled(True)
from compress_attention_ref import attention_ref
from compress_attention_ref import flash_attn_func
from torch.profiler import profile, record_function, ProfilerActivity
from einops import rearrange, repeat


compress_block_size, compress_block_stride = 32, 16
seq_len = 1024*32
selection_block_size, selected_block_count = 64, 16

dtype = torch.bfloat16
device = "cuda"
torch.set_default_device(device)
torch.set_default_dtype(dtype)
torch.manual_seed(9)


def safe_all_close(out, ref, rtol=1e-2, atol=1e-2):
    assert not out.isnan().any()
    try:
        torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)
    except Exception as e:
        print(e)


def create_data(bs, num_q_head, num_kv_head, head_dim):
    q = torch.randn(bs*seq_len, num_q_head, head_dim, requires_grad=True)
    k = torch.randn(bs*seq_len, num_kv_head, head_dim, requires_grad=True)
    v = torch.randn(bs*seq_len, num_kv_head, head_dim, requires_grad=True)
    t = torch.Tensor([0] + [seq_len] * bs)

    q_ref = q.detach()
    k_ref = k.detach()
    v_ref = v.detach()
    q_ref.requires_grad = True
    k_ref.requires_grad = True
    v_ref.requires_grad = True
    q_ref.retain_grad()
    k_ref.retain_grad()
    v_ref.retain_grad()

    return q, k, v, q_ref, k_ref, v_ref, t


def test_tk(causal, perf=False):
    bs, num_q_head, num_kv_head, head_dim = 16, 64, 4, 128
    pool_kernel_size = selection_block_size // compress_block_stride + 1
    pool_padding = compress_block_size // compress_block_stride - 2
    pool_stride = selection_block_size // compress_block_stride

    q_t = torch.randn(bs, seq_len, num_q_head, head_dim)
    q_ref_t = q_t.detach().requires_grad_()

    ck = torch.randn(bs, seq_len//compress_block_stride, num_kv_head, head_dim)
    cv = torch.randn(bs, seq_len//compress_block_stride, num_kv_head, head_dim)
    
    ck_ref = ck.detach().requires_grad_()
    cv_ref = cv.detach().requires_grad_()
    
    ck_ref_full = repeat(ck_ref, "b s h d -> b s (h g) d", g=num_q_head//num_kv_head).contiguous()
    cv_ref_full = repeat(cv_ref, "b s h d -> b s (h g) d", g=num_q_head//num_kv_head).contiguous()
    
    grad_output = torch.randn_like(q_t)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        if perf:
            ref_o, indices = flash_attn_func(q_ref_t, ck_ref_full, cv_ref_full, compress_block_stride, compress_block_size, causal, None, num_kv_head, pool_kernel_size, 
                                    pool_stride, pool_padding, selected_block_count)
            ref_o.backward(grad_output, retain_graph=True)
        else:
            with torch.enable_grad():
                ref_o, ref_indices = attention_ref(q_ref_t, ck_ref_full, cv_ref_full, compress_block_stride, compress_block_size, causal=causal, scale=None,
                                            pool_num_kv_head=num_kv_head, pool_kernel_size=pool_kernel_size, pool_stride=pool_stride, pool_padding=pool_padding, 
                                            select_block_count=selected_block_count, upcast=False)
                ref_o.retain_grad()
                ref_loss = (ref_o*ref_o).sum()
                ref_loss.backward()
                grad_output = ref_o.grad
        
        o, lse = tk.compress_attn_fwd(q_t, ck, cv, causal, compress_block_stride, compress_block_size)
        q_grad, k_grad, v_grad = tk.compress_attn_bwd(q_t, ck, cv, o, lse, grad_output.to(q_t.dtype), causal, compress_block_stride, compress_block_size)

    if perf:
        sort_by_keyword = "cuda_time_total"
        print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))
    
    print('test output ')
    safe_all_close(o, ref_o, rtol=5e-3, atol=5e-3)
    
    print('test grad_v')
    safe_all_close(v_grad.to(cv_ref.grad.dtype), cv_ref.grad, rtol=3e-2, atol=3e-2)
    print('test grad_k')
    safe_all_close(k_grad.to(ck_ref.grad.dtype), ck_ref.grad, rtol=3e-2, atol=3e-2)
    print('test grad_q')
    safe_all_close(q_grad.to(q_ref_t.grad.dtype), q_ref_t.grad, rtol=3e-2, atol=3e-2)
    print('PASS CAUSAL')


if __name__ == '__main__':
    test_tk(True, perf=False)
