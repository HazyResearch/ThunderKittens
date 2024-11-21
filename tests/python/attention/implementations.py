import torch
from functools import partial
import numpy as np
import thunderkittens as tk

try:
    # To compare to Flash Attention 2
    from flash_attn import flash_attn_func as flash_attn_func2
    print(f"Successfully imported flash_attention")
except:
    flash_attn_func = None
    print("Could not import flash_attention. Please: pip install flash-attn")

try:
    # To compare to Flash Attention 3
    from flash_attn_interface import flash_attn_func as flash_attn_func3
    print(f"Successfully imported flash_attn_interface")
except:
    flash_attn_func3 = None
    print("Could not import flash_attn_interface. Please: clone flash-attention and install the hopper kernels")


def get_flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    flops = f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)
    return flops


################ ATTENTION ################

def get_attention_inputs(b, h, n, dv, dt=torch.bfloat16, pad_multiple=0, ):

    if pad_multiple:
        n = (n // pad_multiple) * pad_multiple + pad_multiple  

    q = torch.randn(b, h, n, dv, dtype=dt, device='cuda').requires_grad_(True)
    k = torch.randn(b, h, n, dv, dtype=dt, device='cuda').requires_grad_(True)
    v = torch.randn(b, h, n, dv, dtype=dt, device='cuda').requires_grad_(True)
    dO = torch.randn(b, h, n, dv, dtype=dt, device='cuda')
    return q, k, v, dO


def attention_test(dt, b, h, n, dv, causal, is_forwards, method_str, num_iters=10, verbose=True, torch_compile=False, **kwargs):

    for stage in ['warmup', 'timed']:

        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

        for i in range(num_iters):
            
            def pytorch_method(q, k, v, causal):
                QK = torch.matmul(q, k.transpose(-2, -1))
                QK /= (q.size(-1) ** 0.5)
                if causal:
                    mask = torch.triu(torch.ones(QK.size(-2), QK.size(-1)), 1).to(torch.bool).to(QK.device)
                    QK.masked_fill_(mask, float('-inf'))
                QK = torch.nn.functional.softmax(QK, dim=-1)
                y = torch.matmul(QK, v)
                return y

            if torch_compile and method_str == "pytorch":
                pytorch_method = torch.compile(pytorch_method)
            
            try:
                q, k, v, dO = get_attention_inputs(b, h, n, dv, dt)
                
                if method_str == "pytorch":
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = pytorch_method(q, k, v, causal)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "fa2": 
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == 'tk':
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y, l_vec = tk.mha_forward(q, k, v, causal)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "fa3":
                    q = q.transpose(1,2)
                    k = k.transpose(1,2)
                    v = v.transpose(1,2)
                    dO = dO.transpose(1,2)
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y, lse = flash_attn_func3(q, k, v, causal=causal)
                    end_events[i].record() 
                    torch.cuda.synchronize()

                else:
                    assert 0, f"Unknown method: {method_str}"

                outputs = ( y )

            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                return None, -1

            if is_forwards and stage == 'timed':
                continue

            try:

                if method_str == "pytorch":
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y.backward(dO)
                    q_grad = q.grad.to(torch.bfloat16)
                    k_grad = k.grad.to(torch.bfloat16)
                    v_grad = v.grad.to(torch.bfloat16)
                    y = y.to(torch.bfloat16)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "fa2":
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y.backward(dO)
                    q_grad = q.grad
                    k_grad = k.grad
                    v_grad = v.grad
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == 'tk':
                    torch.cuda.synchronize()
                    start_events[i].record()
                    q_grad, k_grad, v_grad = tk.mha_backward(q, k, v, y, l_vec, dO, causal)
                    end_events[i].record()
                    torch.cuda.synchronize()

                elif method_str == "fa3":
                    torch.cuda.synchronize()
                    start_events[i].record()
                    y.backward(dO)
                    q_grad = q.grad
                    k_grad = k.grad
                    v_grad = v.grad
                    end_events[i].record()
                    torch.cuda.synchronize()

                else:
                    assert 0, f"Unknown method: {method_str}"

                outputs = (q_grad, k_grad, v_grad)

            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                return (None, None, None), -1

            torch.cuda.empty_cache()
    
    tot = sum([s.elapsed_time(e) for s, e in zip(start_events, end_events)])/num_iters
    # for output in outputs:
    #     try:
    #         assert not np.isnan(output.detach().float().cpu()).any(), "NaN values detected in output 'o'"
    #         assert not np.isinf(output.detach().float().cpu()).any(), "Inf values detected in output 'o'"
    #     except: 
    #         breakpoint()
    return outputs, tot


IMPLEMENTATIONS = {
    "attn_fwd_PT_c=t": partial(attention_test, causal=True, is_forwards=True, method_str="pytorch"),
    "attn_fwd_FA2_c=t": partial(attention_test, causal=True, is_forwards=True, method_str="fa2"),
    "attn_fwd_FA3_c=t": partial(attention_test, causal=True, is_forwards=True, method_str="fa3"),
    "attn_fwd_TK_c=t": partial(attention_test, causal=True, is_forwards=True, method_str="tk"),
    "attn_fwd_PT_c=f": partial(attention_test, causal=False, is_forwards=True, method_str="pytorch"),
    "attn_fwd_FA2_c=f": partial(attention_test, causal=False, is_forwards=True, method_str="fa2"),
    "attn_fwd_FA3_c=f": partial(attention_test, causal=False, is_forwards=True, method_str="fa3"),
    "attn_fwd_TK_c=f": partial(attention_test, causal=False, is_forwards=True, method_str="tk"),
}

IMPLEMENTATIONS_BWD = {
    "attn_bwd_PT_c=t": partial(attention_test, causal=True, is_forwards=False, method_str="pytorch"),
    "attn_bwd_FA2_c=t": partial(attention_test, causal=True, is_forwards=False, method_str="fa2"),
    "attn_bwd_FA3_c=t": partial(attention_test, causal=True, is_forwards=False, method_str="fa3"),
    "attn_bwd_TK_c=t": partial(attention_test, causal=True, is_forwards=False, method_str="tk"),
    "attn_bwd_PT_c=f": partial(attention_test, causal=False, is_forwards=False, method_str="pytorch"),
    "attn_bwd_FA2_c=f": partial(attention_test, causal=False, is_forwards=False, method_str="fa2"),
    "attn_bwd_FA3_c=f": partial(attention_test, causal=False, is_forwards=False, method_str="fa3"),
    "attn_bwd_TK_c=f": partial(attention_test, causal=False, is_forwards=False, method_str="tk"),
}

NAME = "ATTENTION"

