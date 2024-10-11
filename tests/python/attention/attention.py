import torch
import numpy as np

import thunderkittens as tk

class AttentionInference(torch.nn.Module):
    def __init__(self, is_causal=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_causal = is_causal

    def forward(self, q, k, v): # assume (b, n, h, d)

        torch.cuda.synchronize()
        outputs = torch.zeros_like(v)

        torch.cuda.synchronize()
        if self.is_causal:
            tk.mha_causal(q, k, v, outputs)
        else:
            tk.mha_forward(q, k, v, outputs)
        torch.cuda.synchronize()
        return outputs

class CausalTKTrainWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        outputs = torch.zeros_like(v)
        l_vec = torch.zeros(q.shape[0], q.shape[1], q.shape[2], device=q.device, dtype=torch.float32).contiguous()
        tk.mha_causal_train(q, k, v, outputs, l_vec)
        ctx.save_for_backward(q, k, v, outputs, l_vec)
        return outputs
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, l_vec = ctx.saved_tensors
        q, k, v, o, l_vec = q.contiguous(), k.contiguous(), v.contiguous(), o.contiguous(), l_vec.contiguous()
        grad_q = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        grad_k, grad_v = torch.zeros_like(k), torch.zeros_like(v)
        d_vec = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 1, device=q.device, dtype=torch.float32).contiguous()
        tk.mha_causal_train_backward(q, k, v, o, l_vec, d_vec, grad_output.contiguous(), grad_q, grad_k, grad_v)
        return grad_q, grad_k, grad_v
class NonCausalTKTrainWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        outputs = torch.zeros_like(v)
        l_vec = torch.zeros(q.shape[0], q.shape[1], q.shape[2], device=q.device, dtype=torch.float32).contiguous()
        print('Starting mha_train fwd')
        tk.mha_train(q, k, v, outputs, l_vec)
        torch.cuda.synchronize()
        print('Finished mha_train fwd')
        ctx.save_for_backward(q, k, v, outputs, l_vec)
        return outputs
    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, l_vec = ctx.saved_tensors
        q, k, v, o, l_vec = q.contiguous(), k.contiguous(), v.contiguous(), o.contiguous(), l_vec.contiguous()
        grad_q = torch.zeros(q.shape, device=q.device, dtype=torch.float32)
        grad_k, grad_v = torch.zeros_like(k), torch.zeros_like(v)
        d_vec = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 1, device=q.device, dtype=torch.float32).contiguous()
        print('Starting mha_train bwd')
        tk.mha_train_backward(q, k, v, o, l_vec, d_vec, grad_output.contiguous(), grad_q, grad_k, grad_v)
        torch.cuda.synchronize()
        print('Finished mha_train bwd')
        return grad_q, grad_k, grad_v

class AttentionTrain(torch.nn.Module):
    def __init__(self, is_causal=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.is_causal = is_causal
    
    def forward(self, q, k, v):
        return CausalTKTrainWrapper.apply(q, k, v) if self.is_causal else NonCausalTKTrainWrapper.apply(q, k, v)

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f * {'fwd': 1, 'bwd': 2.5, 'fwd_bwd': 3.5}[mode]

def efficiency(flop, time):
    # calculate in teraflops
    flop = flop / 1e12
    time = time / 1e6
    return flop / time

def measure_performance(mode, causal, b, h, n, d):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == 'cuda', "CUDA not available"
    print("Using device:", device)

    fn_class = AttentionInference(causal) if mode == "fwd" else AttentionTrain(causal)

    print('Instantiated class')

    # Generate random data for q, k, v
    q = torch.randn(b, h, n, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(b, h, n, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(b, h, n, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    
    q.grad = None
    k.grad = None
    v.grad = None
    
    torch.cuda.synchronize()

    # Check versus reference
    if mode == "fwd":
        ref_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, attn_mask=None, dropout_p=0.0)
        o = fn_class(q, k, v)
        print(f"Avg Fwd Difference: {torch.mean(torch.abs(ref_output.to(torch.float32) - o.to(torch.float32)))}")
        print(f"Max Fwd Difference: {torch.max(torch.abs(ref_output.to(torch.float32) - o.to(torch.float32)))}")
    else:
        o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, attn_mask=None, dropout_p=0.0)
        loss_ref = torch.sum(o_ref.reshape(-1))
        loss_ref.backward()
        grad_q_ref = torch.clone(q.grad).detach()
        grad_k_ref = torch.clone(k.grad).detach()
        grad_v_ref = torch.clone(v.grad).detach()
        q.grad, k.grad, v.grad = None, None, None
        # Forward pass
        o = fn_class(q, k, v)
        loss = torch.sum(o.reshape(-1))
        loss.backward()
        grad_q = torch.clone(q.grad).detach()
        grad_k = torch.clone(k.grad).detach()
        grad_v = torch.clone(v.grad).detach()
        q.grad, k.grad, v.grad = None, None, None
        # Print some information for verification
        print(f"Avg Bwd Difference (Q): {torch.mean(torch.abs(grad_q_ref.to(torch.float32) - grad_q.to(torch.float32)))}")
        print(f"Max Bwd Difference (Q): {torch.max(torch.abs(grad_q_ref.to(torch.float32) - grad_q.to(torch.float32)))}")
        print(f"Avg Bwd Magnitude  (Q): {torch.mean(torch.abs(grad_q_ref.to(torch.float32)))}")
        print(f"Max Bwd Magnitude  (Q): {torch.max(torch.abs(grad_q_ref.to(torch.float32)))}")
        print(f"Avg Bwd Difference (K): {torch.mean(torch.abs(grad_k_ref.to(torch.float32) - grad_k.to(torch.float32)))}")
        print(f"Max Bwd Difference (K): {torch.max(torch.abs(grad_k_ref.to(torch.float32) - grad_k.to(torch.float32)))}")
        print(f"Avg Bwd Magnitude  (K): {torch.mean(torch.abs(grad_k_ref.to(torch.float32)))}")
        print(f"Max Bwd Magnitude  (K): {torch.max(torch.abs(grad_k_ref.to(torch.float32)))}")
        print(f"Avg Bwd Difference (V): {torch.mean(torch.abs(grad_v_ref.to(torch.float32) - grad_v.to(torch.float32)))}")
        print(f"Max Bwd Difference (V): {torch.max(torch.abs(grad_v_ref.to(torch.float32) - grad_v.to(torch.float32)))}")
        print(f"Avg Bwd Magnitude  (V): {torch.mean(torch.abs(grad_v_ref.to(torch.float32)))}")
        print(f"Max Bwd Magnitude  (V): {torch.max(torch.abs(grad_v_ref.to(torch.float32)))}")


# Test configurations
configs = [('fwd', False, 32, 16, 4096, 128), ('fwd', True, 32, 16, 4096, 128),
           ('bwd', False, 32, 16, 4096, 64), ('bwd', True, 32, 16, 4096, 64)]
        #    ('fwd', True, 32, 16, 2048, 64),
        #    ('fwd', True, 32, 16, 4096, 64),
        #    ('fwd', True, 32, 16, 8192, 64),
        #    ('fwd', True, 32, 16, 16384, 64),
        #    ('fwd', True, 32, 16, 1024, 128),
        #    ('fwd', True, 32, 16, 2048, 128),
        #    ('fwd', True, 32, 16, 4096, 128),
        #    ('fwd', True, 32, 16, 8192, 128),
        #    ('fwd', True, 32, 16, 16384, 128)]
# configs = [('fwd', False, 4, 16, 4096, 128)]
for config in configs:
    print('Running config:', config)
    measure_performance(*config)

