import torch
from flash_attn_interface import flash_attn_func
import thunderkittens as tk
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def pytorch_test(Q, K, V, dO, causal):
    q_ = Q.to(torch.float64).requires_grad_()
    k_ = K.to(torch.float64).requires_grad_()
    v_ = V.to(torch.float64).requires_grad_()
    dO_ = dO.to(torch.float64)
    
    # manual pytorch implementation of scaled dot product attention
    QK = torch.matmul(q_, k_.transpose(-2, -1))
    QK /= (q_.size(-1) ** 0.5)
    if causal:
        mask = torch.triu(torch.ones(QK.size(-2), QK.size(-1)), 1).to(torch.bool).to(QK.device)
        QK.masked_fill_(mask, float('-inf'))
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v_)
    
    output.backward(dO_)
    
    q_grad = q_.grad
    k_grad = k_.grad
    v_grad = v_.grad
    
    q_grad = q_grad.to(torch.bfloat16)
    k_grad = k_grad.to(torch.bfloat16)
    v_grad = v_grad.to(torch.bfloat16)
    output = output.to(torch.bfloat16)
    
    return output, q_grad, k_grad, v_grad

def fa2_test(Q, K, V, dO, causal):
    Q.requires_grad = True
    K.requires_grad = True
    V.requires_grad = True
    output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)
    output.backward(dO)
    
    return output, Q.grad, K.grad, V.grad

def fa3_test(Q, K, V, dO, causal):
    Q_  = Q.permute(0, 2, 1, 3).clone().detach().contiguous()
    K_  = K.permute(0, 2, 1, 3).clone().detach().contiguous()
    V_  = V.permute(0, 2, 1, 3).clone().detach().contiguous()
    dO_ = dO.permute(0, 2, 1, 3).clone().detach().contiguous()
    
    Q_.requires_grad = True
    K_.requires_grad = True
    V_.requires_grad = True
    
    out, _ = flash_attn_func(Q_, K_, V_, causal=causal)
    out.backward(dO_)
    
    qgrad = Q_.grad
    kgrad = K_.grad
    vgrad = V_.grad
    
    qg_ = qgrad.permute(0, 2, 1, 3).contiguous()
    kg_ = kgrad.permute(0, 2, 1, 3).contiguous()
    vg_ = vgrad.permute(0, 2, 1, 3).contiguous()
    
    output = out.permute(0, 2, 1, 3).contiguous()
    
    return output, qg_, kg_, vg_

def h100_fwd_kernel_test(Q, K, V, dO, causal, mode): 
    if mode == 'backward':
        o = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)
        
        q_ = Q.to(torch.float64)
        k_ = K.to(torch.float64)
        
        QK = torch.matmul(q_, k_.transpose(-2, -1))
        QK /= (q_.size(-1) ** 0.5)
        
        max_vec = QK.max(dim=-1, keepdim=True).values
        l_vec   = QK - max_vec
        l_vec   = torch.exp(l_vec)
        l_vec   = l_vec.sum(dim=-1, keepdim=True)
        l_vec   = max_vec + torch.log(l_vec)
        
        if (q_.size(-1) == 64):
            l_vec = l_vec * -8.0
        if (q_.size(-1) == 128):
            l_vec = l_vec * -11.313708499
        
        l_vec = l_vec.to(torch.float)
        d_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=torch.float)
        qg, kg, vg = tk.mha_backward(Q, K, V, o, l_vec, d_vec, dO, causal)
        
        return o, qg, kg, vg
    else:
        Q.requires_grad = True
        K.requires_grad = True
        V.requires_grad = True
        
        o = torch.zeros_like(Q).contiguous()
        
        l_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=torch.float)
        tk.mha_forward(Q, K, V, o, l_vec, causal)
        
        d_vec = torch.zeros(Q.shape[0], Q.shape[1], Q.shape[2], 1, device=Q.device, dtype=torch.float)
        qg, kg, vg = tk.mha_backward(Q, K, V, o, l_vec, d_vec, dO, causal)
        
        return o, qg, kg, vg

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)

    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    
    return scaled_tensor.contiguous()

def check_correctness(b, h, n, d, causal, mean, std, num_iterations=100, grad_type='all'):
    results = {
        'TK vs PT': {'qg': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
                     'kg': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
                     'vg': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0}},
        'FA2 vs PT': {'qg': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
                      'kg': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
                      'vg': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0}},
        'FA3 vs PT': {'qg': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
                      'kg': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
                      'vg': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0}}
    }

    for _ in range(num_iterations):
        torch.manual_seed(0)
        
        Q  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        K  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        V  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        
        _, pt_qg, pt_kg, pt_vg = pytorch_test(Q, K, V, dO, causal)
        _, fa2_qg, fa2_kg, fa2_vg = fa2_test(Q, K, V, dO, causal)
        _, fa3_qg, fa3_kg, fa3_vg = fa3_test(Q, K, V, dO, causal)
        _, tk_qg, tk_kg, tk_vg = h100_fwd_kernel_test(Q, K, V, dO, causal, 'backward')
        
        grad_tensors = {'qg': (pt_qg, tk_qg, fa2_qg, fa3_qg),
                        'kg': (pt_kg, tk_kg, fa2_kg, fa3_kg),
                        'vg': (pt_vg, tk_vg, fa2_vg, fa3_vg)}
        
        if grad_type != 'all':
            grad_tensors = {grad_type: grad_tensors[grad_type]}
        
        for gtype, (pt, tk, fa2, fa3) in grad_tensors.items():
            for name, impl in [('TK vs PT', tk), ('FA2 vs PT', fa2), ('FA3 vs PT', fa3)]:
                diff = pt - impl
                abs_diff = torch.abs(diff)
                results[name][gtype]['sum_diff'] += torch.sum(abs_diff).item()
                results[name][gtype]['sum_abs'] += torch.sum(torch.abs(pt)).item()
                results[name][gtype]['max_diff'] = max(results[name][gtype]['max_diff'], torch.max(abs_diff).item())
                
        torch.cuda.empty_cache()

    total_elements = b * h * n * d * num_iterations
    for name, data in results.items():
        for gtype in data:
            avg_diff = data[gtype]['sum_diff'] / total_elements
            max_diff = data[gtype]['max_diff']
            results[name][gtype] = {'avg_diff': avg_diff, 'max_diff': max_diff}

    return results

def generate_error_graphs(b, h, d, causal, mean, std, grad_type='all'):
    seq_lengths = [768 * (2**i) for i in range(3)]

    tk_avg_errors, tk_max_errors = {}, {}
    fa2_avg_errors, fa2_max_errors = {}, {}
    fa3_avg_errors, fa3_max_errors = {}, {}

    grad_types = ['qg', 'kg', 'vg'] if grad_type == 'all' else [grad_type]

    for gtype in grad_types:
        tk_avg_errors[gtype], tk_max_errors[gtype] = [], []
        fa2_avg_errors[gtype], fa2_max_errors[gtype] = [], []
        fa3_avg_errors[gtype], fa3_max_errors[gtype] = [], []

    for n in tqdm(seq_lengths, desc="Generating error data"):
        results = check_correctness(b, h, n, d, causal, mean, std, grad_type=grad_type)
        
        for gtype in grad_types:
            tk_avg_errors[gtype].append(results['TK vs PT'][gtype]['avg_diff'])
            tk_max_errors[gtype].append(results['TK vs PT'][gtype]['max_diff'])
            fa2_avg_errors[gtype].append(results['FA2 vs PT'][gtype]['avg_diff'])
            fa2_max_errors[gtype].append(results['FA2 vs PT'][gtype]['max_diff'])
            fa3_avg_errors[gtype].append(results['FA3 vs PT'][gtype]['avg_diff'])
            fa3_max_errors[gtype].append(results['FA3 vs PT'][gtype]['max_diff'])

    for gtype in grad_types:
        # Generate average error graph
        plt.figure(figsize=(12, 8))
        plt.plot(seq_lengths, tk_avg_errors[gtype], label='TK', marker='o')
        plt.plot(seq_lengths, fa2_avg_errors[gtype], label='FA2', marker='s')
        plt.plot(seq_lengths, fa3_avg_errors[gtype], label='FA3', marker='^')

        plt.xlabel('Sequence Length')
        plt.ylabel(f'Average Error ({gtype})')
        plt.title(f'Average Error vs Sequence Length for {gtype} (b={b}, h={h}, d={d}, causal={causal}, mean={mean}, std={std})')
        plt.legend()
        plt.xscale('log', base=2)
        plt.yscale('linear')
        plt.xticks(seq_lengths, labels=seq_lengths)
        plt.grid(True)
        plt.savefig(f'avg_error_graph_{gtype}_b{b}_h{h}_d{d}_causal{causal}_mean{mean}_std{std}.png')
        plt.close()

        # Generate max error graph
        plt.figure(figsize=(12, 8))
        plt.plot(seq_lengths, tk_max_errors[gtype], label='TK', marker='o')
        plt.plot(seq_lengths, fa2_max_errors[gtype], label='FA2', marker='s')
        plt.plot(seq_lengths, fa3_max_errors[gtype], label='FA3', marker='^')

        plt.xlabel('Sequence Length')
        plt.ylabel(f'Maximum Error ({gtype})')
        plt.title(f'Maximum Error vs Sequence Length for {gtype} (b={b}, h={h}, d={d}, causal={causal}, mean={mean}, std={std})')
        plt.legend()
        plt.xscale('log', base=2)
        plt.yscale('linear')
        plt.xticks(seq_lengths, labels=seq_lengths)
        plt.grid(True)
        plt.savefig(f'max_error_graph_{gtype}_b{b}_h{h}_d{d}_causal{causal}_mean{mean}_std{std}.png')
        plt.close()

# Example usage
b, h, d = 1, 8, 64
causal = True
mean = 1e-1
std = 1

generate_error_graphs(b, h, d, causal, mean, std, grad_type='all')
# for mode in ['output', 'backward', 'all']:
#     generate_error_graphs(b, h, d, causal, mean, std, error_mode=mode)

print("Error graphs generated and saved for all modes.")