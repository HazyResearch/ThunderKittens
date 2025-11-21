import torch
from flash_attn_interface import flash_attn_func
from _C import 
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
        
        QK = torch.einsum('bhnd,bhmd->bhnm', q_, k_)
        
        if causal:
            mask = torch.triu(torch.ones(QK.size(-2), QK.size(-1)), 1).to(torch.bool).to(QK.device)
            QK.masked_fill_(mask, float('-inf'))
        
        # compute rowmax
        max_vec = QK.max(dim=-1, keepdim=True).values
        
        QK = QK * (1.0 / (q_.size(-1) ** 0.5))
        QK = QK * (1.44269504089)
        
        max_vec = max_vec * (1.44269504089) * (1.0 / (q_.size(-1) ** 0.5))

        QK = QK - max_vec
        QK = torch.exp2(QK)
        
        norm_vec = QK.sum(dim=-1, keepdim=True)
        
        max_vec  = max_vec * 0.69314718056
        norm_vec = torch.log(norm_vec)
        l_vec   = max_vec + norm_vec
        
        if (q_.size(-1) == 64):
            l_vec = l_vec * -8.0
        if (q_.size(-1) == 128):
            l_vec = l_vec * -11.313708499
        l_vec = l_vec.to(torch.float)
        
        _, l_vec_fa3 = flash_attn_func(Q.permute(0, 2, 1, 3), K.permute(0, 2, 1, 3), V.permute(0, 2, 1, 3), causal=causal)
        if (q_.size(-1) == 64):
            l_vec_fa3 = l_vec_fa3 * -8.0
        if (q_.size(-1) == 128):
            l_vec_fa3 = l_vec_fa3 * -11.313708499
        l_vec_fa3 = l_vec_fa3.to(torch.float)
        
        qg, kg, vg = tk.mha_backward(Q, K, V, o, l_vec_fa3, dO, causal)
        
        return o, qg, kg, vg
    else:
        Q.requires_grad = True
        K.requires_grad = True
        V.requires_grad = True
        
        o, l_vec   = tk.mha_forward(Q, K, V, causal)
        qg, kg, vg = tk.mha_backward(Q, K, V, o, l_vec, dO, causal)
        
        return o, qg, kg, vg

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)

    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    
    return scaled_tensor.contiguous()

def check_correctness(b, h, n, d, causal, mean, std, num_iterations=100, error_mode='all'):
    results = {
        'TK vs PT': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
        'FA2 vs PT': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0},
        'FA3 vs PT': {'sum_diff': 0, 'sum_abs': 0, 'max_diff': 0}
    }

    for _ in range(num_iterations):
        torch.manual_seed(0)
        
        Q  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        K  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        V  = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        
        pt_o, pt_qg, pt_kg, pt_vg = pytorch_test(Q, K, V, dO, causal)
        fa2_o, fa2_qg, fa2_kg, fa2_vg = fa2_test(Q, K, V, dO, causal)
        fa3_o, fa3_qg, fa3_kg, fa3_vg = fa3_test(Q, K, V, dO, causal)
        tk_o, tk_qg, tk_kg, tk_vg = h100_fwd_kernel_test(Q, K, V, dO, causal, error_mode)
        
        if error_mode == 'output':
            tensors = [(pt_o, tk_o, fa2_o, fa3_o)]
        elif error_mode == 'backward':
            tensors = [(pt_qg, tk_qg, fa2_qg, fa3_qg),
                       (pt_kg, tk_kg, fa2_kg, fa3_kg),
                       (pt_vg, tk_vg, fa2_vg, fa3_vg)]
        else:  # 'all'
            tensors = [(pt_o, tk_o, fa2_o, fa3_o),
                       (pt_qg, tk_qg, fa2_qg, fa3_qg),
                       (pt_kg, tk_kg, fa2_kg, fa3_kg),
                       (pt_vg, tk_vg, fa2_vg, fa3_vg)]
        
        for pt, tk, fa2, fa3 in tensors:
            for name, impl in [('TK vs PT', tk), ('FA2 vs PT', fa2), ('FA3 vs PT', fa3)]:
                diff = pt - impl
                abs_diff = torch.abs(diff)
                results[name]['sum_diff'] += torch.sum(abs_diff).item()
                results[name]['sum_abs'] += torch.sum(torch.abs(pt)).item()
                results[name]['max_diff'] = max(results[name]['max_diff'], torch.max(abs_diff).item())
                
        torch.cuda.empty_cache()

    total_elements = b * h * n * d * num_iterations * (1 if error_mode == 'output' else 3 if error_mode == 'backward' else 4)
    for name, data in results.items():
        avg_diff = data['sum_diff'] / total_elements
        max_diff = data['max_diff']
        results[name] = {'avg_diff': avg_diff, 'max_diff': max_diff}

    return results

def generate_error_graphs(b, h, d, causal, mean, std, error_mode='all'):
    seq_lengths = [768 * (2**i) for i in range(3)]

    tk_avg_errors, tk_max_errors = [], []
    fa2_avg_errors, fa2_max_errors = [], []
    fa3_avg_errors, fa3_max_errors = [], []

    for n in tqdm(seq_lengths, desc="Generating error data"):
        results = check_correctness(b, h, n, d, causal, mean, std, error_mode=error_mode)
        
        tk_avg_errors.append(results['TK vs PT']['avg_diff'])
        tk_max_errors.append(results['TK vs PT']['max_diff'])
        fa2_avg_errors.append(results['FA2 vs PT']['avg_diff'])
        fa2_max_errors.append(results['FA2 vs PT']['max_diff'])
        fa3_avg_errors.append(results['FA3 vs PT']['avg_diff'])
        fa3_max_errors.append(results['FA3 vs PT']['max_diff'])

    # Generate average error graph
    plt.figure(figsize=(12, 8))
    plt.plot(seq_lengths, tk_avg_errors, label='TK',   marker='o')
    plt.plot(seq_lengths, fa2_avg_errors, label='FA2', marker='s')
    plt.plot(seq_lengths, fa3_avg_errors, label='FA3', marker='^')

    plt.xlabel('Sequence Length')
    plt.ylabel('Average Error')
    plt.title(f'Average Error vs Sequence Length (b={b}, h={h}, d={d}, causal={causal}, mean={mean}, std={std}, mode={error_mode})')
    plt.legend()
    plt.xscale('log', base=2)
    plt.yscale('linear')
    plt.xticks(seq_lengths, labels=seq_lengths)
    plt.grid(True)
    plt.savefig(f'avg_error_graph_b{b}_h{h}_d{d}_causal{causal}_mean{mean}_std{std}_mode{error_mode}.png')
    plt.close()

    # Generate max error graph
    plt.figure(figsize=(12, 8))
    plt.plot(seq_lengths, tk_max_errors,  label='TK',  marker='o')
    plt.plot(seq_lengths, fa2_max_errors, label='FA2', marker='s')
    plt.plot(seq_lengths, fa3_max_errors, label='FA3', marker='^')

    plt.xlabel('Sequence Length')
    plt.ylabel('Maximum Error')
    plt.title(f'Maximum Error vs Sequence Length (b={b}, h={h}, d={d}, causal={causal}, mean={mean}, std={std}, mode={error_mode})')
    plt.legend()
    plt.xscale('log', base=2)
    plt.yscale('linear')
    plt.xticks(seq_lengths, labels=seq_lengths)
    plt.grid(True)
    plt.savefig(f'max_error_graph_b{b}_h{h}_d{d}_causal{causal}_mean{mean}_std{std}_mode{error_mode}.png')
    plt.close()

# Example usage
b, h, d = 12, 12, 64
causal = True
mean = 1e-1
std = 10

for mode in ['output', 'backward', 'all']:
    generate_error_graphs(b, h, d, causal, mean, std, error_mode=mode)

print("Error graphs generated and saved for all modes.")