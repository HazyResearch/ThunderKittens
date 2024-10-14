import torch
import thunderkittens as tk
from tqdm import tqdm


def torch_attn(q, k, v, causal):
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    # forward pass
    output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
    # backward pass
    output.backward(torch.randn_like(output))
    return output, q.grad, k.grad, v.grad


def get_inputs(B, H, N, D, causal, i):

    # pick stdevs from [0.1 to 1.0]
    std1 = torch.rand(1).item() * 0.9 + 0.1
    std2 = torch.rand(1).item() * 0.9 + 0.1
    std3 = torch.rand(1).item() * 0.9 + 0.1
    # pick means from [-1.0 to 1.0]
    mean1 = torch.rand(1).item() * 2 - 1
    mean2 = torch.rand(1).item() * 2 - 1
    mean3 = torch.rand(1).item() * 2 - 1
    q = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda') * std1 + mean1
    k = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda') * std2 + mean2
    v = torch.randn(B, H, N, D, dtype=torch.bfloat16, device='cuda') * std3 + mean3

    if i % 50 == 0:
        # print magnitudes of tensors
        print(f"-- q: {torch.max(q)=}, {torch.min(q)=}, {torch.mean(q)=}, {torch.std(q)=}")
        print(f"-- k: {torch.max(k)=}, {torch.min(k)=}, {torch.mean(k)=}, {torch.std(k)=}")
        print(f"-- v: {torch.max(v)=}, {torch.min(v)=}, {torch.mean(v)=}, {torch.std(v)=}")

    o = torch.zeros_like(q)
    grad_output = torch.randn_like(q, requires_grad=False)

    qg = torch.zeros_like(q, requires_grad=False, dtype=torch.float32)
    kg = torch.zeros_like(k, requires_grad=False, dtype=torch.float32)
    vg = torch.zeros_like(v, requires_grad=False, dtype=torch.float32)

    l_vec = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 1, device=q.device, dtype=torch.float32)
    d_vec = torch.zeros(q.shape[0], q.shape[1], q.shape[2], 1, device=q.device, dtype=torch.float32, requires_grad=False)
    return q, k, v, o, grad_output, qg, kg, vg, l_vec, d_vec

def check_nans(q, k, v, o, l_vec, d_vec, grad_output, qg, kg, vg):
    has_nan = False
    for t in [q, k, v, o, l_vec, d_vec, grad_output, qg, kg, vg]:
        if torch.isnan(t).any():
            print("Found NaNs in tensor")
            has_nan = True
    return has_nan
        


def benchmark_attention(configurations):

    for B, H, N, D, causal in configurations:
        
        results_q_grad = []
        results_k_grad = []
        results_v_grad = []
        for i in tqdm(range(50000)):
            if i % 10 == 0:
                q, k, v, o, grad_output, qg, kg, vg, l_vec, d_vec = get_inputs(B, H, N, D, causal, i)

            o.zero_()
            l_vec.zero_()
            # torch.cuda.synchronize()
            tk.mha_forward(q, k, v, o, l_vec, causal)
            # torch.cuda.synchronize()
            
            # zero out gradients
            qg.zero_()
            kg.zero_()
            vg.zero_()
            d_vec.zero_()
            # torch.cuda.synchronize()
            tk.mha_backward(q, k, v, o, l_vec, d_vec, grad_output, qg, kg, vg, causal)
            # torch.cuda.synchronize()

            results_q_grad.append(qg.cpu().detach())
            results_k_grad.append(kg.cpu().detach())
            results_v_grad.append(vg.cpu().detach())

            # has_nans = check_nans(q, k, v, o, l_vec, d_vec, grad_output, qg, kg, vg)
            # has_nans_list.append(has_nans)
            # if sum(has_nans_list) > 0:
            #     print(f"Completed iteration {i}; Number of NaNs found: {sum(has_nans_list)}")

            # Nans
            if i % 100 == 0:
                print(f"Iteration: {i}")
                print(f"Nans in qg: {sum([torch.isnan(t).sum() for t in results_q_grad])}")
                print(f"Nans in kg: {sum([torch.isnan(t).sum() for t in results_k_grad])}")
                print(f"Nans in vg: {sum([torch.isnan(t).sum() for t in results_v_grad])}")
                results_q_grad = []
                results_k_grad = []
                results_v_grad = []

# Example list of configurations to test
configurations = [
    (16, 32, 768, 64, True),
]
results = benchmark_attention(configurations)
