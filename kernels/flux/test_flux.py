from thunderkittens import tk_flux_linear_gate, tk_flux_linear_gelu
import torch
import time

from triton.testing import do_bench

M = 3072
N = 3072
K = 12288
repeats = 100

with torch.device('cuda'):
    x = torch.randn(M, K, dtype=torch.bfloat16)
    linear = torch.nn.Linear(K, N, bias=True).to(torch.bfloat16)
    gate = torch.randn(N, dtype=torch.bfloat16)
    y = torch.randn(M, N, dtype=torch.bfloat16)

def ref_gate(x, linear, gate, y):
    return linear(x) * gate + y

def tk_gate(x, linear, gate, y):
    return tk_flux_linear_gate(x, linear.weight, linear.bias, gate, y)

def ref_gelu(x, linear):
    return torch.nn.functional.gelu(linear(x), approximate='tanh')

def tk_gelu(x, linear):
    return tk_flux_linear_gelu(x, linear.weight, linear.bias)

print('Linear + gate')
output_ref = ref_gate(x, linear, gate, y)
output = tk_gate(x, linear, gate, y)
print((output_ref-output).abs().max())

with torch.inference_mode():

    timing = do_bench(lambda: ref_gate(x, linear, gate, y), warmup=1, rep=repeats) * 1e-3
    tflops = (2 * M * K * N) / 1e12 / (timing)
    print(f"Reference: {timing * 1e3:.3f}ms, {tflops:.2f}TFLOPS")

    timing = do_bench(lambda: tk_gate(x, linear, gate, y), warmup=1, rep=repeats) * 1e-3
    tflops = (2 * M * K * N) / 1e12 / (timing)
    print(f"ThunderKittens: {timing * 1e3:.3f}ms, {tflops:.2f}TFLOPS")

print("Linear + gelu")
output_ref = ref_gelu(x, linear)
output = tk_gelu(x, linear)
print((output_ref-output).abs().max())

with torch.inference_mode():

    timing = do_bench(lambda: ref_gelu(x, linear), warmup=1, rep=repeats) * 1e-3
    tflops = (2 * M * K * N) / 1e12 / (timing)
    print(f"Reference: {timing * 1e3:.3f}ms, {tflops:.2f}TFLOPS")

    timing = do_bench(lambda: tk_gelu(x, linear), warmup=1, rep=repeats) * 1e-3
    tflops = (2 * M * K * N) / 1e12 / (timing)
    print(f"ThunderKittens: {timing * 1e3:.3f}ms, {tflops:.2f}TFLOPS")