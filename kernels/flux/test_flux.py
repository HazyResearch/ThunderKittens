from thunderkittens import tk_flux_linear_gate
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

def ref(x, linear, gate, y):
    return linear(x) * gate + y

def tk(x, linear, gate, y):
    return tk_flux_linear_gate(x, linear.weight, linear.bias, gate, y)

output_ref = ref(x, linear, gate, y)
output = tk(x, linear, gate, y)
print((output_ref-output).abs().max())

with torch.inference_mode():

    timing = do_bench(lambda: ref(x, linear, gate, y), warmup=1, rep=repeats) * 1e-3
    tflops = (2 * M * K * N) / 1e12 / (timing)
    print(f"Reference: {timing * 1e3:.2f}ms, {tflops:.2f}TFLOPS")

    timing = do_bench(lambda: tk(x, linear, gate, y), warmup=1, rep=repeats) * 1e-3
    tflops = (2 * M * K * N) / 1e12 / (timing)
    print(f"ThunderKittens: {timing * 1e3:.2f}ms, {tflops:.2f}TFLOPS")
