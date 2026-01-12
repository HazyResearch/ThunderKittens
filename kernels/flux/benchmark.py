import torch
import time

target = None
try:
    from _C import tk_flux_linear_gate
    target = "flux_gate"
except:
    pass
try:
    from _C import tk_flux_linear_gelu
    target = "flux_gelu"
except:
    pass
if not target:
    print("Compile the kernels first")
    quit()

from triton.testing import do_bench

repeats = 100
warmups = 5

if target == "flux_gate":

    def ref_gate(x, linear, gate, y):
        return linear(x) * gate + y

    def tk_gate(x, linear, gate, y):
        return tk_flux_linear_gate(x, linear.weight, linear.bias, gate, y)

    def benchmark_gated(M, N, K):
        with torch.device('cuda'):
            x = torch.randn(M, K, dtype=torch.bfloat16)
            linear = torch.nn.Linear(K, N, bias=True).to(torch.bfloat16)
            gate = torch.randn(N, dtype=torch.bfloat16)
            y = torch.randn(M, N, dtype=torch.bfloat16)

        with torch.inference_mode():

            timing = do_bench(lambda: ref_gate(x, linear, gate, y), warmup=warmups, rep=repeats) * 1e-3
            tflops = (2 * M * K * N) / 1e12 / (timing)
            print(f"PyTorch Reference: {timing * 1e3:.3f}ms, {tflops:.2f}TFLOPS")

            timing = do_bench(lambda: tk_gate(x, linear, gate, y), warmup=warmups, rep=repeats) * 1e-3
            tflops = (2 * M * K * N) / 1e12 / (timing)
            print(f"ThunderKittens: {timing * 1e3:.3f}ms, {tflops:.2f}TFLOPS")

    gated_size = [
        (3072, 3072, 12288),
        (512, 3072, 12288),
        (256, 3072, 12288),
        (3584, 3072, 15360),
        (3328, 3072, 15360)
    ]

    print('Benchmarking Gated')
    for size in gated_size:
        print('(M, N, K):', size)
        benchmark_gated(*size)
        print()

if target == "flux_gelu":

    def ref_gelu(x, linear):
        return torch.nn.functional.gelu(linear(x), approximate='tanh')

    def tk_gelu(x, linear):
        return tk_flux_linear_gelu(x, linear.weight, linear.bias)

    def benchmark_gelu(M, N, K):
        with torch.device('cuda'):
            x = torch.randn(M, K, dtype=torch.bfloat16)
            linear = torch.nn.Linear(K, N, bias=True).to(torch.bfloat16)

        with torch.inference_mode():

            timing = do_bench(lambda: ref_gelu(x, linear), warmup=warmups, rep=repeats) * 1e-3
            tflops = (2 * M * K * N) / 1e12 / (timing)
            print(f"PyTorch Reference: {timing * 1e3:.3f}ms, {tflops:.2f}TFLOPS")

            timing = do_bench(lambda: tk_gelu(x, linear), warmup=warmups, rep=repeats) * 1e-3
            tflops = (2 * M * K * N) / 1e12 / (timing)
            print(f"ThunderKittens: {timing * 1e3:.3f}ms, {tflops:.2f}TFLOPS")

    gelu_sizes = [
        (3072, 12288, 3072),
        (512, 12288, 3072),
        (256, 12288, 3072)
    ]

    print('Benchmarking Gelu')
    for size in gelu_sizes:
        print('(M, N, K):', size)
        benchmark_gelu(*size)
        print()
