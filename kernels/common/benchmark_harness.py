import warnings

import numpy as np
import pandas as pd
import torch


warnings.filterwarnings("ignore", message=".*not a leaf Tensor is being accessed.*")
warnings.filterwarnings("ignore", message=".*no current CUDA context.*")


def efficiency(flops, time_us):
    tflops = flops / 1e12
    time_ms = time_us / 1e6
    return tflops / time_ms


def measure_efficiency(
    dtype,
    n,
    method_name,
    method,
    *,
    get_flops,
    b,
    h,
    dv,
    verbose=False,
    torch_compile=True,
):
    if verbose:
        print(f"{b=}, {n=}, {h=}, {dv=}")

    causal = "c=t" in method_name
    flops = get_flops(b, n, dv, h, causal=causal)

    outputs, times = method(dtype, b, h, n, dv, verbose=verbose, torch_compile=torch_compile)
    times_us = times * 1000

    eff = efficiency(flops, times_us)
    if verbose:
        print(
            f"Method {method_name} -- Efficiency: {eff:.2f} TFLOPS, "
            f"Time: {times_us:.4f} us and FLOPS: {flops:.2f}"
        )
    torch.cuda.empty_cache()
    return eff, times_us


def run_benchmark_main(
    *,
    name,
    implementations,
    get_flops,
    b,
    h,
    dv,
    verbose=True,
    torch_compile=False,
):
    print("Benchmarking the kernels...")
    print("============" * 4, name, "============" * 4)

    method2tflops = {}
    method2timing = {}

    for method_name, method in implementations.items():
        flops_result, timing_result = {}, {}
        if verbose:
            print(f"Method: {method_name}")
        for n in [1024, 2048, 3072, 6144, 12288]:
            if verbose:
                print(f"Sequence Length: {n}")
            tflops, timing = measure_efficiency(
                torch.bfloat16,
                n,
                method_name,
                method,
                get_flops=get_flops,
                b=b,
                h=h,
                dv=dv,
                verbose=verbose,
                torch_compile=torch_compile,
            )
            if tflops > 0:
                flops_result[n] = tflops
                timing_result[n] = timing
        method2tflops[method_name] = flops_result
        method2timing[method_name] = timing_result

    df = pd.DataFrame(method2tflops).replace(np.nan, "OOM", regex=True)
    print(df)
