import inspect
import warnings

import numpy as np
import pandas as pd
import torch


warnings.filterwarnings("ignore", message=".*not a leaf Tensor is being accessed.*")
warnings.filterwarnings("ignore", message=".*no current CUDA context.*")

DEFAULT_SEQ_LENS = [1024, 2048, 3072, 6144, 12288]


def efficiency(flops, time_us):
    tflops = flops / 1e12
    time_ms = time_us / 1e6
    return tflops / time_ms


def compute_flops(get_flops, b, n, dv, h, method_name):
    params = inspect.signature(get_flops).parameters
    value_by_name = {
        "b": b,
        "batch": b,
        "n": n,
        "seqlen": n,
        "seq_len": n,
        "dv": dv,
        "d": dv,
        "headdim": dv,
        "h": h,
        "heads": h,
        "nheads": h,
    }
    args = []
    kwargs = {}
    for param_name in params:
        if param_name in value_by_name:
            args.append(value_by_name[param_name])
        elif param_name == "causal":
            if "c=t" in method_name:
                kwargs["causal"] = True
            elif "c=f" in method_name:
                kwargs["causal"] = False
            else:
                kwargs["causal"] = False
        elif param_name == "mode":
            if "fwd_bwd" in method_name:
                kwargs["mode"] = "fwd_bwd"
            elif "bwd" in method_name:
                kwargs["mode"] = "bwd"
            else:
                kwargs["mode"] = "fwd"
    return get_flops(*args, **kwargs)


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

    flops = compute_flops(get_flops, b, n, dv, h, method_name)

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
    implementations_bwd=None,
    seq_lens=None,
    should_skip=None,
    dtype=torch.bfloat16,
    dataframe_options=None,
):
    print("Benchmarking the kernels...")
    print("============" * 4, name, "============" * 4)
    seq_lens = DEFAULT_SEQ_LENS if seq_lens is None else seq_lens
    should_skip = (lambda method_name, n, b, h, dv: False) if should_skip is None else should_skip
    implementation_sets = [implementations]
    if implementations_bwd is not None:
        implementation_sets.append(implementations_bwd)

    if dataframe_options is not None:
        for option_name, option_value in dataframe_options.items():
            pd.set_option(option_name, option_value)

    for current_implementations in implementation_sets:
        method2tflops = {}
        method2timing = {}

        for method_name, method in current_implementations.items():
            flops_result, timing_result = {}, {}
            if verbose:
                print(f"Method: {method_name}")
            for n in seq_lens:
                if should_skip(method_name, n, b, h, dv):
                    continue
                if verbose:
                    print(f"Sequence Length: {n}")
                tflops, timing = measure_efficiency(
                    dtype,
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
