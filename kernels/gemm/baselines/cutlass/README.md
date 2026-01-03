# CUTLASS Benchmarks

Here, we describe how we benchmarked the [CUTLASS](https://github.com/NVIDIA/cutlass) kernel performance.

1. Set up.
    ```bash
    git clone https://github.com/NVIDIA/cutlass.git
    cd cutlass
    mkdir build
    cd build
    ```

2. Before compiling the kernels, it is helpful to first look at what kernels are available to us. Run the below to generate a list of available kernels.
    ```bash
    cmake .. \
        -DCUTLASS_NVCC_ARCHS=100a \
        -DCUTLASS_UNITY_BUILD_ENABLED=OFF \
        -DCUTLASS_ENABLE_TESTS=OFF \
        -DCUTLASS_ENABLE_EXAMPLES=OFF \
        -DCUTLASS_LIBRARY_KERNELS=cutlass3x_sm100_*
    ```

3. Now, open up `cutlass/build/tools/library/generated_kernels.txt`. It will list every kernel that will be compiled. **We want to keep this list to less than 1,000 kernels**. Otherwise, it will take >100 years to compile. We can trim this by making `CUTLASS_LIBRARY_KERNELS` more specific, or using `CUTLASS_LIBRARY_EXCLUDE_KERNELS`. The rough GEMM kernel name rule is `{cutlass_api_verson}_{sm_number}_{tensorop/bstensorop}_{Atype}_{Btype}_{AccType}_{Ctype}_{Dtype}_{optional_mixed_dtype_config}_{mma_shape}_{cluster_shape}_{ABC_transpose}_{alignment}_{mma_ncta}`, where GEMM is defined as `D = alpha * (AB) + beta * C`. For transpose configuration, `n` refers to column-major layout (a column is contiguous in memory) and `t` refers to row-major layout (a row is contiguous in memory). Note that `n` and `t` notations are unintuitive because they come from Fortran, where matrices are column-major by default. When considering the transpose, A matrix is logically M x K, B matrix is logically K x N, and C/D matrices are logically M x N. Thus, row-major is K-major for A/C/D but N-major for B.

4. Re-run `cmake` to generate a build configuration of the kernels we actually want to compile. Below are some examples.
    ```bash
    # BF16_BF16_FP32_void_BF16 GEMM
    cmake .. \
        -DCUTLASS_NVCC_ARCHS=100a \
        -DCUTLASS_UNITY_BUILD_ENABLED=OFF \
        -DCUTLASS_ENABLE_TESTS=OFF \
        -DCUTLASS_ENABLE_EXAMPLES=OFF \
        -DCUTLASS_LIBRARY_KERNELS=cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_void_bf16*_tnt_*

    # MXFP8_MXFP8_FP32_void_BF16 GEMM
    cmake .. \
        -DCUTLASS_NVCC_ARCHS=100a \
        -DCUTLASS_UNITY_BUILD_ENABLED=OFF \
        -DCUTLASS_ENABLE_TESTS=OFF \
        -DCUTLASS_ENABLE_EXAMPLES=OFF \
        -DCUTLASS_LIBRARY_KERNELS=cutlass3x_sm100_bstensorop_gemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_bf16*_tnt_*

    # NVFP4_NVFP4_FP32_void_FP32 GEMM
    cmake .. \
        -DCUTLASS_NVCC_ARCHS=100a \
        -DCUTLASS_UNITY_BUILD_ENABLED=OFF \
        -DCUTLASS_ENABLE_TESTS=OFF \
        -DCUTLASS_ENABLE_EXAMPLES=OFF \
        -DCUTLASS_LIBRARY_KERNELS=cutlass3x_sm100_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_void_f32*_tnt_* # as of now, CUTLASS does not provide BF16 output kernel
    
    # All at once
    cmake .. \
      -DCUTLASS_NVCC_ARCHS=100a \
      -DCUTLASS_UNITY_BUILD_ENABLED=OFF \
      -DCUTLASS_ENABLE_TESTS=OFF \
      -DCUTLASS_ENABLE_EXAMPLES=OFF \
      -DCUTLASS_LIBRARY_KERNELS="cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_void_bf16*_tnt_*,cutlass3x_sm100_bstensorop_gemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_bf16*_tnt_*,cutlass3x_sm100_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_void_f32*_tnt_*"
    ```

5. Compile CUTLASS profiler. This will take a while.
    ```bash
    make cutlass_profiler -j64
    ```

6. Run CUTLASS profiler to find the best kernel configuration for the given MxNxK shape. Some options are `--kernels` (string filter for kernels to run), `--operation` (CUTLASS operation to profile), `--profiling-duration` (time to spend per kernel in milliseconds), `--min-iterations` (minimum number of iterations to spend profiling each kernel, even if `profiling-duration` is met), `--verification-enabled` (whether to do correctness check), `--dist` (data distribution of input tensors), and `--output` (path to output file. Operation name and `.csv` are automatically appended). Below are some examples.
    ```bash
    # BF16_BF16_FP32_void_BF16 GEMM
    for N in 1024 2048 4096 8192 16384; do
    ./tools/profiler/cutlass_profiler \
        --operation=gemm \
        --m=${N} \
        --n=${N} \
        --k=${N} \
        --output=bf16_${N} \
        --kernels="cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_void_bf16" \
        --warmup-iterations=500 \
        --profiling-iterations=100 \
        --verification-enabled=false \
        --dist="uniform,min:-1,max:1,scale:-1"
    done

    # MXFP8_MXFP8_FP32_void_BF16 GEMM
    for N in 1024 2048 4096 8192 16384; do
    ./tools/profiler/cutlass_profiler \
        --operation=block_scaled_gemm \
        --m=${N} \
        --n=${N} \
        --k=${N} \
        --output=mxfp8_${N} \
        --kernels="cutlass3x_sm100_bstensorop_gemm_ue8m0xe4m3_ue8m0xe4m3_f32_void_bf16" \
        --warmup-iterations=500 \
        --profiling-iterations=100 \
        --verification-enabled=false \
        --dist="uniform,min:-1,max:1,scale:-1"
    done

    # NVFP4_NVFP4_FP32_void_FP32 GEMM
    for N in 1024 2048 4096 8192 16384; do
    ./tools/profiler/cutlass_profiler \
        --operation=block_scaled_gemm \
        --m=${N} \
        --n=${N} \
        --k=${N} \
        --output=nvfp4_${N} \
        --kernels="cutlass3x_sm100_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_void_f32" \
        --warmup-iterations=500 \
        --profiling-iterations=100 \
        --verification-enabled=false \
        --dist="uniform,min:-1,max:1,scale:-1"
    done
    ```

(Optional) Use the below command to quickly find best configuration / TFLOPs number per shape:

```bash
python3 <<EOF
import glob
import pandas as pd

for f in sorted(glob.glob("*.csv")):
    df = pd.read_csv(f)
    df = df[df["Status"].str.contains("success", na=False)]
    df = df[df["OperationKind"].str.contains("gemm", na=False)]
    best = df.loc[df["GFLOPs"].idxmax()]

    print("-----------------------------------------")
    print(f"File:       {f}")
    print(f"Kernel:     {best['Operation']}")
    print(f"M,N,K:      {best['m']},{best['n']},{best['k']}")
    print(f"Runtime:    {best['Runtime']:.6f} s")
    print(f"TFLOPs:     {best['GFLOPs'] / 1000:.2f}")
    print(f"GB/s:       {best['GB/s']:.2f}")
    print("-----------------------------------------")
EOF
```

For more information on interpreting the profiler results, see: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/profiler.md
