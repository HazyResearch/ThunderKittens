# ThunderKittens Educational GEMM Kernels (SM120)

This folder builds up the SM120 GEMM piece-by-piece. It is only for educational purposes.

Change the `LEVEL` field in the `Makefile` to `00` - `09`, then `make clean && make run`.

## Benchmark Results

| Level | Kernel | RTX 5060Ti TFLOPs | RTX 5090 TFLOPs |
| --- | --- | ---: | ---: |
| 00 | cuBLAS baseline | 46.44 | 225.922 |
| 01 | Simple for loop (float) | 1.45 | 6.75959 |
| 02 | Simple for loop (bf16) | 1.42 | 7.52689 |
| 03 | Use shared memory | 1.81 | 9.5925 |
| 04 | Use tensor cores (WMMA) | 3.58 | 18.2189 |
| 05 | Use TMA for global<->shared memory transfers (+ WMMA) | 7.79 | 42.1222 |
| 06 | Use pseudo warpgroup MMA | 42.65 | 199.427 |
| 07 | Use double buffering + shared-memory reuse + grid swizzling | 45.94 | 220.878 |
| 08 | Use TMA + pseudo warpgroup MMA with double buffering | 46.24 | 220.245 |
| 09 | Use TMA + pseudo warpgroup MMA with warp specialization | 46.62 | 225.244 |

The RTX 5090 numbers are from a single local run per level (n=1) with CUDA 13.0.88; every level reported `Error count: 0`.

Note: SM120 does not support hardware WGMMA, so levels 06-09 use the pseudo warpgroup MMA path in `include/ops/group/mma/warpgroup_pseudo.cuh`.
