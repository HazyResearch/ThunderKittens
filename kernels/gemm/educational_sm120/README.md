# ThunderKittens Educational GEMM Kernels (SM120)

This folder builds up the SM120 GEMM piece-by-piece. It is only for educational purposes. The TFLOPS are measured on RTX 5060Ti.

Change the `LEVEL` field in the `Makefile` to `01` - `09`, then `make clean && make run`.

- Level 01 (1.45 TFLOPs): Simple for loop (float)
- Level 02 (1.42 TFLOPs): Simple for loop (bf16)
- Level 03 (1.81 TFLOPs): Use shared memory
- Level 04 (3.58 TFLOPs): Use tensor cores (WMMA)
- Level 05 (7.79 TFLOPs): Use TMA for global<->shared memory transfers (+ WMMA)
- Level 06 (42.65 TFLOPs): Use pseudo warpgroup MMA
- Level 07 (45.94 TFLOPs): Use double buffering + shared-memory reuse + grid swizzling
- Level 08 (46.24 TFLOPs): Use TMA + pseudo warpgroup MMA with double buffering
- Level 09 (46.62 TFLOPs): Use TMA + pseudo warpgroup MMA with warp specialization

Note: SM120 does not support hardware WGMMA, so levels 06-09 use the pseudo warpgroup MMA path in `include/ops/group/mma/warpgroup_pseudo.cuh`.
