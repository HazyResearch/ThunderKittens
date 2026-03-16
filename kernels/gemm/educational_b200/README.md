# ThunderKittens Educational GEMM Kernels (Blackwell)

This folder builds up the B200 GEMM piece-by-piece. It is only for educational purposes.

Change the `LEVEL` field in the `Makefile` to `01` - `09`, then `make clean && make run`.

- Level 01 (6 TFLOPs): Simple for loop (float) -- this is faster than bf16 because bf16 gets implicitly converted to floats first on cuda cores
- Level 02 (6 TFLOPs): Simple for loop (bf16)
- Level 03 (11 TFLOPs): Use shared memory
- Level 04 (26 TFLOPs): Use tensor cores (WMMA)
- Level 05 (55 TFLOPs): Use TMA for global<->shared memory transfers (+ WMMA)
- Level 06 (293 TFLOPs): Use tensor cores (tcgen05 MMA) with TMA
- Level 07 (731 TFLOPs): Use pipelined warp specialization (TMA loader + MMA issuer)
- Level 08 (1050 TFLOPs): Use epilogue pipelining
- Level 09 (1285 TFLOPs): Use 2-CTA cluster and warpgroup-level parallelism

Note: Our full kernel in ``../bf16_b200`` gives 1540 TFLOPs for the default GEMM size in these examples ($M=N=K=4096$).
