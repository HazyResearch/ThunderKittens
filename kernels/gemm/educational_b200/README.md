# ThunderKittens Educational GEMM Kernels (Blackwell)

This folder builds up the B200 GEMM piece-by-piece. It is only for educational purposes.

Change the `LEVEL` field in the `Makefile` to `01` - `09`, then `make clean && make run`.

- Level 01: Simple for loop (float) -- this is faster than bf16 because bf16 gets implicitly converted to floats first on cuda cores
- Level 02: Simple for loop (bf16)
- Level 03: Use shared memory
- Level 04: Use tensor cores (WMMA)
- Level 05: Use TMA for global<->shared memory transfers (+ WMMA)
- Level 06: Use tensor cores (tcgen05 MMA) with TMA
- Level 07: Use pipelined warp specialization (TMA loader + MMA issuer)
- Level 08: Use epilogue pipelining
- Level 09: Use 2-CTA cluster and warpgroup-level parallelism