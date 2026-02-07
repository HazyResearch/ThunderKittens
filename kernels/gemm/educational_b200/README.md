# ThunderKittens Educational GEMM Kernels (Blackwell)

This folder builds up the B200 GEMM piece-by-piece. It is only for educational purposes.

**Note:** All levels use transposed B layout (B stored as N x K, column-major).

Change the `LEVEL` field in the `Makefile` to `01` - `07`, then `make clean && make run`.

- Level 01: Simple for loop (float) -- this is faster than bf16 because bf16 gets implicitly converted to floats first on cuda cores
- Level 02: Simple for loop (bf16)
- Level 03: Use shared memory
- Level 04: Use tensor cores
- Level 05: Use tensor cores (tcgen05 mma) with TMA