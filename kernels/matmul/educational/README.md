
This folder builds up the H100 GEMM piece-by-piece. It is only for educational purposes. 

Change the ```.cu``` file in the ```Makefile``` to one of the following, then ```make clean && make && ./matmul```.

- Level 01: Simple for loop (float) -- this is faster than bf16 because bf16 gets implicitly converted to floats first on cuda cores
- Level 02: Simple for loop (bf16)
- Level 03: Use shared memory
- Level 04: Use tensor cores (WMMA)
- Level 05: Use tensor cores (WGMMA)
- Level 06: Use tensor memory accelerator (TMA) and double buffering 
- Level 07: Use work partitioning
- Level 08: Use multiple consumer warpgroups

What would you want to add next for peak performance?
- Deeper pipeline
- L2 reuse grid
- Persistent kernel

