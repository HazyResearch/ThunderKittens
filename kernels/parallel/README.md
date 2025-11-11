# ThunderKittens Multi-GPU Kernels ("ParallelKittens")

Each directory contains a single operator (e.g., fused all-gather GEMM). All kernels assume 8 GPUs; in order to change this, you must modify the `static constexpr int NUM_DEVICES` field in the kernel code.

## How to run

First export the `GPU` environment variable.

```bash
export GPU=H100 # or B200
```

Note that B200 is supported if the directory includes a `*_b100.cu` file or a file with no GPU-specific suffix. That is, if only a `*_h100.cu` file is present, B200 is not supported.

```bash
cd <directory> # navigate to the desired operator directory
make run      # compiles and executes with an 8-GPU torchrun configuration
```

Or you can compile without execution by simply running `make`.
