# ThunderKittens Multi-GPU Kernels ("ParallelKittens")

Each directory contains a single operator (e.g., fused all-gather GEMM).

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
