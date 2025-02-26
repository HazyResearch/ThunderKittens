# Demo Attention Kernels

Welcome! This directory contains two (non-causal) attention inference kernels that demonstrate different approaches to high-performance CUDA programming using ThunderKittens.

We have two implementations here - one targeting the RTX 4090 (`4090.cu`) and another for the H100 (`h100_lcf.cu`), and both are competitive with Flash Attention 2/3 performance. (My rough measurements put the `4090.cu`, head dim 64 kernel at about 7% behind FA-2, and the `h100_lcf.cu`, head dim 64 kernel at about 1% ahead of FA-3.)

The H100 kernel leverages one of ThunderKittens' pipeline templates (load-compute-finish) to concisely implement a two-stage pipeline with minimal boilerplate. TK pipeline templates are also used in our matrix multiply, rotary, mamba, and fftconv kernels -- they're pretty flexible. In contrast, the 4090 kernel demonstrates how you would build the pipeline yourself, giving a bit more visibility into the mechanics of programming with TK.

Just for clarity, our reported attention performance numbers are using the kernels in the h100 directory (which are a few % faster than these examples), not these examples. These implementations aren't intended for production deployment - rather, they serve as simple examples to illustrate ThunderKittens' capabilities to help write efficient CUDA kernels while preserving readability, maintainability, and extensibility.

## Attention Tutorial

As an exercise, try to implement the following changes to the H100 kernel:
* Make the attention causal. It may be helpful to use the following register tile conversion:
```
/**
 * @brief Makes a register tile triangular by zeroing elements above the row index
 *
 * @tparam RT The type of the register tile.
 * @param dst[in,out] The register tile to be filled.
 * @param src[in] The register tile to copy from.
 * @param row_idx[in] The row index to triangularize from.
 * @param val[in] The value to fill with.
 */
template<ducks::rt::row_layout RT>
__device__ static inline void tril(RT &dst, const RT &src, const int row_idx, const typename base_types::packing<typename RT::dtype>::unpacked_type &val=0);
```
* Add PyTorch bindings for the kernel. It may be helpful to look at the [example_bind](../../example_bind) directory.

We have reference implementations of the above in `h100_lcf_causal.cu` and `h100_lcf_pybind.cu`.
To run them:
```
make causal
python gentests.py randn 2048 128 causal
./attn_fwd_causal randn_2048_128_causal_True.txt

make pybind
python test_h100_pytorch.py randn 2048 128 [causal|noncausal]
```