# TODO: Add group scope for TMA ops

Part of the reason to do this is to go through and ensure that the synchronization scope of all ops is done consistently -- for example, the lack of a __syncwarp()on tma async wait's is not ideal.

This is also likely a good chance to go through and unify all of the async mechanisms -- we are currently roughly exposing all of PTX's baggage in how that works, and it _probably_ could be hidden better.

# TODO: Add FP8 matmul support to ThunderKittens

My basic assumption for this todo is that we need to add:
1. FP8 register tile support
2. FP8 shared tile support.
3. MMA instructions
4. Loads and stores.

We don't need to provide support for:
1. Maps, reductions, etc, since 8-bit instructions don't even seem to be supported.
2. H100 features (WGMMA, TMA, etc.) All of these can be done. But, I think they should do after the first set.

This list is probably not quite comprehensive -- I'm sure there's something I've missed when thinking it through. But I think it's pretty _close_.

## src

In `common/base_types.cuh`:
1. Add wrappers for the `__nv_fp8_e5m2` and `__nv_fp8_e4m3` types, analogous to `kittens::bf16`.
2. Register the FP8 types in the `packing` and `convertor` structs, if you want to support conversions.

Note: I was thinking about how to handle the fact that the relevant instruction is `mma.sync.aligned.m16n8k16`. Option (1) is to template the width of the `rt_base` tile based on the type. Option (2) is to leave it as-is and instead handle it at `rt` level in the `ops/warp/register/mma.cuh`. The latter may require some ungodly `reinterpret_cast`s when the time comes but I suspect it will be less painful than (1). But I leave it up to you either way.

In `types/register/rt_base.cuh`:
1. Register packed fp8 types as allowable, so that they don't throw static asserts.
2. If using option (1), add templating to make the base tile _sometimes_ width 32. It unfortunately seems HMMA wants that, after all.
3. Add pretty wrappers for the FP8 types.

In `types/register/rt.cuh`:
1. Add pretty wrappers
2. If (1), may want to modify to handle different base tile widths.

In `types/shared/st.cuh`:
1. Add wrappers for whichever fp8 types you want to support. I recommend using unpacked types here, just for consistency.

In `types/shared/st_layout.cuh`:
1. Add an additional (type) template args for the shared_indexer struct, and specialize it for fp8. Since we're ignoring the H100, you only really need to specialize the `naive` layout and the `xor_swizzle`.

In `ops/warp/register/mma.cuh`:
1. Add HMMA 16832 wrapper for FP8. See [here](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#multiply-and-accumulate-instruction-mma) for the relevant instructions.
2. Template dot and mma to run on these tiles.
   
In `ops/warp/register/conversions.cuh`:
1. Add a transpose by unpacking, shuffling, and repacking. This will be horrendously slow compared to the builtin movmatrix for 16 bit types, but it is still worth making sure everything needed can be done.

In `ops/warp/memory/global_to_register.cuh`, `ops/warp/memory/shared_to_register.cuh`, and `ops/warp/memory/global_to_shared.cuh`:
1. Add additional templating / functions.

## tests

First, the testing infrastructure is written assuming bf16 in global, since that's been enough to test everything so far. But here it is really not. So, you'll want to start by modifying `testing_commons.cuh`, and in particular adding a type template to the `initialize` function, so that we can check fp8 loads and stores, too. The `validate` function will also need to be modified, but it looks more straightforward to me.

How hard you want to go on tests is up to you, but a few that would be good to add fp8 test for and run include:
1. `tests/warp/register/tile/mma.cu`
2. `tests/warp/memory/tile/*.cu`
3. `tests/warp/shared/conversuions.cuh`

Just to make sure everything is working as it should!

## bonus integration test

One relatively easy thing to do after all of the above would be to modify the `4090_ker.cu` and its `harness.impl` in `examples/attn_fwd` to be templated including FP8, and see how much faster we can get attention to run on the 4090 in FP8!

## full hopper support

The two things that would need to be done in addition are:
1. TMA -- this is just adding templates to `src/warp/memory/tile/tma.cuh`
2. WGMMA -- templating `src/group/wgmma`. One thing to remember: the transpose flag does NOT work on QGMMA :/ -- this is the main reason we didn't put it in before.