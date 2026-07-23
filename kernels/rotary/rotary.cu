#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/cuda_utils.cuh"

#ifdef TORCH_COMPILE
#define TK_COMPILE_FUSED_ROTARY
#endif

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcsf;
template<int _headdim, int _warps> struct rotary_layout {
    static constexpr int headdim = _headdim, warps = _warps;
    using seq_tile    = st_bf<16, headdim>;
    using seq_global  = gl<bf16, -1, -1, -1, headdim, seq_tile>;
    using rope_global = gl<bf16,  1,  1, -1, headdim/2>;
    struct globals {
        seq_global o, x;
        rope_global sin, cos;
        int batches; // how many batches per block, for sizing grid
    };
    struct input_block    { seq_tile x[warps]; };
    struct output_block   { seq_tile o[warps]; };
    struct producer_state { int active_warps;  };
    struct consumer_state { rt_fl<16, headdim/2> sin, cos; }; // long-resident tiles
};
template<int _headdim> struct rotary_template {
    static constexpr int headdim=_headdim, NUM_CONSUMER_WARPS=8, NUM_BLOCKS=1, OUTPUT_PIPE_STAGES=3, INPUT_PIPE_STAGES=3;
    using layout = rotary_layout<headdim, NUM_CONSUMER_WARPS>;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        if(args.task_iter == 0) {
            args.num_iters = min(args.globals.batches, (int)(args.globals.x.batch()-blockIdx.y*args.globals.batches)) * args.globals.x.depth(); // batches*heads handled by block
        }
        else args.num_iters = -1;
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers();
            args.state.active_warps = min((int)NUM_CONSUMER_WARPS,
                                          (int)(args.globals.x.rows()/16 - blockIdx.x*NUM_CONSUMER_WARPS));
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                kittens::coord idx = { blockIdx.y*args.globals.batches+args.iter/args.globals.x.depth(),
                                       args.iter%args.globals.x.depth(),
                                       blockIdx.x*NUM_CONSUMER_WARPS,
                                       0 };
                warp::tma::expect_bytes(args.inputs_arrived, sizeof(layout::seq_tile)*args.state.active_warps);
                for(int i = 0; i < args.state.active_warps; i++) {
                    warp::tma::load_async(args.input.x[i], args.globals.x, {idx.b,idx.d,idx.r+i,idx.c}, args.inputs_arrived);
                }
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
                __syncwarp();
            }
        }
        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == args.iter%4) {
                kittens::coord idx = { blockIdx.y*args.globals.batches+args.iter/args.globals.x.depth(),
                                       args.iter%args.globals.x.depth(),
                                       blockIdx.x*NUM_CONSUMER_WARPS,
                                       0 };
                for(int i = 0; i < args.state.active_warps; i++) {
                    warp::tma::store_async(args.globals.o, args.output.o[i], {idx.b,idx.d,idx.r+i,idx.c});
                }
                warp::tma::store_async_read_wait();
                if(laneid() == 0) arrive(args.outputs_finished, 4);
                __syncwarp();
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS/4>();
            kittens::coord idx = { blockIdx.x*NUM_CONSUMER_WARPS + warpid(), 0 };
            warp::load(args.state.sin, args.globals.sin, idx); // could be better coalesced but doing just once
            warp::load(args.state.cos, args.globals.cos, idx);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            rt_fl<16, headdim> x;
            rt_fl<16, headdim/2> x1, x2, temp1, temp2;
            warp::load(x, args.input.x[warpid()]);
            if(laneid() == 0) arrive(args.inputs_finished);
            __syncwarp();
            for(int i = 0; i < headdim/32; i++) {
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    x1.tiles[0][i].data[j] = x.tiles[0][i].data[j];
                    x2.tiles[0][i].data[j] = x.tiles[0][i+headdim/32].data[j];
                }
            }
            warp::mul(temp1, x1, args.state.cos);
            warp::mul(temp2, x2, args.state.cos);
            warp::mul(x2, x2, -1.f);
            warp::mul(x1, x1, args.state.sin);
            warp::mul(x2, x2, args.state.sin);
            warp::add(temp1, temp1, x2);
            warp::add(temp2, temp2, x1);
            for(int i = 0; i < headdim/32; i++) {
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    x.tiles[0][i].data[j]            = temp1.tiles[0][i].data[j];
                    x.tiles[0][i+headdim/32].data[j] = temp2.tiles[0][i].data[j];
                }
            }
            warp::store(args.output.o[warpid()], x);
            __syncwarp();
            if(laneid() == 0) arrive(args.outputs_arrived);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            if(laneid() == 0) arrive(args.finish_finished); // nothing to do here
        }
    };
};

template<int ATTN_D>
void dispatch_fused_rotary(
    bf16 * d_o,
    bf16 * d_x,
    bf16 * d_sin_in,
    bf16 * d_cos_in,
    const int ATTN_B, const int ATTN_H, const int ATTN_N,
    cudaStream_t stream = nullptr
) {

    using rope_t = rotary_template<ATTN_D>;
    constexpr int BATCHES_PER_BLOCK = 4;

    using seq_globals   = typename rope_t::layout::seq_global;
    using rope_globals  = typename rope_t::layout::rope_global;
    using globals = typename rope_t::layout::globals;

    seq_globals Og{d_o, ATTN_B, ATTN_H, ATTN_N, nullptr};
    seq_globals Xg{d_x, ATTN_B, ATTN_H, ATTN_N, nullptr};
    rope_globals SINg{d_sin_in, nullptr, nullptr, ATTN_N, nullptr};
    rope_globals COSg{d_cos_in, nullptr, nullptr, ATTN_N, nullptr};
    globals g{Og, Xg, SINg, COSg, BATCHES_PER_BLOCK};

    unsigned long mem_size = (MAX_SHARED_MEMORY-2048);
    constexpr int ROWS_PER_BLOCK = rope_t::NUM_CONSUMER_WARPS * rope_t::layout::seq_tile::rows;
    kittens::detail::throw_if_cuda_error(
        cudaFuncSetAttribute(prototype::lcsf::kernel<rope_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size),
        "cudaFuncSetAttribute"
    );
    dim3 grid((ATTN_N+ROWS_PER_BLOCK-1)/ROWS_PER_BLOCK, (ATTN_B+BATCHES_PER_BLOCK-1)/BATCHES_PER_BLOCK);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<rope_t>);
    kittens::prototype::lcsf::kernel<rope_t><<<grid, block, mem_size, stream>>>(g);
    kittens::detail::throw_if_cuda_error(cudaGetLastError(), "rotary kernel launch");
}

#ifdef TK_COMPILE_FUSED_ROTARY
#include "pyutils/torchutils.cuh"
#include <iostream>
#include <ATen/Functions.h>

at::Tensor fused_rotary(
    const at::Tensor x,
    const at::Tensor cos_in,
    const at::Tensor sin_in
) {
    CHECK_INPUT(x);
    CHECK_INPUT(sin_in);
    CHECK_INPUT(cos_in);

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    
    TORCH_CHECK(B == x.size(0), "Batch size mismatch");
    TORCH_CHECK(H == x.size(1), "Head size mismatch");
    TORCH_CHECK(N == x.size(2), "Sequence length mismatch");
    TORCH_CHECK(x.size(3) == 64 || x.size(3) == 128, "Hidden size mismatch");

    TORCH_CHECK(x.size(2) % 16 == 0, "Sequence length must be multiple of 16");
    TORCH_CHECK(cos_in.size(0) % 16 == 0, "Sequence length must be multiple of 16");
    TORCH_CHECK(sin_in.size(0) % 16 == 0, "Sequence length must be multiple of 16");

    at::Tensor out = at::empty({B, H, N, x.size(3)}, x.options());

    // convert to bf16
    c10::BFloat16 *x_bf16 = x.data_ptr<c10::BFloat16>();
    c10::BFloat16 *sin_in_bf16 = sin_in.data_ptr<c10::BFloat16>();
    c10::BFloat16 *cos_in_bf16 = cos_in.data_ptr<c10::BFloat16>();
    c10::BFloat16 *out_bf16 = out.data_ptr<c10::BFloat16>();

    bf16 *d_x = reinterpret_cast<bf16*>(x_bf16);
    bf16 *d_sin_in = reinterpret_cast<bf16*>(sin_in_bf16);
    bf16 *d_cos_in = reinterpret_cast<bf16*>(cos_in_bf16);
    bf16 *d_out = reinterpret_cast<bf16*>(out_bf16);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if(x.size(3) == 64) {
        dispatch_fused_rotary<64>(
            d_out,
            d_x,
            d_sin_in,
            d_cos_in, 
            B, H, N, stream
        );
    }
    else {
        dispatch_fused_rotary<128>(
            d_out,
            d_x,
            d_sin_in,
            d_cos_in, 
            B, H, N, stream
        );
    }

    return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rotary", fused_rotary, "Rotary TK. Takes tensors (x, cos_in, sin_in). All tensors are bf16. Returns (B, H, N, 128) in bf16.");
}
#elif defined(TVM_FFI_COMPILE)
#include "pyutils/tvm_ffi_utils.cuh"

namespace {

void check_matching_shape(const tvm::ffi::TensorView &output,
                          const tvm::ffi::TensorView &input) {
  TVM_FFI_CHECK(output.ndim() == input.ndim(), ValueError)
      << "Output and input must have the same rank";
  for (int32_t axis = 0; axis < input.ndim(); ++axis) {
    TVM_FFI_CHECK(output.size(axis) == input.size(axis), ValueError)
        << "Output and input shape mismatch at axis " << axis;
  }
}

// The caller-owned output buffer must not overlap any input tensor.
// Violating this precondition results in undefined behavior.
void TkRotary(tvm::ffi::TensorView output, tvm::ffi::TensorView input,
              tvm::ffi::TensorView cos, tvm::ffi::TensorView sin) {
  kittens::tvm_ffi::check_same_device(output, input, cos, sin);
  TVM_FFI_CHECK(input.ndim() == 4, ValueError)
      << "tk_rotary expects input to be 4-dimensional";
  TVM_FFI_CHECK(output.ndim() == 4, ValueError)
      << "tk_rotary expects output to be 4-dimensional";
  TVM_FFI_CHECK(cos.ndim() == 2 && sin.ndim() == 2, ValueError)
      << "tk_rotary expects cos and sin to be 2-dimensional";
  check_matching_shape(output, input);

  constexpr DLDataType expected_dtype{kDLBfloat, 16, 1};
  TVM_FFI_CHECK(
      output.dtype() == expected_dtype && input.dtype() == expected_dtype &&
          cos.dtype() == expected_dtype && sin.dtype() == expected_dtype,
      TypeError)
      << "tk_rotary supports only bfloat16 tensors";

  const int64_t sequence_length = input.size(2);
  const int64_t head_dimension = input.size(3);
  TVM_FFI_CHECK(sequence_length % 16 == 0, ValueError)
      << "tk_rotary expects the sequence length to be a multiple of 16";
  TVM_FFI_CHECK(head_dimension == 64 || head_dimension == 128, ValueError)
      << "tk_rotary expects head dimension 64 or 128";
  TVM_FFI_CHECK(cos.size(0) == sequence_length &&
                    sin.size(0) == sequence_length,
                ValueError)
      << "cos and sin sequence lengths must match input";
  TVM_FFI_CHECK(cos.size(1) == head_dimension / 2 &&
                    sin.size(1) == head_dimension / 2,
                ValueError)
      << "cos and sin widths must be half the input head dimension";

  bf16 *output_ptr = kittens::tvm_ffi::tensor_data_ptr<bf16>(output);
  bf16 *input_ptr = kittens::tvm_ffi::tensor_data_ptr<bf16>(input);
  bf16 *cos_ptr = kittens::tvm_ffi::tensor_data_ptr<bf16>(cos);
  bf16 *sin_ptr = kittens::tvm_ffi::tensor_data_ptr<bf16>(sin);

  kittens::tvm_ffi::check_data_alignment(output, 16);
  kittens::tvm_ffi::check_data_alignment(input, 16);
  kittens::tvm_ffi::check_data_alignment(cos, 16);
  kittens::tvm_ffi::check_data_alignment(sin, 16);

  tvm::ffi::CUDADeviceGuard guard(input.device().device_id);
  cudaStream_t stream = kittens::tvm_ffi::get_cuda_stream(input.device());
  if (head_dimension == 64) {
    dispatch_fused_rotary<64>(
      output_ptr, input_ptr, sin_ptr, cos_ptr,
      static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
      static_cast<int>(sequence_length), stream);
  }
  else {
    dispatch_fused_rotary<128>(
      output_ptr, input_ptr, sin_ptr, cos_ptr,
      static_cast<int>(input.size(0)), static_cast<int>(input.size(1)),
      static_cast<int>(sequence_length), stream);
  }
}

} // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(tk_rotary, TkRotary);
#else
#include "harness.impl"
#endif
