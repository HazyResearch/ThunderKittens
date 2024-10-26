#include "kittens.cuh"
#include "prototype.cuh"

#ifdef TORCH_COMPILE
#define TK_COMPILE_MAMBA2
#endif

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcsf;
struct mamba2_fwd_layout {
	using q_tile   = st_bf<64, 64>;
	using k_tile   = st_bf<64, 64>;
	using v_tile   = st_bf<64, 64>;
	using o_tile   = st_bf<64, 64>;
    using a_vec    = sv_fl<64>; // decays
	using q_global = kittens::gl<bf16, -1, -1, -1, 64, q_tile>; // B, H, N, S
	using k_global = kittens::gl<bf16, -1, -1, -1, 64, k_tile>;
	using v_global = kittens::gl<bf16, -1, -1, -1, 64, v_tile>;
	using o_global = kittens::gl<bf16, -1, -1, -1, 64, o_tile>;
    using a_global = kittens::gl<float, -1, -1, 1, -1, a_vec>;
	struct globals { q_global Q; k_global K; v_global V; o_global O; a_global A; };
	struct input_block    { 
        q_tile q;
        k_tile k;
        v_tile v;
        a_vec  a;
        a_vec  padding[7];
    };
    struct output_block {
        o_tile o;
    };
	struct scratch_block  { 
        st_bf<64, 64> kv, k;
        a_vec         a_cumsum;
        a_vec         padding[7];
    };
    struct common_state {
        int batch, head;
    };
	struct consumer_state {
		rt_fl<16, 64> o_reg;
		rt_fl<16, 64> att_block;
		rt_bf<16, 64> att_block_mma;
        rt_fl<16, 64> local_decay;
        rt_bf<16, 64> q_reg, k_reg;
        rt_fl<16, 64> kv;
	};
};
struct mamba2_fwd_template {
	static constexpr int NUM_CONSUMER_WARPS = 4, NUM_BLOCKS=2,
        OUTPUT_PIPE_STAGES=2, INPUT_PIPE_STAGES=2;
	using layout = mamba2_fwd_layout;
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int task_id = args.task_iter * gridDim.x + blockIdx.x;
		args.common.batch = task_id / args.globals.V.depth; // batch = id / heads.
		task_id -= args.common.batch*args.globals.V.depth;
		args.common.head = task_id;
		args.num_iters = args.common.batch < args.globals.Q.batch ? args.globals.K.rows/layout::k_tile::rows : -1;
    }
	struct producer {
		__device__ static void setup(producer_setup_args<layout> args) {
			warpgroup::producer_registers();
		}
		__device__ static void load(producer_load_args<layout> args) {
			if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input.q, args.input.k, args.input.v, args.input.a);
                tma::load_async(args.input.q, args.globals.Q, {args.common.batch,                 0, args.iter, 0}, args.inputs_arrived);
                tma::load_async(args.input.k, args.globals.K, {args.common.batch,                 0, args.iter, 0}, args.inputs_arrived);
                tma::load_async(args.input.v, args.globals.V, {args.common.batch,  args.common.head, args.iter, 0}, args.inputs_arrived);
                tma::load_async(args.input.a, args.globals.A, {args.common.batch,  args.common.head, 0, args.iter}, args.inputs_arrived);
                if(laneid() == 0) arrive(args.inputs_arrived, 3);
            }
		}
        __device__ static void store(producer_store_args<layout> args) {
            if(warpgroup::warpid() == 0) {;
                tma::store_async(args.globals.O, args.output.o, {args.common.batch, args.common.head, args.iter, 0});
                tma::store_async_read_wait();
                __syncwarp();
                if(laneid() == 0) arrive(args.outputs_finished, 4);
            }
        }
	};
	struct consumer {
		__device__ static void setup(consumer_setup_args<layout> args) {
			warpgroup::increase_registers<224>();
            zero(args.state.kv);
		}
		__device__ static bool compute(consumer_compute_args<layout> args) {
            int warpgroupid = warpgroup::warpid()/kittens::WARPGROUP_WARPS;

            // Start by doing cumsum into shared memory
            warpgroup::sync(warpgroupid + 4);
            warpgroup::copy(args.scratch.a_cumsum, args.input.a);
            warpgroup::sync(warpgroupid + 4);
            if(warpid() <= 1) {
                // Perform the prefix sum (Hillis-Steele scan)
                for (int offset = 1; offset < 64; offset *= 2) {
                    float temp = (threadIdx.x >= offset) ? args.scratch.a_cumsum[threadIdx.x - offset] : 0.0f;
                    group<2>::sync(3);
                    args.scratch.a_cumsum[threadIdx.x] += temp;
                    group<2>::sync(3);
                }
            }
            warpgroup::sync(warpgroupid + 4); // cumulative sum done
            // Calculate decays
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                int base_row = warpgroup::warpid()*16 + laneid()/4;
                int base_col = i*16 + (laneid()%4)*2;
                args.state.local_decay.tiles[0][i].data[0].x = args.scratch.a_cumsum[base_row + 0] - args.scratch.a_cumsum[base_col + 0];
                args.state.local_decay.tiles[0][i].data[0].y = args.scratch.a_cumsum[base_row + 0] - args.scratch.a_cumsum[base_col + 1];
                args.state.local_decay.tiles[0][i].data[1].x = args.scratch.a_cumsum[base_row + 8] - args.scratch.a_cumsum[base_col + 0];
                args.state.local_decay.tiles[0][i].data[1].y = args.scratch.a_cumsum[base_row + 8] - args.scratch.a_cumsum[base_col + 1];
                args.state.local_decay.tiles[0][i].data[2].x = args.scratch.a_cumsum[base_row + 0] - args.scratch.a_cumsum[base_col + 8];
                args.state.local_decay.tiles[0][i].data[2].y = args.scratch.a_cumsum[base_row + 0] - args.scratch.a_cumsum[base_col + 9];
                args.state.local_decay.tiles[0][i].data[3].x = args.scratch.a_cumsum[base_row + 8] - args.scratch.a_cumsum[base_col + 8];
                args.state.local_decay.tiles[0][i].data[3].y = args.scratch.a_cumsum[base_row + 8] - args.scratch.a_cumsum[base_col + 9];
            }
            exp(args.state.local_decay, args.state.local_decay);
            // causal mask
            #pragma unroll
            for(int i = 0; i < 4; i++) { // causal mask
                auto &decay_subtile = reinterpret_cast<rt_fl<16,16>&>(args.state.local_decay.tiles[0][i]);
                if      (i >  warpgroup::warpid()) { zero       (decay_subtile); }
                else if (i == warpgroup::warpid()) { make_causal(decay_subtile, decay_subtile, kittens::base_types::constants<float>::zero()); }
            }
      		// A = Q @ K.T
            warpgroup::load(args.state.q_reg, args.input.q); // we need this later, anyways
			warpgroup::mm_ABt(args.state.att_block, args.state.q_reg, args.input.k);
			warpgroup::mma_async_wait();
            mul(args.state.att_block, args.state.att_block, args.state.local_decay);
            copy(args.state.att_block_mma, args.state.att_block);
            warpgroup::mm_AB(args.state.o_reg, args.state.att_block_mma, args.input.v);
            warpgroup::mma_async_wait();
            // // multiply q by decays
            {
                int base_row = warpgroup::warpid()*16 + laneid()/4;
                bf16 top = __float2bfloat16(expf(args.scratch.a_cumsum[base_row + 0]));
                bf16 bottom = __float2bfloat16(expf(args.scratch.a_cumsum[base_row +8]));
                #pragma unroll
                for(int i = 0; i < 4; i++) {
                    args.state.q_reg.tiles[0][i].data[0].x *= top;
                    args.state.q_reg.tiles[0][i].data[0].y *= top;
                    args.state.q_reg.tiles[0][i].data[1].x *= bottom;
                    args.state.q_reg.tiles[0][i].data[1].y *= bottom;
                    args.state.q_reg.tiles[0][i].data[2].x *= top;
                    args.state.q_reg.tiles[0][i].data[2].y *= top;
                    args.state.q_reg.tiles[0][i].data[3].x *= bottom;
                    args.state.q_reg.tiles[0][i].data[3].y *= bottom;
                }
            }
            warpgroup::store(args.scratch.kv, args.state.kv);
            warpgroup::sync(warpgroupid + 4);
            warpgroup::mma_AB(args.state.o_reg, args.state.q_reg, args.scratch.kv);
            warpgroup::mma_async_wait();
            warpgroup::store(args.output.o, args.state.o_reg);
            warpgroup::sync(warpgroupid + 4);
            float last_decay = args.scratch.a_cumsum[args.scratch.a_cumsum.length-1]; // last element
            float total_decay = expf(last_decay);
            mul(args.state.kv, args.state.kv, total_decay); // decay kv
            warpgroup::load(args.state.k_reg, args.input.k); // multiply k's by decays
            {
                int base_row = warpgroup::warpid()*16 + laneid()/4;
                bf16 top = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[base_row + 0]));
                bf16 bottom = __float2bfloat16(expf(last_decay - args.scratch.a_cumsum[base_row +8]));
                #pragma unroll
                for(int i = 0; i < 4; i++) {
                    args.state.k_reg.tiles[0][i].data[0].x *= top;
                    args.state.k_reg.tiles[0][i].data[0].y *= top;
                    args.state.k_reg.tiles[0][i].data[1].x *= bottom;
                    args.state.k_reg.tiles[0][i].data[1].y *= bottom;
                    args.state.k_reg.tiles[0][i].data[2].x *= top;
                    args.state.k_reg.tiles[0][i].data[2].y *= top;
                    args.state.k_reg.tiles[0][i].data[3].x *= bottom;
                    args.state.k_reg.tiles[0][i].data[3].y *= bottom;
                }
            }
            warpgroup::store(args.scratch.k, args.state.k_reg); // using as dummy memory
            warpgroup::sync(warpgroupid + 4);
            warpgroup::mma_AtB(args.state.kv, args.scratch.k, args.input.v);
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.outputs_arrived);
            if(laneid() == 0) arrive(args.inputs_finished);
		}
        __device__ static void finish(consumer_finish_args<layout> args) {
            if(laneid() == 0) arrive(args.finish_finished);
        }
	};
};

#ifdef TK_COMPILE_MAMBA2
#include "common/pyutils/torch_helpers.cuh"
#include <iostream>
#include <ATen/cuda/CUDAContext.h> 

void dispatch_mamba2(
    bf16 *d_q, bf16 *d_k, bf16 *d_v, 
    bf16 *d_o, float *d_a,
    int B, int H, int N
){
    // Add input validation
    if (!d_q || !d_k || !d_v || !d_o || !d_a) {
        throw std::runtime_error("Null pointer passed to dispatch_mamba2");
    }

    // printf("B %d, H %d, N %d\n", B, H, N);

    // Verify data before kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error before layout setup: %s\n", cudaGetErrorString(err));
    }

    mamba2_fwd_template::layout::q_global Qg(d_q, B, 1, N, nullptr);
    mamba2_fwd_template::layout::k_global Kg(d_k, B, 1, N, nullptr);
    mamba2_fwd_template::layout::a_global Ag(d_a, B, H, nullptr, N);
    mamba2_fwd_template::layout::v_global Vg(d_v, B, H, N, nullptr);
    mamba2_fwd_template::layout::o_global Og(d_o, B, H, N, nullptr);

    mamba2_fwd_template::layout::globals globals = {Qg, Kg, Vg, Og, Ag};
    
    // launch setup
    unsigned long mem_size = (kittens::MAX_SHARED_MEMORY/2)-2048;
    
    // Get current stream early
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    
    // Synchronize and check for errors
    cudaStreamSynchronize(stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after stream sync: %s\n", cudaGetErrorString(err));
    }
    
    cudaFuncSetAttribute(
        prototype::lcsf::kernel<mamba2_fwd_template>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    dim3 grid(264, 1, 1);
    constexpr int BLOCK_SIZE = prototype::detail::NUM_THREADS_v<mamba2_fwd_template>;
    
    // Record event before kernel
    // cudaEvent_t start, end;
    // cudaEventCreate(&start);
    // cudaEventCreate(&end);
    // cudaEventRecord(start, stream);

    prototype::lcsf::kernel<mamba2_fwd_template><<<grid, BLOCK_SIZE, mem_size, stream>>>(globals);

    // Record end event and check timing
    // cudaEventRecord(end, stream);
    cudaStreamSynchronize(stream);
    
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, end);
    // printf("Kernel execution time: %f ms\n", milliseconds);
    
    // Clean up events
    // cudaEventDestroy(start);
    // cudaEventDestroy(end);

    // Final error check
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after kernel: %s\n", cudaGetErrorString(err));
    }
}

torch::Tensor mamba2(
    const torch::Tensor q,
    const torch::Tensor k,
    const torch::Tensor v,
    const torch::Tensor a
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(a);

    // Verify tensors are contiguous
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");

    int B = v.size(0);
    int H = v.size(1);
    int N = v.size(2);
    int D = v.size(3);
    
    // Enhanced input validation
    TORCH_CHECK(q.size(0) == B, "q has incompatible batch");
    TORCH_CHECK(q.size(1) == 1, "q has incompatible heads");
    TORCH_CHECK(q.size(2) == N, "q has incompatible sequence shape");
    TORCH_CHECK(q.size(3) == D, "q has incompatible dimension");

    TORCH_CHECK(k.size(0) == B, "k has incompatible batch");
    TORCH_CHECK(k.size(1) == 1, "k has incompatible heads");
    TORCH_CHECK(k.size(2) == N, "k has incompatible sequence");
    TORCH_CHECK(k.size(3) == D, "k has incompatible dimension");

    TORCH_CHECK(v.size(0) == B, "v has incompatible batch");
    TORCH_CHECK(v.size(1) == H, "v has incompatible heads");
    TORCH_CHECK(v.size(2) == N, "v has incompatible sequence");
    TORCH_CHECK(v.size(3) == D, "v has incompatible dimension");

    TORCH_CHECK(a.size(0) == B, "a has incompatible batch");
    // TORCH_CHECK(a.size(1) == N, "a has incompatible sequence length");
    TORCH_CHECK(a.size(1) == H, "a has incompatible heads");

    // Create output tensor
    auto options = torch::TensorOptions()
        .dtype(q.dtype())
        .device(q.device())
        .requires_grad(q.requires_grad());
    torch::Tensor out = torch::empty({B, H, N, D}, options);

    // Verify output tensor
    TORCH_CHECK(out.is_contiguous(), "Output tensor must be contiguous");

    // Get data pointers with safety checks
    auto q_ptr = q.data_ptr<c10::BFloat16>();
    auto k_ptr = k.data_ptr<c10::BFloat16>();
    auto v_ptr = v.data_ptr<c10::BFloat16>();
    auto a_ptr = a.data_ptr<float>();
    auto out_ptr = out.data_ptr<c10::BFloat16>();

    TORCH_CHECK(q_ptr != nullptr, "q data pointer is null");
    TORCH_CHECK(k_ptr != nullptr, "k data pointer is null");
    TORCH_CHECK(v_ptr != nullptr, "v data pointer is null");
    TORCH_CHECK(a_ptr != nullptr, "a data pointer is null");
    TORCH_CHECK(out_ptr != nullptr, "output data pointer is null");

    // Safe pointer casting
    bf16 *d_q = reinterpret_cast<bf16*>(q_ptr);
    bf16 *d_k = reinterpret_cast<bf16*>(k_ptr);
    bf16 *d_v = reinterpret_cast<bf16*>(v_ptr);
    float *d_a = a_ptr;  // No need to reinterpret cast for float
    bf16 *d_o = reinterpret_cast<bf16*>(out_ptr);

    // Synchronize before dispatch
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream());
    
    dispatch_mamba2(d_q, d_k, d_v, d_o, d_a, B, H, N);
    
    // Synchronize after dispatch
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream().stream());
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    return out;
}
#else
#include "harness2.impl"
#endif