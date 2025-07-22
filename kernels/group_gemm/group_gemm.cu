#include "kittens.cuh"
#include "prototype.cuh"
#include "scheduler.cuh"
#include "utils.cuh"

#include <iomanip> 

#ifdef TORCH_COMPILE
#define TK_COMPILE_GROUP_GEMM
#endif

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

using scale_dtype = float;


template<ducks::rt::all T, ducks::rv::all V>
__device__ static inline void mul_add(T &dst, const T &src, const T &other, const V &row_values) {
    row_map<base_ops::fma_AxCtB, T, V>(dst, src, other, row_values);
}

template<int M_BLOCK>
struct matmul_layout {
    // tiles for the quantized inputs
    using  a_tile   = st_fl8_e4m3<64, 128>; 
    using  b_tile   = st_fl8_e4m3<128, 128>;
    using  c_tile   = st<bf16, 64, 128>;
    using  a_layout = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout = gl<fp8e4m3, 1, -1, -1, -1, b_tile>;
    using  c_layout = gl<bf16, 1, 1, -1, -1, c_tile>;

    using index_layout = gl<int, 1, 1, 1, -1>;
    // tiles for the dequantized inputs
    using a_vec    = sv_fl<64>; // scale_a
    using scale_a_layout = gl<scale_dtype, 1, 1, -1, -1, a_vec>;
    using scale_b_layout = gl<scale_dtype, 1, -1, -1, -1>;

    template<typename T=float> using accum_tile = rt<T, 16, c_tile::cols>;

    struct globals        { 
        a_layout A; b_layout B; c_layout C; index_layout index;
        scale_a_layout scale_a; scale_b_layout scale_b;
    };

    struct input_block    {
        a_tile a[M_BLOCK]; b_tile b;
        a_vec scale_a_sv[M_BLOCK];
    };
    struct finish_block   { 
        c_tile c[M_BLOCK]; 
    };
    struct scratch_block  {
        float* scale_b; // scale_b is a single value for the whole group
    };
    struct common_state   {
         uint32_t block_m_idx, block_n_idx;
         uint32_t group_idx;
         scale_dtype scale_b; // scale_b is a single value for the whole group
         bool is_tma_multicast_valid;
     };
    struct consumer_state { 
        accum_tile<float> accum;// Changed to single tall accumulator
    };
};


template<int _M_BLOCK=2, int _SUPER_M=12>
struct matmul_template {
    static_assert(_M_BLOCK <= 2, "only support _M_BLOCK<=2");
    static constexpr int M_BLOCK = _M_BLOCK, SUPER_M = _SUPER_M, CLUSTER_BLOCKS = 2;
    using layout    = matmul_layout<M_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    template<bool PERISISTENT_GRID=true> __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(PERISISTENT_GRID ? 128 : M*N/(M_BLOCK*layout::c_tile::num_elements));
    }
    // ThunderKittens template functions
    __device__ static inline void common_setup(common_setup_args<layout> args, bool is_prepared = false) {
        if (is_prepared) {
            return;
        }

        auto my_scheduler = deep_gemm::Scheduler<deep_gemm::GemmType::GroupedContiguous, static_cast<uint32_t>(M_BLOCK*layout::c_tile::rows),
                    static_cast<uint32_t>(layout::c_tile::cols), 2, false>(
                        static_cast<uint32_t>(args.globals.C.rows()),
                        static_cast<uint32_t>(args.globals.C.cols()),
                        static_cast<uint32_t>(args.globals.B.depth()),
                        args.globals.index.raw_ptr);
        bool is_valid = my_scheduler.get_next_block(
            args.common.block_m_idx, args.common.block_n_idx, args.task_iter
        );
        if (is_valid) {
            args.num_iters = args.globals.A.cols() / layout::a_tile::cols;
            int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
            args.common.is_tma_multicast_valid = my_scheduler.is_tma_multicast_valid(args.common.block_m_idx);
            args.common.block_m_idx = args.common.block_m_idx * M_BLOCK + id;
            args.common.group_idx = args.globals.index.raw_ptr[args.common.block_m_idx  * layout::c_tile::rows];
        } else {
            args.num_iters = -1; // No more work to do
            return;
        }
    }

    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        __device__ static void load(producer_load_cluster_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input.a, args.input.scale_a_sv);
                #pragma unroll
                for(int i = 0; i < M_BLOCK; i++) {
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.block_m_idx+i, args.iter}, args.inputs_arrived);
                    tma::load_async(args.input.scale_a_sv[i], args.globals.scale_a, {args.iter, args.common.block_m_idx+i}, args.inputs_arrived);
                }

                if (args.common.is_tma_multicast_valid) {
                    if (cluster_ctarank() == 0) {
                        tma::cluster::expect(args.inputs_cluster_arrived, 0, args.input.b);
                        tma::cluster::expect(args.inputs_cluster_arrived, 1, args.input.b);
                        tma::cluster::load_async(args.input.b, args.globals.B,
                            {args.common.group_idx, args.common.block_n_idx, args.iter}, args.inputs_cluster_arrived, 0b0011);
                    }
                } else {
                    tma::expect(args.inputs_cluster_arrived, args.input.b);
                    tma::load_async(args.input.b, args.globals.B,
                        {args.common.group_idx, args.common.block_n_idx, args.iter}, args.inputs_arrived);
                }
            }
        }
    };


    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args, bool is_prepared = true, int iter = 0) {
            if (is_prepared) {
                warpgroup::increase_registers<232>(); // increase registers for consumers
                zero(args.state.accum);
            } else {
                args.common.scale_b = args.globals.scale_b[{
                    args.common.group_idx,
                    args.common.block_n_idx,
                    iter
                }];
            }
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            rt_fl<16, layout::c_tile::cols> accum_tmp;
            warpgroup::mm_ABt(
                accum_tmp,
                args.input.a[warpgroup::groupid()],
                args.input.b
            );
            col_vec<rt<scale_dtype, 16, layout::c_tile::cols>> scale_a_rv;
            warpgroup::load(scale_a_rv, args.input.scale_a_sv[warpgroup::groupid()]);
            mul(scale_a_rv, scale_a_rv, args.common.scale_b);
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
            mul_add(args.state.accum, accum_tmp, args.state.accum, scale_a_rv);
            
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if(warpgroup::warpid() == 0) {
                tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()],
                                {args.common.block_m_idx, args.common.block_n_idx});
                tma::store_async_read_wait();
            }
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};
template<typename mmt>
void inner_run(
    fp8e4m3 *d_A, fp8e4m3 *d_B, bf16 *d_C, int* index,
    scale_dtype *d_scale_a, scale_dtype *d_scale_b,
    size_t M, size_t N, size_t K, size_t groups,
    dim3 grid, dim3 block
) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using index_layout = typename mmt::layout::index_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, groups, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};
    index_layout index_g{index, nullptr, nullptr, nullptr, M};

    // scales
    using scale_a_layout = typename mmt::layout::scale_a_layout;
    using scale_b_layout = typename mmt::layout::scale_b_layout;
    scale_a_layout scale_a{d_scale_a, nullptr, nullptr, K/128, M};
    scale_b_layout scale_b{d_scale_b, nullptr, groups, N/128, K/128};

    globals G{Ag, Bg, Cg, index_g, scale_a, scale_b};
    prototype::lcf::cluster_kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
}


#ifdef TK_COMPILE_GROUP_GEMM
#include <ATen/cuda/CUDAContext.h> 
#include "pyutils/torch_helpers.cuh"

// A: M x K
// B: GROUP x N x K
// scale_a: K/128 x M
// scale_b: GROUP x N/128 x K/128
// index: M x 1, M[i] is the index of the group that A[i] belongs to B[M[i]], C[i]= A[i] @ B[M[i]]
// Returns: C: M x N
torch::Tensor& group_gemm(
    const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& scale_a,
    const torch::Tensor& scale_b, const torch::Tensor& index, torch::Tensor& C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(scale_a);
    CHECK_INPUT(scale_b);

    auto M = A.size(0);
    auto N = B.size(1);
    auto K = A.size(1);
    auto groups = B.size(0);
    TORCH_CHECK(scale_a.size(1) == M, "scale_a must have the row of  A.size(0)");
    TORCH_CHECK(scale_a.size(0) == K/128, "scale_a must have the row of  A.size(1)/128");

    TORCH_CHECK(scale_b.size(1) == N/128, "scale_b must have the row of  B.size(0)/128");
    TORCH_CHECK(scale_b.size(2) == K/128, "scale_b must have the row of  B.size(1)/128");
    TORCH_CHECK(B.size(2) == K, "B must have the same number of columns as A");
    TORCH_CHECK(M % 128 == 0, "M must be divisible by 128");
    TORCH_CHECK(N % 128 == 0, "N must be divisible by 128");
    TORCH_CHECK(K % 128 == 0, "K must be divisible by 128");
    TORCH_CHECK(A.dtype() == c10::ScalarType::Float8_e4m3fn, "A must have the same dtype as Float8_e4m3fn");
    TORCH_CHECK(B.dtype() == c10::ScalarType::Float8_e4m3fn, "B must have the same dtype as Float8_e4m3fn");
    TORCH_CHECK(scale_a.dtype() == c10::ScalarType::Float, "scale_a must have the same dtype as A");
    TORCH_CHECK(scale_b.dtype() == c10::ScalarType::Float, "scale_b must have the same dtype as B");
    TORCH_CHECK(index.dtype() == c10::ScalarType::Int, "index must have the same dtype as Int");
    TORCH_CHECK(C.dtype() == c10::ScalarType::BFloat16, "C must have the same dtype as BFloat16");

    c10::Float8_e4m3fn *A_fp8 = A.data_ptr<c10::Float8_e4m3fn>();
    c10::Float8_e4m3fn *B_fp8 = B.data_ptr<c10::Float8_e4m3fn>();

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3*>(A_fp8);
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3*>(B_fp8);
    int *index_ptr = index.data_ptr<int>();
    scale_dtype *d_scale_a = scale_a.data_ptr<scale_dtype>();
    scale_dtype *d_scale_b = scale_b.data_ptr<scale_dtype>();
    bf16 *d_C = reinterpret_cast<bf16*>(C.data_ptr<c10::BFloat16>());
    using mnt = matmul_template<2, 8>;
    dim3 grid(mnt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mnt>);
    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::cluster_kernel<mnt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    inner_run<mnt>(d_A, d_B, d_C, index_ptr, d_scale_a, d_scale_b, M, N, K, groups, grid, block);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}
#else
#endif

