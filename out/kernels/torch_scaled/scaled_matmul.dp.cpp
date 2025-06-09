#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kittens.dp.hpp"
#include "prototype.dp.hpp"

#ifdef TORCH_COMPILE
#define TK_COMPILE_SCALED_MATMUL
#endif

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

using c_dtype = float;

struct matmul_layout {
    // tiles for the quantized inputs
    using  a_tile   = st_fl8_e4m3<64, 128>; 
    using  b_tile   = st_fl8_e4m3<128, 128>;
    using  c_tile   = st<c_dtype, 64, 128>;
    using  a_layout = gl<fp8e4m3, 1, 1, -1, -1, a_tile>;
    using  b_layout = gl<fp8e4m3, 1, 1, -1, -1, b_tile>;
    using  c_layout = gl<c_dtype, 1, 1, -1, -1, c_tile>;

    // tiles for the dequantized inputs
    using scale_a_layout = gl<c_dtype, 1, 1, 1, -1>;
    using scale_b_layout = gl<c_dtype, 1, 1, 1, -1>;

    template<typename T=float> using accum_tile = rt<T, 16, c_tile::cols>;

    struct globals        { 
        a_layout A; b_layout B; c_layout C; 
        scale_a_layout scale_a; scale_b_layout scale_b;
    };

    struct input_block    { 
        a_tile a[2]; b_tile b; 
    };
    struct finish_block   { 
        c_tile c[2]; 
    };
    struct scratch_block  {
    };
    struct common_state { sycl::int2 coord; };
    struct consumer_state { 
        accum_tile<c_dtype> accum;      // Changed to single tall accumulator
    };
};

template<int _SUPER_M=12>
struct matmul_template {
    static constexpr int SUPER_M = _SUPER_M;
    using layout    = matmul_layout;
    static constexpr int NUM_CONSUMER_WARPS=8, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    // Helper functions
    template <bool PERISISTENT_GRID = true>
    static inline dpct::dim3 grid(int M, int N, int K) {
        return dpct::dim3(PERISISTENT_GRID
                              ? 132
                              : M * N / (2 * layout::c_tile::num_elements));
    }
    // ThunderKittens template functions
    static inline void common_setup(common_setup_args<layout> args) {
        auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
        int Rblocks = args.globals.C.rows() / (2 * layout::c_tile::rows),
            Cblocks = args.globals.C.cols() / layout::c_tile::cols;
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;
        int task_id = args.task_iter * item_ct1.get_group_range(2) +
                      item_ct1.get_group(2);
        if (task_id < super_rows * Cblocks)
            args.common.coord = { SUPER_M*(task_id/super_repeat) + task_id%SUPER_M, (task_id%super_repeat)/SUPER_M };
        else if (task_id < Rblocks*Cblocks) {
            int remainder_id = task_id - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else { // Id is too high, no more work to do
            args.num_iters = -1;
            return;
        }
        args.num_iters = args.globals.A.cols()/layout::a_tile::cols;
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = {args.common.coord.x() * 2 + id,
                             args.common.coord.y()};
    }

    struct producer {
        static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                #pragma unroll
                for(int i = 0; i < 2; i++) {
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {args.common.coord.x() + i, args.iter},
                                    args.inputs_arrived);
                }
                tma::load_async(args.input.b, args.globals.B,
                                {args.common.coord.y(), args.iter},
                                args.inputs_arrived);
            }
        }
    };

    struct consumer {
        static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            zero(args.state.accum); 
        }
        static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_ABt(
                args.state.accum,
                args.input.a[warpgroup::groupid()],
                args.input.b
            );
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }
        /*
        DPCT1110:418: The total declared local variable size in device function
        finish exceeds 128 bytes and may cause high register pressure. Consult
        with your hardware vendor to find the total register size available and
        adjust the code, or use smaller sub-group size to avoid high register
        pressure.
        */
        static void finish(consumer_finish_args<layout> args) {
            col_vec<rt<c_dtype, 16, 128>> scale_a_rv;
            row_vec<rt<c_dtype, 16, 128>> scale_b_rv;
            warpgroup::load(scale_a_rv, args.globals.scale_a,
                            {args.common.coord.x()});
            load(scale_b_rv, args.globals.scale_b, {args.common.coord.y()});
            mul_col(args.state.accum, args.state.accum, scale_b_rv);
            mul_row(args.state.accum, args.state.accum, scale_a_rv);
            warpgroup::store(args.finish.c[warpgroup::groupid()], args.state.accum);
            warpgroup::sync(warpgroup::groupid()+4);
            if(warpgroup::warpid() == 0) {
                tma::store_async(
                    args.globals.C, args.finish.c[warpgroup::groupid()],
                    {args.common.coord.x(), args.common.coord.y()});
                tma::store_async_read_wait();
            }
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};

template <typename mmt>
void inner_run(fp8e4m3 *d_A, fp8e4m3 *d_B, c_dtype *d_C, c_dtype *d_scale_a,
               c_dtype *d_scale_b, size_t M, size_t N, size_t K,
               dpct::dim3 grid, dpct::dim3 block) {
    using a_layout = typename mmt::layout::a_layout;
    using b_layout = typename mmt::layout::b_layout;
    using c_layout = typename mmt::layout::c_layout;
    using globals  = typename mmt::layout::globals;
    a_layout Ag{d_A, nullptr, nullptr, M, K};
    b_layout Bg{d_B, nullptr, nullptr, N, K};
    c_layout Cg{d_C, nullptr, nullptr, M, N};

    // scales
    using scale_a_layout = typename mmt::layout::scale_a_layout;
    using scale_b_layout = typename mmt::layout::scale_b_layout;
    scale_a_layout scale_a{d_scale_a, nullptr, nullptr, nullptr, M};
    scale_b_layout scale_b{d_scale_b, nullptr, nullptr, nullptr, N};

    globals G{Ag, Bg, Cg, scale_a, scale_b};
    /*
    DPCT1049:419: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    {
        auto exp_props = sycl::ext::oneapi::experimental::properties{
            sycl::ext::oneapi::experimental::use_root_sync};

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(MAX_SHARED_MEMORY - 1024), cgh);

            cgh.depends_on(
                dpct::get_current_device().get_in_order_queues_last_events());

            cgh.parallel_for(
                sycl::nd_range<3>(grid * block, block), exp_props,
                [=](sycl::nd_item<3> item_ct1) {
                    prototype::lcf::kernel<mmt>(
                        G, dpct_local_acc_ct1
                               .get_multi_ptr<sycl::access::decorated::no>()
                               .get());
                });
        });
    }
}


#ifdef TK_COMPILE_SCALED_MATMUL
#include <ATen/cuda/CUDAContext.h> 
#include "pyutils/torch_helpers.cuh"

torch::Tensor scaled_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor scale_a, torch::Tensor scale_b) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(scale_a);
    CHECK_INPUT(scale_b);

    auto M = A.size(0);
    auto N = B.size(0);
    auto K = A.size(1);
    TORCH_CHECK(scale_a.size(0) == M, "scale_a must have the same number of rows as A");
    TORCH_CHECK(scale_b.size(0) == N, "scale_b must have the same number of rows as B");
    TORCH_CHECK(B.size(1) == K, "B must have the same number of columns as A");
    TORCH_CHECK(M % 64 == 0, "M must be divisible by 64");
    TORCH_CHECK(N % 128 == 0, "N must be divisible by 128");
    TORCH_CHECK(K % 128 == 0, "K must be divisible by 128");
    TORCH_CHECK(A.dtype() == c10::ScalarType::Float8_e4m3fn, "A must have the same dtype as Float8_e4m3fn");
    TORCH_CHECK(B.dtype() == c10::ScalarType::Float8_e4m3fn, "B must have the same dtype as Float8_e4m3fn");
    TORCH_CHECK(scale_a.dtype() == c10::ScalarType::Float, "scale_a must have the same dtype as A");
    TORCH_CHECK(scale_b.dtype() == c10::ScalarType::Float, "scale_b must have the same dtype as B");
    auto C_options = A.options().dtype(c10::ScalarType::Float);
    torch::Tensor C = torch::empty({M, N}, C_options);

    // convert to bf16
    c10::Float8_e4m3fn *A_fp8 = A.data_ptr<c10::Float8_e4m3fn>();
    c10::Float8_e4m3fn *B_fp8 = B.data_ptr<c10::Float8_e4m3fn>();

    fp8e4m3 *d_A = reinterpret_cast<fp8e4m3*>(A_fp8);
    fp8e4m3 *d_B = reinterpret_cast<fp8e4m3*>(B_fp8);
    c_dtype *d_scale_a = scale_a.data_ptr<c_dtype>();
    c_dtype *d_scale_b = scale_b.data_ptr<c_dtype>();
    c_dtype *d_C = C.data_ptr<c_dtype>();

    using mnt = matmul_template<8>;
    dim3 grid(mnt::grid(M, N, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mnt>);
    unsigned long mem_size = MAX_SHARED_MEMORY - 1024;
    cudaFuncSetAttribute(prototype::lcf::kernel<mnt>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);

    inner_run<mnt>(d_A, d_B, d_C, d_scale_a, d_scale_b, M, N, K, grid, block);
    CHECK_CUDA_ERROR(cudaGetLastError());
    return C;
}
#else
#include "harness.impl"
#endif