#include "src/kittens.cuh"

#define NUM_WORKERS (4) // this comes from the fact that we want a 64-long sliding window
using namespace kittens;


#define WINDOW_WIDTH (64)
static_assert(WINDOW_WIDTH%64==0 && WINDOW_WIDTH<=256);
#define WINDOW_TILES ((WINDOW_WIDTH/64)+1)
#define WINDOW_MINI_TILES ((WINDOW_WIDTH/16)+1)

__global__ __launch_bounds__(NUM_WORKERS*kittens::WARP_THREADS, 2)
void sliding_window(int n, int d, const bf16* __restrict__ __q__, const bf16* __restrict__ __k__, const bf16* __restrict__ __v__, bf16* __o__) {

    using G = kittens::group<NUM_WORKERS>;

    auto warpid        = kittens::warpid();
    auto block_start   = blockIdx.x*(n*64);
    const bf16 *_q = __q__ + block_start, *_k = __k__ + block_start, *_v = __v__ + block_start;
          bf16 *_o = __o__ + block_start;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);
    
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&k_smem)[WINDOW_TILES][NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, WINDOW_TILES, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&v_smem)[WINDOW_TILES][NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, WINDOW_TILES, NUM_WORKERS>();

    rt_bf_1x4<> q_reg, k_reg, v_reg;
    rt_fl_1x1<> att_block[WINDOW_MINI_TILES];
    rt_bf_1x1<> att_block_bf;
    rt_fl_1x4<> o_reg;
    rt_fl_1x1<>::col_vec max_vec, norm_vec;
    
    int qo_blocks = n / (q_reg.rows*NUM_WORKERS), kv_blocks = n / (q_reg.rows*NUM_WORKERS);

    int start_block = 0, last_block = WINDOW_TILES-1;
    for(auto qo_blk = 0; qo_blk < qo_blocks; qo_blk++, start_block=(start_block+1)%WINDOW_TILES, last_block=(last_block+1)%WINDOW_TILES) {

        __syncthreads(); // we need to make sure all warps are done before we can start loading the next kv chunk

        // load the curent k, v blocks into last_block. If qo_blk > 0, then the previous tiles stick around.
        load(k_smem[last_block][warpid], _k + (qo_blk*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
        load(v_smem[last_block][warpid], _v + (qo_blk*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);

        // load q registers
        load(q_reg, _q + (qo_blk*NUM_WORKERS + warpid)*q_reg.num_elements, q_reg.cols);
        mul(q_reg, q_reg, __float2bfloat16(0.125f)); // temperature adjustment

        neg_infty(max_vec); // zero registers for the Q chunk
        zero(norm_vec);
        zero(o_reg);

        __syncthreads(); // we need to make sure all memory is loaded before we can begin the compute phase

        for(int subtile = 0; subtile < WINDOW_MINI_TILES; subtile++) {
            int src_idx = warpid+subtile;
            if (4*qo_blk + src_idx >= 4*(WINDOW_TILES-1)) {

                load(k_reg, k_smem[(start_block+(src_idx/4))%WINDOW_TILES][src_idx%4]);

                zero(att_block[subtile]);
                dot(att_block[subtile], q_reg, k_reg, att_block[subtile]);
                if(subtile == WINDOW_MINI_TILES-1) {
                    // last tile becomes causal
                    make_causal(att_block[subtile], att_block[subtile], base_types::constants<float>::neg_infty());
                }
            }
            else {
                neg_infty(att_block[subtile]); // initial blocks must be zero
            }
        }
        // now do the softmax. first we subtract max for numerical stability. then exp.
        #pragma unroll
        for(int subtile = 0; subtile < WINDOW_MINI_TILES; subtile++) {
            row_max(max_vec, att_block[subtile], max_vec); // accumulate onto the max_vec
        }
        #pragma unroll
        for(int subtile = 0; subtile < WINDOW_MINI_TILES; subtile++) {
            sub_row(att_block[subtile], att_block[subtile], max_vec);
            exp(att_block[subtile], att_block[subtile]);
        }
        // now we sum so that we can divide.
        #pragma unroll
        for(int subtile = 0; subtile < WINDOW_MINI_TILES; subtile++) {
            row_sum(norm_vec, att_block[subtile], norm_vec);
        }
        #pragma unroll
        for(int subtile = 0; subtile < WINDOW_MINI_TILES; subtile++) {
            div_row(att_block[subtile], att_block[subtile], norm_vec);
        }
        for(int subtile = 0; subtile < WINDOW_MINI_TILES; subtile++) {
            int src_idx = warpid+subtile;
            load(v_reg, v_smem[(start_block+(src_idx/4))%WINDOW_TILES][src_idx%4]);
            rt_bf_1x4<ducks::rt_layout::col> &v_reg_col = swap_layout_inplace(v_reg); // this is a reference and the call has invalidated v_reg

            copy(att_block_bf, att_block[subtile]);
            mma(o_reg, att_block_bf, v_reg_col, o_reg); // accumulate
        }

        store(_o + (qo_blk*NUM_WORKERS + warpid)*q_reg.num_elements, o_reg, d); // write out o. compiler has an issue with register usage if d is made constexpr q_reg.rows :/
    }
}


// For testing via C++
// #include "harness.impl" // (comment out when using the code below)


// For binding to PyTorch (comment out include for harness.imple when using the code below)
#include "src/common/pyutils/torch_helpers.cuh"
#include <iostream>
void sliding_window_tk(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o) {
    std::cout << "Entered Sliding window handler" << std::endl;
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);
    
    auto batch = q.size(0);
    auto heads = q.size(1);
    auto threads = NUM_WORKERS * kittens::WARP_THREADS;
    auto n     = q.size(2);
    uint d     = q.size(3);

    bool k_same = true, v_same = true;
    for(auto i = 0; i < 2; i++) { 
        k_same &= q.size(i) == k.size(i);
        v_same &= q.size(i) == v.size(i);
    }
    k_same &= d == k.size(3);
    v_same &= d == v.size(3);
    v_same &= v.size(2) == n;
    
    // This is just a restriction of what we're doing now...
    TORCH_CHECK(k_same, "X and K_out should be same size");
    TORCH_CHECK(v_same, "X and V_out should be same size");
    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "Q is a Bfloat");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "K is a Bfloat");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "V is a Bfloat");
    TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "O is a Bfloat");
    TORCH_CHECK(n % (NUM_WORKERS*kittens::TILE_DIM) == 0, "The number of elements should be divisible the number of workers times stored fragments");

    // convert to bf16
    c10::BFloat16 *q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *v_ptr = v.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr = o.data_ptr<c10::BFloat16>();

    const bf16* q_bf = reinterpret_cast<const bf16*>(q_ptr);
    const bf16* k_bf = reinterpret_cast<const bf16*>(k_ptr);
    const bf16* v_bf = reinterpret_cast<const bf16*>(v_ptr);
          bf16* o_bf = reinterpret_cast<bf16*>(o_ptr);

    std::cout << "Checks and casts" << std::endl;
    unsigned long mem_size = kittens::MAX_SHARED_MEMORY;
    cudaFuncSetAttribute(
        sliding_window,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    std::cout << "Set dynamic memory" << std::endl;
    sliding_window<<<batch*heads,threads,mem_size>>>(n, d, q_bf, k_bf, v_bf, o_bf);

    // TODO: setup to launch with CUDA STREAM
    // auto stream_wrapper = at::cuda::getCurrentCUDAStream(q.device().index());
    // cudaStream_t stream = stream_wrapper.stream();
    // sliding_window<H,T><<<batch*head,threads,0,stream>>>(n, 
    //                     q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(), 
    //                     o.data_ptr<T>());

    std::cout << "Launched kernel" << std::endl;
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    std::cout << "Exiting" << std::endl;
}

