#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
namespace py = pybind11;

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <vector>

#include "kittens.cuh"
#include "pyutils/torch_helpers.cuh"

constexpr int NUM_DEVICES         = 8;
constexpr int CONSUMER_WARPGROUPS = 3; 
constexpr int PRODUCER_WARPGROUPS = 1; 
constexpr int NUM_WARPGROUPS      = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS; 
constexpr int NUM_WORKERS         = NUM_WARPGROUPS * kittens::WARPGROUP_WARPS;

using namespace kittens;

template<int D> struct fwd_tile_dims {};
template<> struct fwd_tile_dims<64> {
    constexpr static int tile_width = 64;
    constexpr static int QO_height  = 4 * 16;
    constexpr static int KV_height  = 8 * 16;
    constexpr static int stages     = 4; 
};
template<> struct fwd_tile_dims<128> {
    constexpr static int tile_width = 128;
    constexpr static int QO_height  = 4 * 16;
    constexpr static int KV_height  = 8 * 16;
    constexpr static int stages     = 2;
};
template<int D> struct fwd_pglobals {
    using Q_tile = st_bf<fwd_tile_dims<D>::QO_height, fwd_tile_dims<D>::tile_width>;
    using K_tile = st_bf<fwd_tile_dims<D>::KV_height, fwd_tile_dims<D>::tile_width>;
    using V_tile = st_bf<fwd_tile_dims<D>::KV_height, fwd_tile_dims<D>::tile_width>;
    using O_tile = st_bf<fwd_tile_dims<D>::QO_height, fwd_tile_dims<D>::tile_width>;

    using Q_pgl = pgl<gl<bf16, -1, -1, -1, -1, Q_tile>, NUM_DEVICES, true>; 
    using K_pgl = pgl<gl<bf16, -1, -1, -1, -1, K_tile>, NUM_DEVICES, true>; 
    using V_pgl = pgl<gl<bf16, -1, -1, -1, -1, V_tile>, NUM_DEVICES, true>; 
    using O_pgl = pgl<gl<bf16, -1, -1, -1, -1, O_tile>, NUM_DEVICES, true>;

    Q_pgl Q;
    K_pgl K;
    V_pgl V;
    O_pgl O;

    const int N;
};

template<int D, bool is_causal>
__global__  __launch_bounds__(NUM_WORKERS * kittens::WARP_THREADS, 1)
void blockwise_attn_ker(const __grid_constant__ fwd_pglobals<D> p_G, const __grid_constant__ int dev_idx) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = fwd_tile_dims<D>;

    using q_tile    =         st_bf<K::QO_height, K::tile_width>;
    using k_tile    =         st_bf<K::KV_height, K::tile_width>;
    using v_tile    =         st_bf<K::KV_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::QO_height, K::tile_width>>;
    using o_tile    =         st_bf<K::QO_height, K::tile_width>;
    
    q_tile    (&q_smem)[CONSUMER_WARPGROUPS] = al.allocate<q_tile, CONSUMER_WARPGROUPS>();
    k_tile    (&k_smem)[K::stages]           = al.allocate<k_tile, K::stages          >();
    v_tile    (&v_smem)[K::stages]           = al.allocate<v_tile, K::stages          >();
    l_col_vec (&l_smem)[CONSUMER_WARPGROUPS] = al.allocate<l_col_vec, CONSUMER_WARPGROUPS>();
    auto      (*o_smem)                      = reinterpret_cast<o_tile(*)>(q_smem);
    
    int kv_blocks   = p_G.N / (K::KV_height);
    // int kv_head_idx = blockIdx.y / g.hr;
    int kv_head_idx = blockIdx.y;
    int seq_idx     = blockIdx.x * CONSUMER_WARPGROUPS; 

    __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived[K::stages], v_smem_arrived[K::stages], compute_done[K::stages];
    if (threadIdx.x == 0) { 
        init_semaphore(qsmem_semaphore, 0, 1); 
        for(int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1); 
            init_semaphore(v_smem_arrived[j], 0, 1); 
            init_semaphore(compute_done[j], CONSUMER_WARPGROUPS, 0); 
        }

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));

        for (int wg = 0; wg < CONSUMER_WARPGROUPS; wg++) {
            coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + wg, 0};
            tma::load_async(q_smem[wg], p_G.Q[dev_idx], q_tile_idx, qsmem_semaphore);
        }

        for (int j = 0; j < K::stages - 1; j++) {
            coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, j, 0};
            tma::expect_bytes(k_smem_arrived[j], sizeof(k_tile));
            tma::load_async(k_smem[j], p_G.K[dev_idx], kv_tile_idx, k_smem_arrived[j]);
            tma::expect_bytes(v_smem_arrived[j], sizeof(v_tile));
            tma::load_async(v_smem[j], p_G.V[dev_idx], kv_tile_idx, v_smem_arrived[j]);
        }
    }
    __syncthreads(); 

    int pipe_idx = K::stages - 1; 
    
    if(warpgroupid == NUM_WARPGROUPS-1) {
        warpgroup::decrease_registers<32>();      
        
        int kv_iters; 
        if constexpr (is_causal) {
            kv_iters = (seq_idx * (K::QO_height/kittens::TILE_ROW_DIM<bf16>)) - 1 + (CONSUMER_WARPGROUPS * (K::QO_height/kittens::TILE_ROW_DIM<bf16>)); 
            kv_iters = ((kv_iters / (K::KV_height/kittens::TILE_ROW_DIM<bf16>)) == 0) ? (0) : ((kv_iters / (K::KV_height/kittens::TILE_ROW_DIM<bf16>)) - 1);
        }
        else { kv_iters = kv_blocks-2; }

        if(warpid == NUM_WORKERS-4) {
            for (auto kv_idx = pipe_idx - 1; kv_idx <= kv_iters; kv_idx++) {
                coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, kv_idx + 1, 0};
                tma::expect_bytes(k_smem_arrived[(kv_idx+1)%K::stages], sizeof(k_tile));
                tma::load_async(k_smem[(kv_idx+1)%K::stages], p_G.K[dev_idx], kv_tile_idx, k_smem_arrived[(kv_idx+1)%K::stages]);
                tma::expect_bytes(v_smem_arrived[(kv_idx+1)%K::stages], sizeof(v_tile));
                tma::load_async(v_smem[(kv_idx+1)%K::stages], p_G.V[dev_idx], kv_tile_idx, v_smem_arrived[(kv_idx+1)%K::stages]);
                
                wait(compute_done[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            }
        }
    }
    else {
        warpgroup::increase_registers<160>();

        rt_fl<16, K::KV_height>  att_block;
        rt_bf<16, K::KV_height>  att_block_mma;
        rt_fl<16, K::tile_width> o_reg;
        
        col_vec<rt_fl<16, K::KV_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;
        
        neg_infty(max_vec);
        zero(norm_vec);
        zero(o_reg);

        int kv_iters; 
        if constexpr (is_causal) {
            kv_iters = (seq_idx * 4) - 1 + (CONSUMER_WARPGROUPS * 4);
            kv_iters = (kv_iters/8);
        }
        else { kv_iters = kv_blocks - 1; }

        wait(qsmem_semaphore, 0);

        for (auto kv_idx = 0; kv_idx <= kv_iters; kv_idx++) {
        
            wait(k_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2);
            warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[(kv_idx)%K::stages]);
            
            copy(max_vec_last_scaled, max_vec);
            if constexpr (D == 64) { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.125f); }
            else                   { mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f); }
            
            warpgroup::mma_async_wait();

            if constexpr (is_causal) {
                const int q_blk = (seq_idx * (K::QO_height/kittens::TILE_ROW_DIM<bf16>)) + warpid; 
                      int k_blk = (kv_idx * (K::KV_height/kittens::TILE_ROW_DIM<bf16>)); 

                #pragma unroll
                for(int _ = 0; k_blk == (kv_iters-1)*(K::KV_height/kittens::TILE_ROW_DIM<bf16>) || k_blk == (kv_iters)*(K::KV_height/kittens::TILE_ROW_DIM<bf16>); k_blk+=10000) {
                    #pragma unroll
                    for (auto j = 0; j < (K::KV_height/kittens::TILE_ROW_DIM<bf16>); j++) {
                        auto k_idx = k_blk + j;
                        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(att_block.tiles[0][j]);

                        if      (k_idx >  q_blk) { neg_infty  (attn_subtile); }
                        else if (k_idx == q_blk) { make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty()); }
                        __syncwarp();
                    }
                }
            }

            row_max(max_vec, att_block, max_vec);
            
            if constexpr (D == 64) { 
                mul(att_block, att_block,    1.44269504089f*0.125f); 
                mul(max_vec_scaled, max_vec, 1.44269504089f*0.125f);
            }
            else                   { 
                mul(att_block, att_block,    1.44269504089f*0.08838834764f); 
                mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
            }

            sub_row(att_block, att_block, max_vec_scaled);
            exp2(att_block, att_block);
            sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            exp2(max_vec_last_scaled,       max_vec_last_scaled);
            mul(norm_vec,            norm_vec,     max_vec_last_scaled);
            row_sum(norm_vec,  att_block, norm_vec);
            add(att_block, att_block, 0.f);
            copy(att_block_mma, att_block); 
            mul_row(o_reg, o_reg, max_vec_last_scaled); 

            wait(v_smem_arrived[(kv_idx)%K::stages], (kv_idx/K::stages)%2); 

            warpgroup::mma_AB(o_reg, att_block_mma, v_smem[(kv_idx)%K::stages]);
            warpgroup::mma_async_wait();

            if(warpgroup::laneid() == 0) arrive(compute_done[(kv_idx)%K::stages], 1);
        }

        div_row(o_reg, o_reg, norm_vec);
        warpgroup::store(o_smem[warpgroupid], o_reg); 
        warpgroup::sync(warpgroupid+4);

        if (warpid % 4 == 0) {
            coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + warpgroupid, 0};
            tma::store_async(p_G.O[dev_idx], o_smem[warpgroupid], o_tile_idx);
        }

        mul(max_vec_scaled,   max_vec_scaled, 0.69314718056f);
        log(norm_vec, norm_vec);
        add(norm_vec, norm_vec, max_vec_scaled);

        if constexpr (D == 64) { mul(norm_vec, norm_vec, -8.0f); }
        else                   { mul(norm_vec, norm_vec, -11.313708499f); }
    
        warpgroup::store(l_smem[warpgroupid], norm_vec);
        warpgroup::sync(warpgroupid+4);

        tma::store_async_wait();
    }
}

#ifdef TORCH_COMPILE

template <int I, int SIZE> struct CHECK_INPUTS {
    static inline void apply(const int64_t B,
                             const int64_t H_qo,
                             const int64_t H_kv,
                             const int64_t N,
                             const int64_t D_h,
                             const std::vector<torch::Tensor>& Qs,
                             const std::vector<torch::Tensor>& Ks,
                             const std::vector<torch::Tensor>& Vs) {
        CHECK_INPUT(Qs[I]);
        CHECK_INPUT(Ks[I]);
        CHECK_INPUT(Vs[I]);

        TORCH_CHECK(Qs[I].size(0) == B, "Q batch dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Ks[I].size(0) == B, "K batch dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Vs[I].size(0) == B, "V batch dimension (device ", I, ") does not match with other inputs");

        TORCH_CHECK(Qs[I].size(1) == H_qo, "QO head dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Ks[I].size(1) == H_kv, "KV head dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Vs[I].size(1) == H_kv, "KV head dimension (device ", I, ") does not match with other inputs");

        TORCH_CHECK(Qs[I].size(2) == N, "Q sequence length dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Ks[I].size(2) == N, "K sequence length dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Vs[I].size(2) == N, "V sequence length dimension (device ", I, ") does not match with other inputs");

        TORCH_CHECK(Qs[I].size(3) == D_h, "Q head dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Ks[I].size(3) == D_h, "K head dimension (device ", I, ") does not match with other inputs");
        TORCH_CHECK(Vs[I].size(3) == D_h, "V head dimension (device ", I, ") does not match with other inputs");
        
        CHECK_INPUTS<I + 1, SIZE>::apply(B, H_qo, H_kv, N, D_h, Qs, Ks, Vs);  
    }
};
template <int SIZE> struct CHECK_INPUTS<SIZE, SIZE> {
    static inline void apply(const int64_t B,
                             const int64_t H_qo,
                             const int64_t H_kv,
                             const int64_t N,
                             const int64_t D_h,
                             const std::vector<torch::Tensor>&, 
                             const std::vector<torch::Tensor>&, 
                             const std::vector<torch::Tensor>&) {}
};

torch::Tensor pgl_tensor(
    const std::vector<int64_t> &sizes,
    const at::ScalarType dtype,
    const std::vector<int> &device_ids,
    const int device_id,
    const bool requires_grad
);
torch::Tensor pgl_tensor(
    const std::vector<int64_t> &sizes,
    const at::ScalarType dtype,
    const int *device_ids,
    const int device_id,
    const bool requires_grad
);
torch::Tensor pgl_tensor(
    const torch::Tensor &other, 
    const std::vector<int> &device_ids, 
    const int device_id
);

// TODO: combine outputs before returning
std::vector<torch::Tensor> ring_attention_forward(
    const std::vector<torch::Tensor> &Qs, 
    const std::vector<torch::Tensor> &Ks, 
    const std::vector<torch::Tensor> &Vs, 
    bool causal
) {
    // Input checking (up to CHECK_INPUTS<...>) takes about 3us 
    TORCH_CHECK(Qs.size() == NUM_DEVICES, "Qs must be of size ", NUM_DEVICES);
    TORCH_CHECK(Ks.size() == NUM_DEVICES, "Ks must be of size ", NUM_DEVICES);
    TORCH_CHECK(Vs.size() == NUM_DEVICES, "Vs must be of size ", NUM_DEVICES);

    int64_t B    = Qs[0].size(0);
    int64_t H_qo = Qs[0].size(1);
    int64_t H_kv = Ks[0].size(1);
    int64_t N    = Qs[0].size(2); // per-block sequence length
    int64_t D_h  = Qs[0].size(3);

    TORCH_CHECK(H_qo >= H_kv, "QO heads must be greater than or equal to KV heads");
    TORCH_CHECK(H_qo % H_kv == 0, "QO heads must be divisible by KV heads");

    CHECK_INPUTS<0, NUM_DEVICES>::apply(B, H_qo, H_kv, N, D_h, Qs, Ks, Vs);

    // TODO: support different head sizes
    TORCH_CHECK(H_qo == H_kv, "For now, different head sizes not supported");
    // TODO: support different head dims
    TORCH_CHECK(D_h == 64, "For now, head dim must be 64");
    // TODO: support causal attention
    TORCH_CHECK(!causal, "Causal attention not supported yet");

    // Initialize the KC threadpool
    int device_ids[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;
    KittensClub club(device_ids, NUM_DEVICES);

    // Initialize output tensor, device pointers, and streams
    std::vector<torch::Tensor> Os(NUM_DEVICES);
    bf16 *d_Q[NUM_DEVICES];
    bf16 *d_K[NUM_DEVICES];
    bf16 *d_V[NUM_DEVICES];
    bf16 *d_O[NUM_DEVICES];
    cudaStream_t streams[NUM_DEVICES];
    club.execute([&](int i) {
        Os[i] = pgl_tensor({B, H_qo, N, D_h}, at::kBFloat16, device_ids, i, true);
        d_Q[i] = reinterpret_cast<bf16*>(Qs[i].data_ptr<c10::BFloat16>());
        d_K[i] = reinterpret_cast<bf16*>(Ks[i].data_ptr<c10::BFloat16>());
        d_V[i] = reinterpret_cast<bf16*>(Vs[i].data_ptr<c10::BFloat16>());
        d_O[i] = reinterpret_cast<bf16*>(Os[i].data_ptr<c10::BFloat16>());
        streams[i] = at::cuda::getCurrentCUDAStream().stream();
        cudaStreamSynchronize(streams[i]);
        CHECK_CUDA_ERROR(cudaGetLastError());
    });

    // Initialize the parallel global layouts
    using pglobals = fwd_pglobals<64>;

    pglobals::Q_pgl p_Q(device_ids, d_Q, B, H_qo, N, D_h);
    pglobals::K_pgl p_K(device_ids, d_K, B, H_kv, N, D_h);
    pglobals::V_pgl p_V(device_ids, d_V, B, H_kv, N, D_h);
    pglobals::O_pgl p_O(device_ids, d_O, B, H_qo, N, D_h);
    pglobals p_G{p_Q, p_K, p_V, p_O, static_cast<int>(N)};

    // Initialize and run the kernel
    TORCH_CHECK(N % (CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16> * 4) == 0, "sequence length must be divisible by 192");
    dim3 grid(N / (CONSUMER_WARPGROUPS * kittens::TILE_ROW_DIM<bf16> * 4), H_qo, B);
    constexpr int smem = kittens::MAX_SHARED_MEMORY;

    club.execute([&](int i) {
        cudaFuncSetAttribute(blockwise_attn_ker<64, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        blockwise_attn_ker<64, false><<<grid, NUM_WORKERS * kittens::WARP_THREADS, smem, streams[i]>>>(p_G, i);
        cudaStreamSynchronize(streams[i]);
        CHECK_CUDA_ERROR(cudaGetLastError());
    });

    return Os;
}

std::vector<torch::Tensor> ring_attention_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, 
    torch::Tensor l_vec, torch::Tensor og, bool causal
) {
    TORCH_CHECK(false, "Backward ring attention not implemented");
    return {q, k, v, o, l_vec, og};
}

struct pgl_tensor_context {
    int device_id;
    void *raw_ptr;
    size_t size;
};

void _pgl_tensor_deleter(void* ptr) {
    pgl_tensor_context *ctx = static_cast<pgl_tensor_context*>(ptr);
    pglCudaFree(ctx->device_id, ctx->raw_ptr, ctx->size);
    free(ctx);
}

torch::Tensor pgl_tensor(
    const std::vector<int64_t> &sizes,
    const at::ScalarType dtype,
    const int *device_ids,
    const int device_id,
    const bool requires_grad
) {
    TORCH_CHECK(device_id >= 0 && device_id < NUM_DEVICES, "Invalid device ID");

    // Calculate number of elements and bytes
    int64_t numel = 1;
    for (auto s : sizes) {
        TORCH_CHECK(s > 0, "Size dimensions must be positive");
        numel *= s;
    }

    // Allocate CUDA memory
    pgl_tensor_context *ctx = new pgl_tensor_context;
    ctx->device_id = device_id;
    ctx->raw_ptr = nullptr;
    ctx->size = numel * c10::elementSize(dtype);
    pglCudaMalloc<true>(NUM_DEVICES, const_cast<int*>(device_ids), device_id, &ctx->raw_ptr, ctx->size);

    // Construct Tensor
    c10::DataPtr data_ptr(ctx->raw_ptr, ctx, _pgl_tensor_deleter,
        c10::Device(c10::DeviceType::CUDA, device_id));
    at::TensorOptions options = at::TensorOptions().dtype(dtype).device(torch::kCUDA, device_id);
    at::Storage storage = at::Storage({}, ctx->size, std::move(data_ptr), nullptr, false);
    torch::Tensor tensor = at::empty(0, options).set_(storage, 0, at::IntArrayRef(sizes.data(), sizes.size()), {});
    tensor.set_requires_grad(requires_grad);

    // Sanity check. Can be removed in production code
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");

    return tensor;
}

torch::Tensor pgl_tensor(
    const std::vector<int64_t> &sizes,
    const at::ScalarType dtype,
    const std::vector<int> &device_ids,
    const int device_id,
    const bool requires_grad
) {
    TORCH_CHECK(device_id >= 0 && device_id < static_cast<int>(device_ids.size()), "Invalid device ID");
    return pgl_tensor(sizes, dtype, device_ids.data(), device_id, requires_grad);
}

torch::Tensor pgl_tensor(
    const torch::Tensor &other, 
    const std::vector<int> &device_ids, 
    const int device_id
) {
    TORCH_CHECK(device_id >= 0 && device_id < static_cast<int>(device_ids.size()), "Invalid device ID");

    bool on_gpu = other.device().is_cuda();
    if (on_gpu) {
        std::cerr << "WARNING (pgl_tensor): the given tensor is already on GPU. "
                  << "This will result in a redundant memory allocation and copy.\n";
    }
    
    // Allocate CUDA memory
    pgl_tensor_context *ctx = new pgl_tensor_context;
    ctx->device_id = device_id;
    ctx->raw_ptr = nullptr;
    ctx->size = other.nbytes();
    pglCudaMalloc<true>(NUM_DEVICES, const_cast<int*>(device_ids.data()), device_id, &ctx->raw_ptr, ctx->size);

    // Copy data
    cudaMemcpyKind copy_kind = on_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
    cudaMemcpy(ctx->raw_ptr, other.data_ptr(), ctx->size, copy_kind);

    // Construct Tensor (this is required because data_ptr is a smart pointer)
    c10::DataPtr data_ptr(ctx->raw_ptr, ctx, _pgl_tensor_deleter,
        c10::Device(c10::DeviceType::CUDA, device_id));
    at::TensorOptions options = other.options().device(torch::kCUDA, device_id); // includes dtype, device, layout
    at::Storage storage = at::Storage({}, ctx->size, std::move(data_ptr), nullptr, false);
    torch::Tensor tensor = at::empty(0, options).set_(storage, 0, other.sizes(), {});
    if (other.requires_grad()) tensor.set_requires_grad(true);

    // Sanity check. Can be removed in production code
    TORCH_CHECK(tensor.is_contiguous(), "Tensor must be contiguous");

    return tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ThunderKittens Ring Attention Kernels";
    m.def(
        "ring_mha_forward",  
        torch::wrap_pybind_function(ring_attention_forward),
        "Forward ring MHA"
    );
    m.def(
        "ring_mha_backward", 
        torch::wrap_pybind_function(ring_attention_backward), 
        "Backward ring MHA"
    );
    m.def(
        "pgl_tensor", 
        static_cast<torch::Tensor(*)(const torch::Tensor&, const std::vector<int>&, const int)>(&pgl_tensor),
        "Create a PGL tensor from existing tensor"
    );
    m.def(
        "pgl_tensor", 
        static_cast<torch::Tensor(*)(const std::vector<int64_t>&, const at::ScalarType, const std::vector<int>&, const int, const bool)>(&pgl_tensor),
        "Create a new PGL tensor from sizes and dtype"
    );
}

#else

#endif
