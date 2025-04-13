#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

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

    Q_pgl q;
    K_pgl k;
    V_pgl v;
    O_pgl o;

    const int N;
};

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
        Os[i] = torch::empty({static_cast<const uint>(B), 
                              static_cast<const uint>(H_qo), 
                              static_cast<const uint>(N), 
                              static_cast<const uint>(D_h)},
                              Vs[i].options()); // put on device i
        d_Q[i] = reinterpret_cast<bf16*>(Qs[i].data_ptr<c10::BFloat16>());
        d_K[i] = reinterpret_cast<bf16*>(Ks[i].data_ptr<c10::BFloat16>());
        d_V[i] = reinterpret_cast<bf16*>(Vs[i].data_ptr<c10::BFloat16>());
        d_O[i] = reinterpret_cast<bf16*>(Os[i].data_ptr<c10::BFloat16>());
        streams[i] = at::cuda::getCurrentCUDAStream().stream();
        cudaStreamSynchronize(streams[i]);
        CHECK_CUDA_ERROR(cudaGetLastError());
    });

    using pglobals = fwd_pglobals<64>;

    pglobals::Q_pgl p_Q(device_ids, d_Q, B, H_qo, N, D_h);
    pglobals::K_pgl p_K(device_ids, d_K, B, H_kv, N, D_h);
    pglobals::V_pgl p_V(device_ids, d_V, B, H_kv, N, D_h);
    pglobals::O_pgl p_O(device_ids, d_O, B, H_qo, N, D_h);
    pglobals p_G{p_Q, p_K, p_V, p_O, static_cast<int>(N)};

    int mem_size = kittens::MAX_SHARED_MEMORY;
    auto threads  = NUM_WORKERS * kittens::WARP_THREADS;

    // TORCH_CHECK(seq_len % (CONSUMER_WARPGROUPS*kittens::TILE_DIM*4) == 0, "sequence length must be divisible by 192");
    // dim3 grid(seq_len/(CONSUMER_WARPGROUPS*kittens::TILE_ROW_DIM<bf16>*4), qo_heads, batch);

    club.execute([&](int i) {
        // cudaFuncSetAttribute(
        //     fwd_attend_ker<64, false>,
        //     cudaFuncAttributeMaxDynamicSharedMemorySize,
        //     mem_size
        // );
        // fwd_attend_ker<64, false><<<grid, (32*NUM_WORKERS), mem_size, streams[i]>>>(p_G);
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
}

#else

#endif
