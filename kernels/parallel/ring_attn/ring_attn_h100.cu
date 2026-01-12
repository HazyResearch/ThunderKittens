#include <kittens.cuh>
#include <prototype.cuh>
#include "pyutils/torchutils.cuh"

using namespace kittens;
using namespace kittens::prototype;

struct config {
    static constexpr int CLUSTER_SIZE = 1;

    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY = 227 * 1024 - STATIC_SHARED_MEMORY;

    static constexpr int CONSUMER_WARPGROUPS = 3; 
    static constexpr int PRODUCER_WARPGROUPS = 1; 
    static constexpr int NUM_WARPGROUPS      = CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS; 
    static constexpr int NUM_WARPS           = NUM_WARPGROUPS * WARPGROUP_WARPS; 
    static constexpr int NUM_THREADS         = NUM_WARPS * WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 32;
    static constexpr int CONSUMER_REGISTERS = 160;
};

struct globals {
    static constexpr int NUM_DEVICES = 8;

    static constexpr int D = 128;
    static constexpr int QO_BLOCK = 64; // for partial; for reduction it's 2x
    static constexpr int KV_BLOCK = 128;
    static constexpr int PIPELINE_STAGES = 2;

    using Q_tile = st_bf<QO_BLOCK, D>;
    using K_tile = st_bf<KV_BLOCK, D>;
    using V_tile = st_bf<KV_BLOCK, D>;
    using L_vec = col_vec<st_fl<QO_BLOCK, D>>;
    using O_tile = st_bf<QO_BLOCK, D>;
    using L_vec_2x = col_vec<st_fl<2 * QO_BLOCK, D>>; // for reduction
    using O_tile_2x = st_bf<2 * QO_BLOCK, D>; // for reduction

    using Q_gl = gl<bf16, -1, -1, -1, D, Q_tile>; // Batch, Head, Seq, Dim (full MHA)
    using K_pgl = pgl<gl<bf16, -1, -1, -1, D, K_tile>, NUM_DEVICES, false>;
    using V_pgl = pgl<gl<bf16, -1, -1, -1, D, V_tile>, NUM_DEVICES, false>;
    using L_gl = gl<float, 1, -1, -1, -1, L_vec, L_vec_2x>;
    using O_gl = gl<bf16, -1, -1, -1, D, O_tile, O_tile_2x>;
    using barrier_pgl = pgl<gl<int, -1, -1, -1, -1>, NUM_DEVICES, true>;

    Q_gl Q;
    K_pgl K0;
    K_pgl K1;
    V_pgl V0;
    V_pgl V1;
    L_gl L_block;
    L_gl L;
    O_gl O_block;
    O_gl O;

    barrier_pgl barrier;

    int ring_stage;
    const int dev_idx;
    const int num_comm_sms; // must be even

    __host__ inline int num_partial_blocks() const {
        return Q.batch() * Q.depth() * Q.rows() / (config::CONSUMER_WARPGROUPS * QO_BLOCK);
    }

    __host__ inline int num_reduction_blocks() const {
        return O.batch() * O.depth() * O.rows() / (config::CONSUMER_WARPGROUPS * QO_BLOCK * 2);
    }

    __host__ inline dim3 grid() const {
        throw std::runtime_error("Should not be called through utils.");
        return dim3(0, 0, 0);
    }
};

__device__ inline void attn_partial(const globals &G, const int block_idx) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);

    static_assert(sizeof(globals::Q_tile) * config::CONSUMER_WARPGROUPS + 
                  sizeof(globals::K_tile) * globals::PIPELINE_STAGES + 
                  sizeof(globals::V_tile) * globals::PIPELINE_STAGES + 
                  sizeof(globals::L_vec) * config::CONSUMER_WARPGROUPS +
                  sizeof(globals::O_tile) * config::CONSUMER_WARPGROUPS <= config::DYNAMIC_SHARED_MEMORY);
    typename globals::Q_tile (&Q_smem)[config::CONSUMER_WARPGROUPS] = al.allocate<typename globals::Q_tile, config::CONSUMER_WARPGROUPS>();
    typename globals::K_tile (&K_smem)[globals::PIPELINE_STAGES] = al.allocate<typename globals::K_tile, globals::PIPELINE_STAGES>();
    typename globals::V_tile (&V_smem)[globals::PIPELINE_STAGES] = al.allocate<typename globals::V_tile, globals::PIPELINE_STAGES>();
    typename globals::L_vec (&L_smem)[config::CONSUMER_WARPGROUPS] = al.allocate<typename globals::L_vec, config::CONSUMER_WARPGROUPS>();
    typename globals::O_tile (&O_smem)[config::CONSUMER_WARPGROUPS] = al.allocate<typename globals::O_tile, config::CONSUMER_WARPGROUPS>();

    const int num_heads = G.Q.depth();
    const int QO_blocks = G.Q.rows() / (config::CONSUMER_WARPGROUPS * globals::QO_BLOCK);
    const int KV_blocks = G.K0.rows() / globals::KV_BLOCK;
    const int batch_idx = block_idx / (QO_blocks * num_heads);
    const int head_idx = (block_idx % (QO_blocks * num_heads)) / QO_blocks; 
    const int QO_idx = (block_idx % QO_blocks) * config::CONSUMER_WARPGROUPS; 
    const int warpgroup_id = warpgroup::groupid();

    __shared__ kittens::semaphore Q_arrived[config::CONSUMER_WARPGROUPS];
    __shared__ kittens::semaphore L_arrived[config::CONSUMER_WARPGROUPS];
    __shared__ kittens::semaphore O_arrived[config::CONSUMER_WARPGROUPS];
    __shared__ kittens::semaphore K_arrived[globals::PIPELINE_STAGES];
    __shared__ kittens::semaphore V_arrived[globals::PIPELINE_STAGES];
    __shared__ kittens::semaphore compute_done[globals::PIPELINE_STAGES];
    if (threadIdx.x == 0) { 
        #pragma unroll
        for (int i = 0; i < config::CONSUMER_WARPGROUPS; i++) {
            init_semaphore(Q_arrived[i], 0, 1); 
            init_semaphore(L_arrived[i], 0, 1); 
            init_semaphore(O_arrived[i], 0, 1); 
        }
        #pragma unroll
        for(int i = 0; i < globals::PIPELINE_STAGES; i++) {
            init_semaphore(K_arrived[i], 0, 1); 
            init_semaphore(V_arrived[i], 0, 1); 
            init_semaphore(compute_done[i], config::CONSUMER_WARPGROUPS, 0); 
        }
    }
    __syncthreads(); 

    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>();

        for (int KV_idx = 0; KV_idx < KV_blocks; KV_idx++) {
            wait(compute_done[KV_idx % globals::PIPELINE_STAGES], (KV_idx / globals::PIPELINE_STAGES + 1) % 2);
            if (G.ring_stage % 2 == 0) {
                warpgroup::tma::expect_bytes(K_arrived[KV_idx % globals::PIPELINE_STAGES], sizeof(globals::K_tile));
                warpgroup::tma::load_async(K_smem[KV_idx % globals::PIPELINE_STAGES], G.K0[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, K_arrived[KV_idx % globals::PIPELINE_STAGES]);
                warpgroup::tma::expect_bytes(V_arrived[KV_idx % globals::PIPELINE_STAGES], sizeof(globals::V_tile));
                warpgroup::tma::load_async(V_smem[KV_idx % globals::PIPELINE_STAGES], G.V0[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, V_arrived[KV_idx % globals::PIPELINE_STAGES]);
            } else {
                warpgroup::tma::expect_bytes(K_arrived[KV_idx % globals::PIPELINE_STAGES], sizeof(globals::K_tile));
                warpgroup::tma::load_async(K_smem[KV_idx % globals::PIPELINE_STAGES], G.K1[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, K_arrived[KV_idx % globals::PIPELINE_STAGES]);
                warpgroup::tma::expect_bytes(V_arrived[KV_idx % globals::PIPELINE_STAGES], sizeof(globals::V_tile));
                warpgroup::tma::load_async(V_smem[KV_idx % globals::PIPELINE_STAGES], G.V1[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, V_arrived[KV_idx % globals::PIPELINE_STAGES]);
            }
        }
    } else {
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>();

        rt_fl<16, globals::KV_BLOCK> att_block;
        rt_bf<16, globals::KV_BLOCK> att_block_mma;
        rt_fl<16, globals::D> o_reg;

        col_vec<rt_fl<16, globals::KV_BLOCK>> max_vec;
        col_vec<rt_fl<16, globals::KV_BLOCK>> norm_vec;
        col_vec<rt_fl<16, globals::KV_BLOCK>> max_vec_last_scaled;
        col_vec<rt_fl<16, globals::KV_BLOCK>> max_vec_scaled;

        warpgroup::tma::expect_bytes(Q_arrived[warpgroup_id], sizeof(Q_smem[warpgroup_id]));
        warpgroup::tma::load_async(Q_smem[warpgroup_id], G.Q, {batch_idx, head_idx, QO_idx + warpgroup_id, 0}, Q_arrived[warpgroup_id]);

        warp::zero(norm_vec);
        warp::zero(o_reg);
        warp::neg_infty(max_vec);

        wait(Q_arrived[warpgroup_id], 0);

        for (auto KV_idx = 0; KV_idx < KV_blocks; KV_idx++) {
        
            wait(K_arrived[KV_idx % globals::PIPELINE_STAGES], (KV_idx / globals::PIPELINE_STAGES) % 2);
            warpgroup::mm_ABt(att_block, Q_smem[warpgroup_id], K_smem[KV_idx % globals::PIPELINE_STAGES]);
            
            warp::copy(max_vec_last_scaled, max_vec);
            warp::mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f);
            
            warpgroup::mma_async_wait();
            warp::row_max(max_vec, att_block, max_vec);

            warp::mul(att_block, att_block, 1.44269504089f*0.08838834764f); 
            warp::mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);

            warp::sub_row(att_block, att_block, max_vec_scaled);
            warp::exp2(att_block, att_block);
            warp::sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
            warp::exp2(max_vec_last_scaled, max_vec_last_scaled);
            warp::mul(norm_vec, norm_vec, max_vec_last_scaled);
            warp::row_sum(norm_vec,  att_block, norm_vec);
            warp::add(att_block, att_block, 0.f);
            warp::copy(att_block_mma, att_block); 
            warp::mul_row(o_reg, o_reg, max_vec_last_scaled); 

            wait(V_arrived[KV_idx % globals::PIPELINE_STAGES], (KV_idx / globals::PIPELINE_STAGES) % 2); 

            warpgroup::mma_AB(o_reg, att_block_mma, V_smem[KV_idx % globals::PIPELINE_STAGES]);
            warpgroup::mma_async_wait();

            warpgroup::arrive(compute_done[KV_idx % globals::PIPELINE_STAGES], 1);
        }

        warp::div_row(o_reg, o_reg, norm_vec);

        warpgroup::store(O_smem[warpgroup_id], o_reg); 
        warpgroup::sync(warpgroup_id + 4);
        if (G.ring_stage == 0)
            warpgroup::tma::store_async(G.O, O_smem[warpgroup_id], {batch_idx, head_idx, QO_idx + warpgroup_id, 0});
        else
            warpgroup::tma::store_async(G.O_block, O_smem[warpgroup_id], {batch_idx, head_idx, QO_idx + warpgroup_id, 0});

        warp::mul(max_vec_scaled, max_vec_scaled, 0.69314718056f);
        warp::log(norm_vec, norm_vec);
        warp::add(norm_vec, norm_vec, max_vec_scaled);
        // warp::mul(norm_vec, norm_vec, -11.313708499f); // Do not scale by -sqrt(D)

        warpgroup::store(L_smem[warpgroup_id], norm_vec);
        warpgroup::sync(warpgroup_id + 4);
        if (warpgroup::laneid() == 0) {
            if (G.ring_stage == 0)
                tma::store_async(G.L, L_smem[warpgroup_id], {batch_idx, head_idx, QO_idx + warpgroup_id});
            else
                tma::store_async(G.L_block, L_smem[warpgroup_id], {batch_idx, head_idx, QO_idx + warpgroup_id});
        }
    }
}

__device__ inline void attn_reduction(const globals &G, const int block_idx) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);

    static_assert(sizeof(globals::O_tile_2x) * config::CONSUMER_WARPGROUPS * 2 + 
                  sizeof(globals::L_vec_2x) * config::CONSUMER_WARPGROUPS * 2 <= config::DYNAMIC_SHARED_MEMORY);
    typename globals::O_tile_2x (&O_block_smem)[config::CONSUMER_WARPGROUPS] = al.allocate<typename globals::O_tile_2x, config::CONSUMER_WARPGROUPS>();
    typename globals::O_tile_2x (&O_smem)[config::CONSUMER_WARPGROUPS] = al.allocate<typename globals::O_tile_2x, config::CONSUMER_WARPGROUPS>();
    typename globals::L_vec_2x (&L_block_smem)[config::CONSUMER_WARPGROUPS] = al.allocate<typename globals::L_vec_2x, config::CONSUMER_WARPGROUPS>();
    typename globals::L_vec_2x (&L_smem)[config::CONSUMER_WARPGROUPS] = al.allocate<typename globals::L_vec_2x, config::CONSUMER_WARPGROUPS>();

    const int warpgroup_id = warpgroup::groupid();
    const int num_heads = G.O.depth();
    const int QO_blocks = G.O.rows() / (2 * globals::QO_BLOCK * config::CONSUMER_WARPGROUPS);
    const int batch_idx = block_idx / (QO_blocks * num_heads);
    const int head_idx = (block_idx % (QO_blocks * num_heads)) / QO_blocks; 
    const int QO_idx = (block_idx % QO_blocks) * config::CONSUMER_WARPGROUPS;

    __shared__ kittens::semaphore inputs_arrived[config::CONSUMER_WARPGROUPS];
    if (threadIdx.x == 0) { 
        #pragma unroll
        for (int i = 0; i < config::CONSUMER_WARPGROUPS; i++) {
            init_semaphore(inputs_arrived[i], 0, 1); 
        }
        #pragma unroll
        for (int i = 0; i < config::CONSUMER_WARPGROUPS; i++) {
            tma::expect_bytes(inputs_arrived[i], (sizeof(globals::L_vec_2x) + sizeof(globals::O_tile_2x)) * 2);
            tma::load_async(L_smem[i], G.L, {batch_idx, head_idx, QO_idx + i}, inputs_arrived[i]);
            tma::load_async(O_smem[i], G.O, {batch_idx, head_idx, QO_idx + i, 0}, inputs_arrived[i]);
            tma::load_async(L_block_smem[i], G.L_block, {batch_idx, head_idx, QO_idx + i}, inputs_arrived[i]);
            tma::load_async(O_block_smem[i], G.O_block, {batch_idx, head_idx, QO_idx + i, 0}, inputs_arrived[i]);
        }
    }
    __syncthreads(); 

    if (warpgroup_id == config::NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<config::PRODUCER_REGISTERS>(); // Unused warpgroup (24 is the min)
    } else {
        warpgroup::increase_registers<config::CONSUMER_REGISTERS>(); // This is the max (160 * 128 * 3 + 8 * 128 = 64512 < 65536)

        wait(inputs_arrived[warpgroup_id], 0);

        rt_fl<32, globals::D> O_reg;
        rt_fl<32, globals::D> O_block_reg;
        col_vec<rt_fl<32, globals::D>> L_reg;
        col_vec<rt_fl<32, globals::D>> L_block_reg;
        col_vec<rt_fl<32, globals::D>> L_new_reg;

        // L_new = L + torch.log(1 + torch.exp(L_block - L))
        warpgroup::load(L_reg, L_smem[warpgroup_id]);
        warpgroup::load(L_block_reg, L_block_smem[warpgroup_id]);
        warp::sub(L_new_reg, L_block_reg, L_reg);
        warp::exp(L_new_reg, L_new_reg);
        warp::add(L_new_reg, L_new_reg, 1.f);
        warp::log(L_new_reg, L_new_reg);
        warp::add(L_new_reg, L_new_reg, L_reg);
        warpgroup::store(L_smem[warpgroup_id], L_new_reg);
        
        // O = torch.exp(L - L_new).unsqueeze(-1) * O + torch.exp(L_block - L_new).unsqueeze(-1) * O_block
        warp::sub(L_reg, L_reg, L_new_reg);
        warp::exp(L_reg, L_reg); // torch.exp(L - L_new)
        warp::sub(L_block_reg, L_block_reg, L_new_reg);
        warp::exp(L_block_reg, L_block_reg); // torch.exp(L_block - L_new)
        warpgroup::load(O_reg, O_smem[warpgroup_id]);
        warp::mul_row(O_reg, O_reg, L_reg); // torch.exp(L - L_new).unsqueeze(-1) * O
        warpgroup::load(O_block_reg, O_block_smem[warpgroup_id]);
        warp::mul_row(O_block_reg, O_block_reg, L_block_reg); // torch.exp(L_block - L_new).unsqueeze(-1) * O_block
        warp::add(O_reg, O_reg, O_block_reg); // torch.exp(L - L_new).unsqueeze(-1) * O + torch.exp(L_block - L_new).unsqueeze(-1) * O_block
        warpgroup::store(O_smem[warpgroup_id], O_reg);

        warpgroup::sync(warpgroup_id + 4);

        if (warpgroup::laneid() == 0) {
            tma::store_async(G.O, O_smem[warpgroup_id], {batch_idx, head_idx, QO_idx + warpgroup_id, 0});
            tma::store_async(G.L, L_smem[warpgroup_id], {batch_idx, head_idx, QO_idx + warpgroup_id});
        }
    }
}

__device__ inline void attn_comm(const globals &G, const int block_idx) {
    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);

    static constexpr int NUM_CHUNKS = 7;
    static_assert(sizeof(globals::K_tile) * NUM_CHUNKS <= config::DYNAMIC_SHARED_MEMORY);
    typename globals::K_tile (&KV_smem)[NUM_CHUNKS] = al.allocate<typename globals::K_tile, NUM_CHUNKS>();

    const int warp_id = warp::groupid();
    const int num_batches = G.Q.batch();
    const int num_heads = G.Q.depth();
    const int KV_blocks = G.K0.rows() / globals::KV_BLOCK;
    const int num_blocks = num_batches * num_heads * KV_blocks;
    const int dst_dev_idx = (G.dev_idx + 1) % globals::NUM_DEVICES;

    __shared__ kittens::semaphore inputs_arrived[NUM_CHUNKS];
    __shared__ kittens::semaphore inputs_finished[NUM_CHUNKS];
    if (threadIdx.x == 0) { 
        #pragma unroll
        for (int i = 0; i < NUM_CHUNKS; i++) {
            init_semaphore(inputs_arrived[i], 0, 1); 
            init_semaphore(inputs_finished[i], 0, 1); 
        }
    }
    __syncthreads();

    uint32_t phasebits = 0xFFFF0000;

    if (warp_id < NUM_CHUNKS && laneid() == 0) { // GPU flex: use 1 thread per warp (needed for concurrency)
        int chunk_id = warp_id;

        for (int task_id = NUM_CHUNKS * (block_idx / 2) + chunk_id; task_id < num_blocks; task_id += NUM_CHUNKS * (G.num_comm_sms / 2)) {
            int batch_idx = task_id / (num_heads * KV_blocks);
            int head_idx = (task_id % (num_heads * KV_blocks)) / KV_blocks;
            int KV_idx = task_id % KV_blocks;

            wait(inputs_finished[chunk_id], get_phasebit<1>(phasebits, 0));
            update_phasebit<1>(phasebits, 0);

            tma::expect_bytes(inputs_arrived[chunk_id], sizeof(globals::K_tile));
            if (block_idx % 2 == 0) { // send K
                if (G.ring_stage % 2 == 0)
                    tma::load_async(KV_smem[chunk_id], G.K0[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, inputs_arrived[chunk_id]);
                else
                    tma::load_async(KV_smem[chunk_id], G.K1[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, inputs_arrived[chunk_id]);
            }
            else { // send V
                if (G.ring_stage % 2 == 0)
                    tma::load_async(KV_smem[chunk_id], G.V0[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, inputs_arrived[chunk_id]);
                else
                    tma::load_async(KV_smem[chunk_id], G.V1[G.dev_idx], {batch_idx, head_idx, KV_idx, 0}, inputs_arrived[chunk_id]);
            }
        }
    } else if (NUM_CHUNKS <= warp_id && warp_id < 2 * NUM_CHUNKS && laneid() == 0) {
        int chunk_id = warp_id - NUM_CHUNKS;

        for (int task_id = NUM_CHUNKS * (block_idx / 2) + chunk_id; task_id < num_blocks; task_id += NUM_CHUNKS * (G.num_comm_sms / 2)) {
            int batch_idx = task_id / (num_heads * KV_blocks);
            int head_idx = (task_id % (num_heads * KV_blocks)) / KV_blocks;
            int KV_idx = task_id % KV_blocks;

            wait(inputs_arrived[chunk_id], get_phasebit<0>(phasebits, 0));
            update_phasebit<0>(phasebits, 0);

            if (block_idx % 2 == 0) { // send K
                if (G.ring_stage % 2 == 0)
                    tma::store_async(G.K1[dst_dev_idx], KV_smem[chunk_id], {batch_idx, head_idx, KV_idx, 0});
                else
                    tma::store_async(G.K0[dst_dev_idx], KV_smem[chunk_id], {batch_idx, head_idx, KV_idx, 0});
            } else { // send V
                if (G.ring_stage % 2 == 0)
                    tma::store_async(G.V1[dst_dev_idx], KV_smem[chunk_id], {batch_idx, head_idx, KV_idx, 0});
                else
                    tma::store_async(G.V0[dst_dev_idx], KV_smem[chunk_id], {batch_idx, head_idx, KV_idx, 0});
            }

            tma::store_async_read_wait();
            arrive(inputs_finished[chunk_id]);
        }
    }
}

__device__ inline void attn_comm_partial_kernel(const globals &G) {
    if (blockIdx.x < G.num_comm_sms)
        attn_comm(G, blockIdx.x);
    else
        attn_partial(G, blockIdx.x - G.num_comm_sms);
}

__device__ inline void attn_reduction_kernel(const globals &G) {
    attn_reduction(G, blockIdx.x);
}

struct barrier_config {
    static constexpr int CLUSTER_SIZE = 1;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int NUM_THREADS = 1;
    static constexpr int DYNAMIC_SHARED_MEMORY = 0;
};

__device__ inline void barrier_kernel(const globals &G) {
    // Ensure all devices exit together
    barrier_all(G.barrier, {1, 0, 0}, G.dev_idx);
}

void entrypoint(
    const at::Tensor &Q,
    kittens::py::TKParallelTensor &K0,
    kittens::py::TKParallelTensor &K1,
    kittens::py::TKParallelTensor &V0,
    kittens::py::TKParallelTensor &V1,
    at::Tensor &L,
    at::Tensor &L_block,
    at::Tensor &O,
    at::Tensor &O_block,
    kittens::py::TKParallelTensor &barrier,
    const int ring_stage,
    const int num_comm_sms
) {
    globals G {
        .Q = kittens::py::tensor_to_gl<typename globals::Q_gl>(Q),
        .K0 = kittens::py::parallel_tensor_to_pgl<typename globals::K_pgl>(K0),
        .K1 = kittens::py::parallel_tensor_to_pgl<typename globals::K_pgl>(K1),
        .V0 = kittens::py::parallel_tensor_to_pgl<typename globals::V_pgl>(V0),
        .V1 = kittens::py::parallel_tensor_to_pgl<typename globals::V_pgl>(V1),
        .L_block = kittens::py::tensor_to_gl<typename globals::L_gl>(L_block),
        .L = kittens::py::tensor_to_gl<typename globals::L_gl>(L),
        .O_block = kittens::py::tensor_to_gl<typename globals::O_gl>(O_block),
        .O = kittens::py::tensor_to_gl<typename globals::O_gl>(O),
        .barrier = kittens::py::parallel_tensor_to_pgl<typename globals::barrier_pgl>(barrier),
        .ring_stage = ring_stage,
        .dev_idx = barrier.local_rank_,
        .num_comm_sms = num_comm_sms
    };

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    CUDACHECK(cudaFuncSetAttribute(kittens::py::global_kernel_unclustered<config, globals, attn_comm_partial_kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, config::DYNAMIC_SHARED_MEMORY));
    kittens::py::global_kernel_unclustered<config, globals, attn_comm_partial_kernel><<<num_comm_sms + G.num_partial_blocks(), config::NUM_THREADS, config::DYNAMIC_SHARED_MEMORY, stream>>>(G);

    if (ring_stage > 0) {
        CUDACHECK(cudaFuncSetAttribute(kittens::py::global_kernel_unclustered<config, globals, attn_reduction_kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, config::DYNAMIC_SHARED_MEMORY));
        kittens::py::global_kernel_unclustered<config, globals, attn_reduction_kernel><<<G.num_reduction_blocks(), config::NUM_THREADS, config::DYNAMIC_SHARED_MEMORY, stream>>>(G);
    }

    CUDACHECK(cudaFuncSetAttribute(kittens::py::global_kernel_unclustered<barrier_config, globals, barrier_kernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, barrier_config::DYNAMIC_SHARED_MEMORY));
    kittens::py::global_kernel_unclustered<barrier_config, globals, barrier_kernel><<<barrier_config::NUM_BLOCKS, barrier_config::NUM_THREADS, barrier_config::DYNAMIC_SHARED_MEMORY, stream>>>(G);
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("tk_mha_fwd_d128", &entrypoint);
}
