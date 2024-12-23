#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

constexpr int NUM_CONSUMER_WARPGROUPS = 2; // hardcoded, don't touch
constexpr int STATE_PER_SM = 2;            // how many maps get run on each SM. 
                                           // hardcoded, don't touch

constexpr int NUM_CONSUMER_WARPS = NUM_CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS;
constexpr int NUM_WARPS          = NUM_CONSUMER_WARPS + kittens::WARPGROUP_WARPS; 
constexpr int NUM_THREADS        = NUM_WARPS * kittens::WARP_THREADS;

using namespace kittens;
namespace cg = cooperative_groups;

struct fwd_globals {
    using q_tile        = st_bf<64, 64>;
    using k_tile        = st_bf<64, 64>;
    using v_tile        = st_bf<64, 64>;
    using o_tile        = st_bf<64, 64>;
    using kv_state_tile = st_fl<64, 64>;
    using q_map_tile    = st_bf<64, 64>;
    using k_map_tile    = st_bf<64, 64>;

    using q_gl        = gl<bf16, -1, -1, -1, -1, q_tile>;
    using k_gl        = gl<bf16, -1, -1, -1, -1, k_tile>;
    using v_gl        = gl<bf16, -1, -1, -1, -1, v_tile>;
    using o_gl        = gl<bf16, -1, -1, -1, -1, o_tile>;
    using kv_state_gl = gl<float, -1, -1, -1, -1, kv_state_tile>;
    using q_map_gl    = gl<bf16, -1, -1, -1, -1, q_map_tile>;
    using k_map_gl    = gl<bf16, -1, -1, -1, -1, k_map_tile>;

    q_gl        q;
    k_gl        k;
    v_gl        v;
    o_gl        o;
    kv_state_gl kv_state;
    q_map_gl    q_map;
    k_map_gl    k_map;

    const int N;
}; 

struct bwd_globals {
    using q_tile          = st_bf<64, 64>;
    using k_tile          = st_bf<64, 64>;
    using v_tile          = st_bf<64, 64>;
    using q_map_tile      = st_bf<64, 64>;
    using k_map_tile      = st_bf<64, 64>;
    using o_grad_tile     = st_bf<64, 64>;
    using kv_state_tile   = st_fl<64, 64>;
    using q_grad_tile     = st_fl<64, 64>;
    using k_grad_tile     = st_fl<64, 64>;
    using v_grad_tile     = st_fl<64, 64>;
    using q_map_grad_tile = st_fl<64, 64>;
    using k_map_grad_tile = st_fl<64, 64>;

    using q_gl          = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl          = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl          = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using q_map_gl      = gl<bf16,  -1, -1, -1, -1, q_map_tile>;
    using k_map_gl      = gl<bf16,  -1, -1, -1, -1, k_map_tile>;
    using o_grad_gl     = gl<bf16,  -1, -1, -1, -1, o_grad_tile>;
    using kv_state_gl   = gl<float, -1, -1, -1, -1, kv_state_tile>;
    using q_grad_gl     = gl<float, -1, -1, -1, -1, q_grad_tile>;
    using k_grad_gl     = gl<float, -1, -1, -1, -1, k_grad_tile>;
    using v_grad_gl     = gl<float, -1, -1, -1, -1, v_grad_tile>;
    using q_map_grad_gl = gl<float, -1, -1, -1, -1, q_map_grad_tile>;
    using k_map_grad_gl = gl<float, -1, -1, -1, -1, k_map_grad_tile>;

    q_gl          q;
    k_gl          k;
    v_gl          v;
    q_map_gl      q_map;
    k_map_gl      k_map;
    o_grad_gl     o_grad;
    kv_state_gl   kv_state;
    q_grad_gl     q_grad;
    k_grad_gl     k_grad;
    v_grad_gl     v_grad;
    q_map_grad_gl q_map_grad;
    k_map_grad_gl k_map_grad;

    const int N;
};

template<ducks::rt::all RT> __device__ static inline void featurize(RT &reg) {
    relu(reg, reg);
}

static __global__ __launch_bounds__(NUM_THREADS, 1)
void cylon_forwards(const __grid_constant__ fwd_globals g) 
{
    int laneid = kittens::laneid(); 
    int warpid = kittens::warpid(); 
    int warpgroupid = kittens::warpgroupid(); 

    int tic = 0, toc = 1; // these are used to track the two-stage pipeline.
    unsigned int batch_id = blockIdx.z; // which batch?
    unsigned int head_id  = blockIdx.y; // which head?
    unsigned int state_id = blockIdx.x; // which part of the KV state are we handling? (Important that this is on x.)
    int n_chunks = g.N / 64; // this kernel handles 64 rows per chunk.

    extern __shared__ int __shm[];
    tma_swizzle_allocator alloc((int*)&__shm[0]);

    using q_tile          = st_bf<64, 64>;
    using k_tile          = st_bf<64, 64>;
    using v_tile          = st_bf<64, 64>;
    using o_tile          = st_bf<64, 64>;
    using kv_scratch_tile = st_bf<64, 64>;
    using kv_state_tile   = st_fl<64, 64>;
    using q_map_tile      = st_bf<64, 64>;
    using k_map_tile      = st_bf<64, 64>;

    q_tile (&q_smem)[2]    = alloc.allocate<q_tile, 2>();    // 64x64, tic-toc'd (16384)
    k_tile (&k_smem)[2]    = alloc.allocate<k_tile, 2>();    // 64x64, tic-toc'd (16384)
    v_tile (&v_smem)[2][2] = alloc.allocate<v_tile, 2, 2>(); // 64x128, but easier to work with when split up. (32768)
    o_tile (&o_smem)[2]    = alloc.allocate<o_tile, 2>();    // 64x128, but easier to work with when split up. (16384)

    kv_scratch_tile (&kv_scratch)[2][2] = alloc.allocate<kv_scratch_tile, 2, 2>(); // This is scratch for doing wgmma's (32768)

    q_map_tile (&q_map)[2] = alloc.allocate<q_map_tile, 2>(); // featurized q (16384)
    k_map_tile (&k_map)[2] = alloc.allocate<k_map_tile, 2>(); // featurized k (16384)

    kv_state_tile (&kv_state_smem)[2][2] = reinterpret_cast<kv_state_tile(&)[2][2]>(q_smem); // we can reuse old memory for the writeout at the end

    // Initialize barriers
    __shared__ kittens::semaphore inputs_arrived[2], inputs_finished[2], outputs_ready[2];
    if (warpid == 0) {
        init_semaphore(inputs_arrived[0], 0, 2); // needs to wait on just one memory transaction, plus confirmation that o is available for writing.
        init_semaphore(inputs_arrived[1], 0, 2);
        if(laneid == 0) { arrive(inputs_arrived[0]); }
        init_semaphore(inputs_finished[0], NUM_CONSUMER_WARPS, 0);
        init_semaphore(inputs_finished[1], NUM_CONSUMER_WARPS, 0);
        init_semaphore(outputs_ready[0],   NUM_CONSUMER_WARPGROUPS, 0);
        init_semaphore(outputs_ready[1],   NUM_CONSUMER_WARPGROUPS, 0);
    }
    // Launch first load. No sync needed since thread 0 is doing these, too.
    if(warpid == 0) {
        tma::expect_bytes(inputs_arrived[0], sizeof(q_tile) * 8); // register a transaction for q, k, v, and q_map, k_map

        // load q, k, v
        coord<q_tile> q_tile_idx   = {batch_id, head_id, 0, 0};
        coord<k_tile> k_tile_idx   = {batch_id, head_id, 0, 0};
        coord<v_tile> v_tile_idx   = {batch_id, head_id, 0, 0};
        coord<v_tile> v_tile_idx_2 = {batch_id, head_id, 0, 1};
        tma::load_async(q_smem[tic], g.q, q_tile_idx,      inputs_arrived[0]);
        tma::load_async(k_smem[tic], g.k, k_tile_idx,      inputs_arrived[0]);
        tma::load_async(v_smem[tic][0], g.v, v_tile_idx,   inputs_arrived[0]);
        tma::load_async(v_smem[tic][1], g.v, v_tile_idx_2, inputs_arrived[0]);

        // load q map, k map blocks
        coord<q_map_tile> q_map_tile_idx   = {head_id, state_id, 0, 0};
        coord<q_map_tile> q_map_tile_idx_2 = {head_id, state_id, 1, 0};
        coord<k_map_tile> k_map_tile_idx   = {head_id, state_id, 0, 0};
        coord<k_map_tile> k_map_tile_idx_2 = {head_id, state_id, 1, 0};
        tma::load_async(q_map[0], g.q_map, q_map_tile_idx,   inputs_arrived[0]);
        tma::load_async(q_map[1], g.q_map, q_map_tile_idx_2, inputs_arrived[0]);
        tma::load_async(k_map[0], g.k_map, k_map_tile_idx,   inputs_arrived[0]);
        tma::load_async(k_map[1], g.k_map, k_map_tile_idx_2, inputs_arrived[0]);
    }

    __syncthreads(); // all warps must arrive here, confirming barrier initialization is visible to all threads.

    if(warpgroupid == NUM_CONSUMER_WARPGROUPS) { // last warpgroup is a producer
        warpgroup::decrease_registers<24>();
   
        if(warpid == NUM_CONSUMER_WARPS) { // just need a single warp to handle input loads
            for (int chunk_idx = 0; chunk_idx < n_chunks-1; chunk_idx++, tic=tic^1, toc=toc^1) {
                tma::expect_bytes(inputs_arrived[toc], sizeof(q_tile) * 4); // register that another block is coming in
                
                coord<q_tile> q_tile_idx   = {batch_id, head_id, chunk_idx + 1, 0};
                coord<k_tile> k_tile_idx   = {batch_id, head_id, chunk_idx + 1, 0};
                coord<v_tile> v_tile_idx   = {batch_id, head_id, chunk_idx + 1, 0};
                coord<v_tile> v_tile_idx_2 = {batch_id, head_id, chunk_idx + 1, 1};
                tma::load_async(q_smem[toc],    g.q, q_tile_idx,   inputs_arrived[toc]);
                tma::load_async(k_smem[toc],    g.k, k_tile_idx,   inputs_arrived[toc]);
                tma::load_async(v_smem[toc][0], g.v, v_tile_idx,   inputs_arrived[toc]);
                tma::load_async(v_smem[toc][1], g.v, v_tile_idx_2, inputs_arrived[toc]);

                wait(inputs_finished[tic], (chunk_idx/2)%2); // phase changes at half the rate of the tic/toc
            }
        }
        else if(warpid == NUM_CONSUMER_WARPS + 1) { // responsible for storing outputs
            for (int chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++, tic=tic^1, toc=toc^1) {
                wait(outputs_ready[tic], (chunk_idx/2)%2); // phase changes at half the rate of the tic/toc
                
                coord<o_tile> o_tile_idx;
                
                o_tile_idx = {batch_id, head_id, chunk_idx, 0};
                tma::store_add_async(g.o, o_smem[0], o_tile_idx);
                
                o_tile_idx = {batch_id, head_id, chunk_idx, 1};
                tma::store_add_async(g.o, o_smem[1], o_tile_idx);
                
                tma::store_commit_group();

                tma::store_async_read_wait();
                if(laneid == 0) arrive(inputs_arrived[toc]); // tell the consumers they can write to o again
            }
        }
    }
    else { // other warpgroups are consumers
        warpgroup::increase_registers<240>(); // 240 registers, no spills!

        rt_fl<16, 64> kv_state[2];
        zero(kv_state[0]);
        zero(kv_state[1]);

        for (int chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++, tic^=1, toc^=1) {
            
            wait(inputs_arrived[tic], (chunk_idx/2)%2); // wait for memory to arrive

            rt_fl<16, 64> o_reg, qf_reg, kf_reg;
            rt_bf<16, 64> qf_reg_bf, kf_reg_bf;
            
            // First thing we need to do is do [ReLU(q_map @ q)] @ kv_state
            warpgroup::mm_AB(qf_reg, q_smem[tic], q_map[0]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait(); // compiler wants this here
            warpgroup::store(kv_scratch[warpgroupid][0], kv_state[0]); // need to store k, v back to smem for wgmma

            featurize(qf_reg);
            copy(qf_reg_bf, qf_reg); // now q has been featurized the first way

            group<4>::sync(warpgroup::groupid()+4);
            // launch first kv_state matmul
            warpgroup::mm_AB(o_reg, qf_reg_bf, kv_scratch[warpgroupid][0]); // overwrite o_reg
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();
            
            // launch second featurization
            warpgroup::mm_AB(qf_reg, q_smem[tic], q_map[1]);
            warpgroup::mma_commit_group();
            warpgroup::store(kv_scratch[warpgroupid][1], kv_state[1]); // need to store k, v back to smem for wgmma
            warpgroup::mma_async_wait();

            featurize(qf_reg);
            copy(qf_reg_bf, qf_reg); // now q has been featurized the second way

            // shared memory writes need to have finished by this point.
            group<4>::sync(warpgroup::groupid()+4);
            // launch second kv_state matmul
            warpgroup::mma_AB(o_reg, qf_reg_bf, kv_scratch[warpgroupid][1]); // accumulate into o_reg
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait(); // at this point all compute on o is done.

            // NEXT: we need to update the state. We can immediately launch a k_map matmul from shared memory
            warpgroup::mm_AtBt(kf_reg, k_map[0], k_smem[tic]); // note we want to transpose k here on its way into registers
            warpgroup::mma_commit_group();
            
            // We've now done all the work on o that's required -- send it up to shared memory and let the producer run a cp.reduce.async.bulk.
            warpgroup::store(o_smem[warpgroupid], o_reg); // bf16

            warpgroup::mma_async_wait();

            featurize(kf_reg);
            copy(kf_reg_bf, kf_reg); // now k has been featurized with the first map

            warpgroup::mma_AB(kv_state[0], kf_reg_bf, v_smem[tic][warpgroupid]); // not pre-transposed so AtB
            warpgroup::mma_commit_group();

            // o memory arrived in smem!
            group<4>::sync(warpgroup::groupid()+4);
            if(warpgroup::laneid() == 0) arrive(outputs_ready[tic]); // we've now told the producer it can send o along once it's ready.

            warpgroup::mma_async_wait();

            // launch second featurization
            warpgroup::mm_AtBt(kf_reg, k_map[1], k_smem[tic]); // note we want to transpose k here on its way into registers
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            featurize(kf_reg);
            copy(kf_reg_bf, kf_reg); // now k has been featurized the second way

            warpgroup::mma_AB(kv_state[1], kf_reg_bf, v_smem[tic][warpgroupid]); // pre-transposed so AB
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            // Finished with inputs
            if(laneid == 0) arrive(inputs_finished[tic]); // we can now start loading the next q, k, v
        }
        kittens::group<8>::sync(10); // Everyone must be done with everything before we do this! Otherwise we will overwrite Q, K, V.
        warpgroup::store(kv_state_smem[0][warpgroupid], kv_state[0]);
        group<4>::sync(warpgroup::groupid()+4);
        if(warpgroup::warpid() == 0) {
            coord<kv_state_tile> kv_state_tile_idx = {batch_id, head_id, (state_id * STATE_PER_SM), warpgroupid};
            tma::store_async(g.kv_state, kv_state_smem[0][warpgroupid], kv_state_tile_idx);
            tma::store_commit_group();
        }
        warpgroup::store(kv_state_smem[1][warpgroupid], kv_state[1]);
        group<4>::sync(warpgroup::groupid()+4);
        if(warpgroup::warpid() == 0) {
            coord<kv_state_tile> kv_state_tile_idx = {batch_id, head_id, (state_id * STATE_PER_SM) + 1, warpgroupid};
            tma::store_async(g.kv_state, kv_state_smem[1][warpgroupid], kv_state_tile_idx);
            tma::store_commit_group();
        }
        tma::store_async_read_wait();
        group<4>::sync(warpgroup::groupid()+4);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<ducks::rt::all RT> __device__ static inline void featurize_backwards(RT &a_prefeature, RT &a_grad) {
    #pragma unroll
    for(int h = 0; h < RT::height; h++) {
        for(int w = 0; w < RT::width; w++) {
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                a_grad.tiles[h][w].data[i].x *= float(a_prefeature.tiles[h][w].data[i].x > 0);
                a_grad.tiles[h][w].data[i].y *= float(a_prefeature.tiles[h][w].data[i].y > 0);
            }
        }
    }
}

__device__ static inline void wg_async_store_add(auto &tma_dest, st_fl<64, 64> &shared, const rt_fl<16, 64> &reg, const auto index) {
    tma::store_async_wait();
    group<8>::sync(10);
    warpgroup::store(shared, reg);
    group<8>::sync(10); 
    if(warpgroup::warpid() == 0) {
        tma::store_add_async(tma_dest, shared, index); 
        tma::store_commit_group();
    }
}

// This is a producer+consumer copy kernel that demonstrates the use of TMA to implement a two-stage pipeline.
static __global__ __launch_bounds__(NUM_THREADS, 1)
void cylon_backwards(const __grid_constant__ bwd_globals g) { // in fp32

    int laneid = kittens::laneid(), warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();
    int tic = 0, toc = 1; // these are used to track the two-stage pipeline.
    unsigned int batch_id = blockIdx.z; // which batch?
    unsigned int head_id  = blockIdx.y; // which head?
    unsigned int state_id = blockIdx.x; // which part of the KV state are we handling? (Important that this is on x.)
    int n_chunks = g.N / 64; // this kernel handles 64 rows per chunk.

    extern __shared__ int __shm[];
    tma_swizzle_allocator alloc(&__shm[0]); // allocate shared memory
    st_bf<64, 64> (&q_smem)[2]     = alloc.allocate<st_bf<64, 64>, 2>(); // 64x64, tic-toc'd (16384)
    st_bf<64, 64> (&k_smem)[2]     = alloc.allocate<st_bf<64, 64>, 2>(); // 64x64, tic-toc'd (16384)
    st_bf<64, 64> (&v_smem)[2][2]  = alloc.allocate<st_bf<64, 64>, 2, 2>(); // 64x128, but easier to work with when split up. (32768)
    st_bf<64, 64> (&og_smem)[2][2] = alloc.allocate<st_bf<64, 64>, 2, 2>(); // 64x128, but easier to work with when split up. (32768)

    // cumulative smem at this point: 16384 + 16384 + 32768 + 32768 = 98304

    st_bf<64, 64> (&kv_state_smem)[2]         = alloc.allocate<st_bf<64, 64>, 2>();
    st_bf<64, 64> (&kv_state_grad_smem)[2]    = alloc.allocate<st_bf<64, 64>, 2>();
    st_bf<64, 64> (&qf_smem)                  = alloc.allocate<st_bf<64, 64>>();
    st_bf<64, 64> (&kf_smem)                  = alloc.allocate<st_bf<64, 64>>();
    st_bf<64, 64> (&qf_grad_smem)             = alloc.allocate<st_bf<64, 64>>();
    st_bf<64, 64> (&kf_grad_smem)             = alloc.allocate<st_bf<64, 64>>();


    // cumulative smem at this point: 98304 + 65536 = 163840

    st_bf<64, 64> (&q_map) = alloc.allocate<st_bf<64, 64>>(); // featurized q (8192)
    st_bf<64, 64> (&k_map) = alloc.allocate<st_bf<64, 64>>(); // featurized k (8192)

    // cumulative smem at this point: 163840 + 8192 + 8192 = 180224

    st_fl<64, 64> (&v_grad_smem)[2]        = alloc.allocate<st_fl<64, 64>, 2>(); // 64x64, tic-toc'd (32768)
    st_fl<64, 64> (&q_grad_smem)           = reinterpret_cast<st_fl<64, 64>&>(v_grad_smem[0]);
    st_fl<64, 64> (&k_grad_smem)           = reinterpret_cast<st_fl<64, 64>&>(v_grad_smem[1]);
    st_fl<64, 64> (&kv_state_load_smem)[2] = reinterpret_cast<st_fl<64, 64>(&)[2]>(v_grad_smem); // we can reuse wgmma scratch for the initial kv load

    // cumulative smem at this point: 180224 + 32768 = 212992

    // Initialize barriers
    __shared__ kittens::semaphore inputs_arrived[2], inputs_finished[2], setup_barrier;
    if (warpid == 0) {
        init_semaphore(inputs_arrived[0], 0, 1); // needs to wait on just one memory transaction, plus confirmation that o is available for writing.
        init_semaphore(inputs_arrived[1], 0, 1);
        init_semaphore(setup_barrier, 0, 1); // wait for kv state to load
        init_semaphore(inputs_finished[0], NUM_CONSUMER_WARPS, 0);
        init_semaphore(inputs_finished[1], NUM_CONSUMER_WARPS, 0);
    }
    // Launch first load. No sync needed since thread 0 is doing these, too.
    if(warpid == 0) {
        // tma::expect<st_fl<64, 64>, 2>(setup_barrier);
        tma::expect_bytes(setup_barrier, 2 * sizeof(st_fl<64, 64>));
        
        // load kv state
        coord<st_fl<64, 64>> idx_0 = {batch_id, head_id, state_id, 0};
        coord<st_fl<64, 64>> idx_1 = {batch_id, head_id, state_id, 1};
        tma::load_async(kv_state_load_smem[0], g.kv_state, idx_0, setup_barrier);
        tma::load_async(kv_state_load_smem[1], g.kv_state, idx_1, setup_barrier);
        
        // load q, k, v, q_map, k_map
        tma::expect_bytes(inputs_arrived[0], 8 * sizeof(st_bf<64, 64>));

        coord<st_bf<64, 64>> q_idx    = {batch_id, head_id, n_chunks-1, 0};
        coord<st_bf<64, 64>> k_idx    = {batch_id, head_id, n_chunks-1, 0};
        coord<st_bf<64, 64>> v_idx_0  = {batch_id, head_id, n_chunks-1, 0};
        coord<st_bf<64, 64>> v_idx_1  = {batch_id, head_id, n_chunks-1, 1};
        coord<st_bf<64, 64>> og_idx_0 = {batch_id, head_id, n_chunks-1, 0};
        coord<st_bf<64, 64>> og_idx_1 = {batch_id, head_id, n_chunks-1, 1};
        tma::load_async(q_smem[tic],     g.q,      q_idx,    inputs_arrived[0]); // launch the initial load
        tma::load_async(k_smem[tic],     g.k,      k_idx,    inputs_arrived[0]); // launch the initial load
        tma::load_async(v_smem[tic][0],  g.v,      v_idx_0,  inputs_arrived[0]); // launch the initial load
        tma::load_async(v_smem[tic][1],  g.v,      v_idx_1,  inputs_arrived[0]); // launch the initial load
        tma::load_async(og_smem[tic][0], g.o_grad, og_idx_0, inputs_arrived[0]); // launch the initial load
        tma::load_async(og_smem[tic][1], g.o_grad, og_idx_1, inputs_arrived[0]); // launch the initial load
        
        // load q map, k map blocks
        coord<st_bf<64, 64>> q_map_idx = {head_id, state_id/STATE_PER_SM, state_id%STATE_PER_SM, 0};
        coord<st_bf<64, 64>> k_map_idx = {head_id, state_id/STATE_PER_SM, state_id%STATE_PER_SM, 0};
        tma::load_async(q_map, g.q_map, q_map_idx, inputs_arrived[0]);
        tma::load_async(k_map, g.k_map, k_map_idx, inputs_arrived[0]);
    }

    __syncthreads(); // all warps must arrive here, confirming barrier initialization is visible to all threads.

    if(warpgroupid == NUM_CONSUMER_WARPGROUPS) { // last warpgroup is a producer
        warpgroup::decrease_registers<24>();
   
        if(warpid == NUM_CONSUMER_WARPS) { // just need a single warp to handle input loads
            for (int chunk_idx = 0; chunk_idx < n_chunks-1; chunk_idx++, tic=tic^1, toc=toc^1) {
                tma::expect_bytes(inputs_arrived[toc], 6 * sizeof(st_bf<64, 64>));

                coord<st_bf<64, 64>> q_idx    = {batch_id, head_id, n_chunks - chunk_idx - 2, 0};
                coord<st_bf<64, 64>> k_idx    = {batch_id, head_id, n_chunks - chunk_idx - 2, 0};
                coord<st_bf<64, 64>> v_idx_0  = {batch_id, head_id, n_chunks - chunk_idx - 2, 0};
                coord<st_bf<64, 64>> v_idx_1  = {batch_id, head_id, n_chunks - chunk_idx - 2, 1};
                coord<st_bf<64, 64>> og_idx_0 = {batch_id, head_id, n_chunks - chunk_idx - 2, 0};
                coord<st_bf<64, 64>> og_idx_1 = {batch_id, head_id, n_chunks - chunk_idx - 2, 1};
                tma::load_async(q_smem[toc],     g.q,      q_idx,    inputs_arrived[toc]); // load that block
                tma::load_async(k_smem[toc],     g.k,      k_idx,    inputs_arrived[toc]); // load that block
                tma::load_async(v_smem[toc][0],  g.v,      v_idx_0,  inputs_arrived[toc]); // load that block
                tma::load_async(v_smem[toc][1],  g.v,      v_idx_1,  inputs_arrived[toc]); // load that block
                tma::load_async(og_smem[toc][0], g.o_grad, og_idx_0, inputs_arrived[toc]); // load that block
                tma::load_async(og_smem[toc][1], g.o_grad, og_idx_1, inputs_arrived[toc]); // load that block
                
                wait(inputs_finished[tic], (chunk_idx/2)%2); // phase changes at half the rate of the tic/toc
            }
        }
    }
    else { // other warpgroups are consumers
        warpgroup::increase_registers<240>(); // 240 registers, no spills!
        using consumers = group<8>;

        rt_fl<16, 64> kv_state, kv_state_grad;
        zero(kv_state_grad); // zero initial grads
        wait(setup_barrier, 0);
        warpgroup::load(kv_state, kv_state_load_smem[warpgroupid]);
        mul(kv_state, kv_state, -1.f); // this is so that we can do mma's onto it with + later.
        warpgroup::store(kv_state_smem[warpgroupid], kv_state);
        consumers::sync(0);

        if(warpgroupid == 0) {
            rt_fl<16, 64> q_map_grad;
            zero(q_map_grad);    // zero initial grads

            for (int chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++, tic^=1, toc^=1) {
                wait(inputs_arrived[tic], (chunk_idx/2)%2); // wait for memory to arrive
                consumers::sync(0); // DEBUGGER

                rt_fl<16, 64> qf, qf_relu;
                warpgroup::mm_AB(qf, q_smem[tic], q_map);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();
                copy(qf_relu, qf);
                featurize(qf_relu);
                warpgroup::store(qf_smem, qf_relu);

                consumers::sync(0); // everyone done, no prior dependencies

                // next, we'll run kv and kv_grad matmuls and accumulate new ones in registers.
                // kv must be written out immediately, but kv_grad will be written to shared memory next iteration.

                // we need the previous iteration's kv grad for the stuff that comes next.
                warpgroup::store(kv_state_grad_smem[warpgroupid], kv_state_grad); // store to smem

                // update kv state
                warpgroup::mma_AtB(kv_state, kf_smem, v_smem[tic][warpgroupid]); // update kv state
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                consumers::sync(0); // DEBUGGER

                warpgroup::store(kv_state_smem[warpgroupid], kv_state);

                warpgroup::mma_AtB(kv_state_grad, qf_smem, og_smem[tic][warpgroupid]);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                consumers::sync(0); // everyone done, no prior dependencies

                // Next this warpgroup will be responsible for handling q_grad, and q_map_grad
            
                // q_head_chunk_grad = -torch.einsum('nv,kv->nk', o_grad_chunk, kv) * (q_head_prerelu>0)
                rt_fl<16, 64> qf_grad, q_grad;
                warpgroup::mm_ABt(qf_grad, og_smem[tic][0], kv_state_smem[0]);
                warpgroup::mma_ABt(qf_grad, og_smem[tic][1], kv_state_smem[1]);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();
                featurize_backwards(qf, qf_grad);
                mul(qf_grad, qf_grad, -1.f);
                warpgroup::store(qf_grad_smem, qf_grad); // need to store qf_grad_bf to smem
                warpgroup::sync(1); // memory must have arrived

                consumers::sync(0); // apparently necessary

                // q_grad += torch.einsum('nk,dk->nd', q_head_chunk_grad, qm)
                warpgroup::mm_ABt(q_grad, qf_grad_smem, q_map);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                coord<st_fl<64, 64>> idx = {batch_id, head_id, n_chunks - chunk_idx - 1, 0};
                wg_async_store_add(g.q_grad, q_grad_smem, q_grad, idx);

                // q_map_grad[s] += torch.einsum('nd,nk->dk', q_chunk, q_head_chunk_grad)
                warpgroup::mma_AtB(q_map_grad, q_smem[tic], qf_grad_smem);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                // everyone must do v_grad:
                rt_fl<16, 64> v_grad;
                warpgroup::mm_AB(v_grad, kf_smem, kv_state_grad_smem[warpgroupid]);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait(); // v_grad is now finished and in fp32

                coord<st_fl<64, 64>> v_idx = {batch_id, head_id, n_chunks - chunk_idx - 1, warpgroupid};
                wg_async_store_add(g.v_grad, v_grad_smem[warpgroupid], v_grad, v_idx); // v is wider

                // Finished with inputs
                if(laneid == 0) arrive(inputs_finished[tic]); // we can now start loading the next q, k, v
            }
            // write out q map grad
            consumers::sync(0); // wait for all warpgroups to finish
            coord<st_fl<64, 64>> idx = {head_id, state_id/STATE_PER_SM, state_id%STATE_PER_SM, 0};
            wg_async_store_add(g.q_map_grad, q_grad_smem, q_map_grad, idx);
            tma::store_async_wait();
            group<4>::sync(warpgroupid + 2); 
        }
        else {
            rt_fl<16, 64> k_map_grad;
            zero(k_map_grad);    // zero initial grads

            for (int chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++, tic^=1, toc^=1) {
                wait(inputs_arrived[tic], (chunk_idx/2)%2); // wait for memory to arrive
                consumers::sync(0); // DEBUGGER

                rt_fl<16, 64> kf, kf_relu;
                warpgroup::mm_AB(kf, k_smem[tic], k_map);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();
                copy(kf_relu, kf);
                featurize(kf_relu);
                warpgroup::store(kf_smem, kf_relu);

                consumers::sync(0); // everyone has now done phase one!

                // next, we'll run kv and kv_grad matmuls and accumulate new ones in registers.
                // kv must be written out immediately, but kv_grad will be written to shared memory next iteration.

                // we need the previous iteration's kv grad for the stuff that comes next.
                warpgroup::store(kv_state_grad_smem[warpgroupid], kv_state_grad); // store to smem
                // update kv state
                warpgroup::mma_AtB(kv_state, kf_smem, v_smem[tic][warpgroupid]); // update kv state
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                consumers::sync(0); // DEBUGGER

                warpgroup::store(kv_state_smem[warpgroupid], kv_state);

                warpgroup::mma_AtB(kv_state_grad, qf_smem, og_smem[tic][warpgroupid]);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                consumers::sync(0); // everyone done, no prior dependencies

                // Next this warpgroup will be responsible for handling k_grad, and k_map_grad

                rt_fl<16, 64> kf_grad, k_grad;
                warpgroup::mm_ABt(kf_grad, v_smem[tic][0], kv_state_grad_smem[0]);
                warpgroup::mma_ABt(kf_grad, v_smem[tic][1], kv_state_grad_smem[1]);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();
                featurize_backwards(kf, kf_grad);
                warpgroup::store(kf_grad_smem, kf_grad);
                warpgroup::sync(2);

                consumers::sync(0); // apparently necessary

                // k_grad += torch.einsum('nk,dk->nd', k_head_chunk_grad, k_map)
                warpgroup::mm_ABt(k_grad, kf_grad_smem, k_map);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait(); // k_grad is now finished and in fp32

                coord<st_fl<64, 64>> idx = {batch_id, head_id, n_chunks - chunk_idx - 1, 0};
                wg_async_store_add(g.k_grad, k_grad_smem, k_grad, idx);

                // k_map_grad[s] += torch.einsum('nd,nk->dk', k_chunk, k_head_chunk_grad)
                warpgroup::mma_AtB(k_map_grad, k_smem[tic], kf_grad_smem);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                // everyone must do v_grad:
                // v_grad += torch.einsum('nk,kv->nv', torch.relu(k_head_prerelu), kv_grad)
                rt_fl<16, 64> v_grad;
                warpgroup::mm_AB(v_grad, kf_smem, kv_state_grad_smem[warpgroupid]);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait(); // v_grad is now finished and in fp32

                coord<st_fl<64, 64>> v_idx = {batch_id, head_id, n_chunks - chunk_idx - 1, warpgroupid};
                wg_async_store_add(g.v_grad, v_grad_smem[warpgroupid], v_grad, v_idx); // v is wider

                // Finished with inputs
                if(laneid == 0) arrive(inputs_finished[tic]); // we can now start loading the next q, k, v
            }
            // write out k map grad
            consumers::sync(0); // wait for all warpgroups to finish
            coord<st_fl<64, 64>> idx = {head_id, state_id/STATE_PER_SM, state_id%STATE_PER_SM, 0};
            wg_async_store_add(g.k_map_grad, k_grad_smem, k_map_grad, idx);
            tma::store_async_wait();
            group<4>::sync(warpgroupid + 2); 
        }
    }
}

#ifdef TORCH_COMPILE

#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

std::vector<torch::Tensor>
cylon_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
              torch::Tensor q_map, torch::Tensor k_map) 
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(q_map);
    CHECK_INPUT(k_map);

    auto b = q.size(0);
    auto h = q.size(1);
    auto n = q.size(2);
    auto d_qk = q.size(3);
    auto d_vo = v.size(3);

    bf16* d_q     = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    bf16* d_k     = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    bf16* d_v     = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
    bf16* d_q_map = reinterpret_cast<bf16*>(q_map.data_ptr<c10::BFloat16>());
    bf16* d_k_map = reinterpret_cast<bf16*>(k_map.data_ptr<c10::BFloat16>());

    torch::Tensor o        = torch::zeros({static_cast<const uint>(b),
                                           static_cast<const uint>(h),
                                           static_cast<const uint>(n),
                                           static_cast<const uint>(d_vo)}, 
                                           q.options());

    torch::Tensor kv_state = torch::zeros({static_cast<const uint>(b),
                                           static_cast<const uint>(h),
                                           static_cast<const uint>(STATE_PER_SM * COLLABORATIVE_SMS * d_qk),
                                           static_cast<const uint>(d_vo)}, 
                                           torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));

    bf16* d_o        = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16* d_kv_state = reinterpret_cast<bf16*>(kv_state.data_ptr<c10::BFloat16>());

    using q_tile        = st_bf<64, 64>;
    using k_tile        = st_bf<64, 64>;
    using v_tile        = st_bf<64, 64>;
    using o_tile        = st_bf<64, 64>;
    using o_grad_tile   = st_bf<64, 64>;
    using kv_state_tile = st_fl<64, 64>;
    using q_map_tile    = st_bf<64, 64>;
    using k_map_tile    = st_bf<64, 64>;

    using q_grad_tile     = st_fl<64, 64>;
    using k_grad_tile     = st_fl<64, 64>;
    using v_grad_tile     = st_fl<64, 64>;
    using q_map_grad_tile = st_fl<64, 64>;
    using k_map_grad_tile = st_fl<64, 64>;

    using q_global        = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_global        = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_global        = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using o_global        = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using o_grad_global   = gl<bf16,  -1, -1, -1, -1, o_grad_tile>;
    using kv_state_global = gl<float, -1, -1, -1, -1, kv_state_tile>;
    using q_map_global    = gl<bf16, -1, -1, -1, -1,  q_map_tile>;
    using k_map_global    = gl<bf16, -1, -1, -1, -1,  k_map_tile>;

    using q_grad_global        = gl<float, -1, -1, -1, -1, q_grad_tile>;
    using k_grad_global        = gl<float, -1, -1, -1, -1, k_grad_tile>;
    using v_grad_global        = gl<float, -1, -1, -1, -1, v_grad_tile>;
    using q_map_grad_global    = gl<float, -1, -1, -1, -1, q_map_grad_tile>;
    using k_map_grad_global    = gl<float, -1, -1, -1, -1, k_map_grad_tile>;

    using globals_fwd = fwd_globals; 

    q_global           qg_arg{d_q,           static_cast<unsigned int>(b), static_cast<unsigned int>(h), static_cast<unsigned int>(n), static_cast<unsigned int>(d_qk)};
    k_global           kg_arg{d_k,           static_cast<unsigned int>(b), static_cast<unsigned int>(h), static_cast<unsigned int>(n), static_cast<unsigned int>(d_qk)};
    v_global           vg_arg{d_v,           static_cast<unsigned int>(b), static_cast<unsigned int>(h), static_cast<unsigned int>(n), static_cast<unsigned int>(d_vo)};
    o_global           og_arg{d_o,           static_cast<unsigned int>(b), static_cast<unsigned int>(h), static_cast<unsigned int>(n), static_cast<unsigned int>(d_vo)};
    kv_state_global kv_state_arg{d_kv_state, static_cast<unsigned int>(b), static_cast<unsigned int>(h), STATE_PER_SM * COLLABORATIVE_SMS * d_qk, static_cast<unsigned int>(d_vo)};
    q_map_global       q_map_arg{d_q_map,    static_cast<unsigned int>(h), COLLABORATIVE_SMS, static_cast<unsigned int>(d_qk * STATE_PER_SM), static_cast<unsigned int>(d_qk)};
    k_map_global       k_map_arg{d_k_map,    static_cast<unsigned int>(h), COLLABORATIVE_SMS, static_cast<unsigned int>(d_qk * STATE_PER_SM), static_cast<unsigned int>(d_qk)};

    globals_fwd g_fwd{qg_arg, kg_arg, vg_arg, og_arg, kv_state_arg, q_map_arg, k_map_arg, static_cast<unsigned int>(n)};

    auto stream   = at::cuda::getCurrentCUDAStream().stream();
    auto mem_size = 225000; 

    dim3 grid(COLLABORATIVE_SMS, H, B);

    cudaFuncSetAttribute(
        cylon_forwards, 
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    cylon_forwards<<<grid, NUM_THREADS, mem_size, stream>>>(g_fwd); 

    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaStreamSynchronize(stream);

    return {o, kv_state};
}

std::vector<torch::Tensor>
cylon_backward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
               torch::Tensor q_map, torch::Tensor k_map,
               torch::Tensor o_grad, torch::Tensor kv_state) 
{
    CHECK_INPUT(q);
    CHECK_INPUT(k); 
    CHECK_INPUT(v);
    CHECK_INPUT(q_map);
    CHECK_INPUT(k_map);
    CHECK_INPUT(o_grad);
    CHECK_INPUT(kv_state);

    auto b = q.size(0);
    auto h = q.size(1);
    auto n = q.size(2);
    auto d_qk = q.size(3);
    auto d_vo = v.size(3);

    torch::Tensor q_grad     = torch::zeros_like(q, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor k_grad     = torch::zeros_like(k, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor v_grad     = torch::zeros_like(v, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor q_map_grad = torch::zeros_like(q_map, torch::TensorOptions().dtype(torch::kFloat32));
    torch::Tensor k_map_grad = torch::zeros_like(k_map, torch::TensorOptions().dtype(torch::kFloat32));

    bf16* d_q         = reinterpret_cast<bf16*>(q.data_ptr<c10::BFloat16>());
    bf16* d_k         = reinterpret_cast<bf16*>(k.data_ptr<c10::BFloat16>());
    bf16* d_v         = reinterpret_cast<bf16*>(v.data_ptr<c10::BFloat16>());
    bf16* d_q_map     = reinterpret_cast<bf16*>(q_map.data_ptr<c10::BFloat16>());
    bf16* d_k_map     = reinterpret_cast<bf16*>(k_map.data_ptr<c10::BFloat16>());
    bf16* d_o_grad    = reinterpret_cast<bf16*>(o_grad.data_ptr<c10::BFloat16>());
    float* d_kv_state = reinterpret_cast<float*>(kv_state.data_ptr<float>());

    float* d_q_grad     = reinterpret_cast<float*>(q_grad.data_ptr<float>());
    float* d_k_grad     = reinterpret_cast<float*>(k_grad.data_ptr<float>());
    float* d_v_grad     = reinterpret_cast<float*>(v_grad.data_ptr<float>());
    float* d_q_map_grad = reinterpret_cast<float*>(q_map_grad.data_ptr<float>());
    float* d_k_map_grad = reinterpret_cast<float*>(k_map_grad.data_ptr<float>());

    using q_tile        = st_bf<64, 64>;
    using k_tile        = st_bf<64, 64>;
    using v_tile        = st_bf<64, 64>;
    using o_tile        = st_bf<64, 64>;
    using o_grad_tile   = st_bf<64, 64>;
    using kv_state_tile = st_fl<64, 64>;
    using q_map_tile    = st_bf<64, 64>;
    using k_map_tile    = st_bf<64, 64>;

    using q_grad_tile     = st_fl<64, 64>;
    using k_grad_tile     = st_fl<64, 64>;
    using v_grad_tile     = st_fl<64, 64>;
    using q_map_grad_tile = st_fl<64, 64>;
    using k_map_grad_tile = st_fl<64, 64>;

    using q_global        = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_global        = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_global        = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using o_global        = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using o_grad_global   = gl<bf16,  -1, -1, -1, -1, o_grad_tile>;
    using kv_state_global = gl<float, -1, -1, -1, -1, kv_state_tile>;
    using q_map_global    = gl<bf16, -1, -1, -1, -1,  q_map_tile>;
    using k_map_global    = gl<bf16, -1, -1, -1, -1,  k_map_tile>;

    using q_grad_global        = gl<float, -1, -1, -1, -1, q_grad_tile>;
    using k_grad_global        = gl<float, -1, -1, -1, -1, k_grad_tile>;
    using v_grad_global        = gl<float, -1, -1, -1, -1, v_grad_tile>;
    using q_map_grad_global    = gl<float, -1, -1, -1, -1, q_map_grad_tile>;
    using k_map_grad_global    = gl<float, -1, -1, -1, -1, k_map_grad_tile>;

    using globals_bwd = bwd_globals;

    q_global              qg_arg{d_q,        static_cast<unsigned int>(b), static_cast<unsigned int>(h), static_cast<unsigned int>(n), static_cast<unsigned int>(d_qk)};
    k_global              kg_arg{d_k,        static_cast<unsigned int>(b), static_cast<unsigned int>(h), static_cast<unsigned int>(n), static_cast<unsigned int>(d_qk)};
    v_global              vg_arg{d_v,        static_cast<unsigned int>(b), static_cast<unsigned int>(h), static_cast<unsigned int>(n), static_cast<unsigned int>(d_vo)};
    o_global              og_arg{d_o,        static_cast<unsigned int>(b), static_cast<unsigned int>(h), static_cast<unsigned int>(n), static_cast<unsigned int>(d_vo)};
    kv_state_global kv_state_arg{d_kv_state, static_cast<unsigned int>(b), static_cast<unsigned int>(h), STATE_PER_SM * COLLABORATIVE_SMS * d_qk, static_cast<unsigned int>(d_vo)};
    
    q_map_global       q_map_arg{d_q_map,    static_cast<unsigned int>(h), COLLABORATIVE_SMS, static_cast<unsigned int>(d_qk * STATE_PER_SM), static_cast<unsigned int>(d_qk)};
    k_map_global       k_map_arg{d_k_map,    static_cast<unsigned int>(h), COLLABORATIVE_SMS, static_cast<unsigned int>(d_qk * STATE_PER_SM), static_cast<unsigned int>(d_qk)};
    o_grad_global    og_grad_arg{d_o_grad,   static_cast<unsigned int>(b), static_cast<unsigned int>(h), static_cast<unsigned int>(n), static_cast<unsigned int>(d_vo)};

    q_grad_global           qg_grad_arg{d_q_grad,     static_cast<unsigned int>(b), static_cast<unsigned int>(h), static_cast<unsigned int>(n), static_cast<unsigned int>(d_qk)};
    k_grad_global           kg_grad_arg{d_k_grad,     static_cast<unsigned int>(b), static_cast<unsigned int>(h), static_cast<unsigned int>(n), static_cast<unsigned int>(d_qk)};
    v_grad_global           vg_grad_arg{d_v_grad,     static_cast<unsigned int>(b), static_cast<unsigned int>(h), static_cast<unsigned int>(n), static_cast<unsigned int>(d_vo)};
    q_map_grad_global    q_map_grad_arg{d_q_map_grad, static_cast<unsigned int>(h), COLLABORATIVE_SMS, static_cast<unsigned int>(d_qk * STATE_PER_SM), static_cast<unsigned int>(d_qk)};
    k_map_grad_global    k_map_grad_arg{d_k_map_grad, static_cast<unsigned int>(h), COLLABORATIVE_SMS, static_cast<unsigned int>(d_qk * STATE_PER_SM), static_cast<unsigned int>(d_qk)};

    globals_bwd g_bwd{qg_arg, kg_arg, vg_arg, q_map_arg, k_map_arg, og_grad_arg, kv_state_arg, qg_grad_arg, kg_grad_arg, vg_grad_arg, q_map_grad_arg, k_map_grad_arg, static_cast<unsigned int>(n)};

    auto stream   = at::cuda::getCurrentCUDAStream().stream();
    auto mem_size = 225000; 

    dim3 grid(COLLABORATIVE_SMS*2, H, B);

    cudaFuncSetAttribute(
        cylon_backwards, 
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    cylon_backwards<<<grid, NUM_THREADS, mem_size, stream>>>(g_bwd);

    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaStreamSynchronize(stream);

    return {q_grad, k_grad, v_grad, q_map_grad, k_map_grad};
}

#else
#include "harness.impl"
#endif