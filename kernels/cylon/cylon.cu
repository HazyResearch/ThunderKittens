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

    using q_tile = st_bf<64, 64>;
    using k_tile = st_bf<64, 64>;
    using v_tile = st_bf<64, 64>;
    using o_tile = st_bf<64, 64>;
    using kv_scratch_tile = st_bf<64, 64>;
    using kv_state_tile   = st_fl<64, 64>;
    using q_map_tile = st_bf<64, 64>;
    using k_map_tile = st_bf<64, 64>;

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
        if(laneid == 0) { // o is available for writing on the initial load, so mark that appropriately
            arrive(inputs_arrived[0]);
        }
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
        tma::load_async(q_smem[tic], g.q, q_tile_idx,      inputs_arrived[0]);
        coord<k_tile> k_tile_idx   = {batch_id, head_id, 0, 0};
        tma::load_async(k_smem[tic], g.k, k_tile_idx,      inputs_arrived[0]);
        coord<v_tile> v_tile_idx   = {batch_id, head_id, 0, 0};
        tma::load_async(v_smem[tic][0], g.v, v_tile_idx,   inputs_arrived[0]);
        coord<v_tile> v_tile_idx_2 = {batch_id, head_id, 0, 1};
        tma::load_async(v_smem[tic][1], g.v, v_tile_idx_2, inputs_arrived[0]);

        // load q map, k map blocks
        coord<q_map_tile> q_map_tile_idx   = {head_id, state_id, 0, 0};
        tma::load_async(q_map[0], g.q_map, q_map_tile_idx,   inputs_arrived[0]);
        coord<q_map_tile> q_map_tile_idx_2 = {head_id, state_id, 1, 0};
        tma::load_async(q_map[1], g.q_map, q_map_tile_idx_2, inputs_arrived[0]);
        coord<k_map_tile> k_map_tile_idx   = {head_id, state_id, 0, 0};
        tma::load_async(k_map[0], g.k_map, k_map_tile_idx,   inputs_arrived[0]);
        coord<k_map_tile> k_map_tile_idx_2 = {head_id, state_id, 1, 0};
        tma::load_async(k_map[1], g.k_map, k_map_tile_idx_2, inputs_arrived[0]);
    }

    __syncthreads(); // all warps must arrive here, confirming barrier initialization is visible to all threads.

    if(warpgroupid == NUM_CONSUMER_WARPGROUPS) { // last warpgroup is a producer
        warpgroup::decrease_registers<24>();
   
        if(warpid == NUM_CONSUMER_WARPS) { // just need a single warp to handle input loads
            for (int chunk_idx = 0; chunk_idx < n_chunks-1; chunk_idx++, tic=tic^1, toc=toc^1) {
                tma::expect_bytes(inputs_arrived[toc], sizeof(q_tile) * 4); // register that another block is coming in
                
                coord<q_tile> q_tile_idx   = {batch_id, head_id, chunk_idx + 1, 0};
                tma::load_async(q_smem[toc], g.q, q_tile_idx, inputs_arrived[toc]);
                coord<k_tile> k_tile_idx   = {batch_id, head_id, chunk_idx + 1, 0};
                tma::load_async(k_smem[toc], g.k, k_tile_idx, inputs_arrived[toc]);
                coord<v_tile> v_tile_idx   = {batch_id, head_id, chunk_idx + 1, 0};
                tma::load_async(v_smem[toc][0], g.v, v_tile_idx, inputs_arrived[toc]);
                coord<v_tile> v_tile_idx_2 = {batch_id, head_id, chunk_idx + 1, 1};
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

#include "harness.impl"