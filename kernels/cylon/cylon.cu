#include "kittens.cuh"
#include "common/debug.cuh"
#include "cooperative_groups.h"
namespace cg = cooperative_groups;
using namespace kittens;

constexpr int NUM_CONSUMER_WARPGROUPS = 2; // hardcoded, don't touch
constexpr int NUM_CONSUMER_WARPS = NUM_CONSUMER_WARPGROUPS * WARPGROUP_WARPS; // hardcoded, don't touch
constexpr int NUM_WARPS = NUM_CONSUMER_WARPS + WARPGROUP_WARPS; // hardcoded, don't touch
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS; // hardcoded, don't touch
constexpr int STATE_PER_SM = 2; // how many maps get run on each SM. hardcoded, don't touch

template<ducks::rt::all RT> __device__ inline void featurize(RT &reg) {
    // This is a placeholder for whatever featurization we want to do.
    // In this case, I'll just use a ReLU for simplicity.
    relu(reg, reg);
}

// This is a producer+consumer copy kernel that demonstrates the use of TMA to implement a two-stage pipeline.
// TODO: We do actually have enough shared memory to make it a three-stage pipeline. 
__global__ __launch_bounds__(NUM_THREADS, 1)
void cylon_forwards(int N,
                    const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v,
                    CUtensorMap* tma_o, CUtensorMap* tma_kv_state,
                    const CUtensorMap* tma_q_map, const CUtensorMap* tma_k_map)  {

    int laneid = kittens::laneid(), warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();
    int tic = 0, toc = 1; // these are used to track the two-stage pipeline.
    unsigned int batch_id = blockIdx.x; // which batch?
    unsigned int head_id  = blockIdx.y; // which head?
    unsigned int state_id = blockIdx.z; // which part of the KV state are we handling?
    int n_chunks = N / 64; // this kernel handles 64 rows per chunk.

    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]); // allocate shared memory
    st_bf_4x4 (&q_smem)[2]    = alloc.allocate<st_bf_4x4, 2>(); // 64x64, tic-toc'd (16384)
    st_bf_4x4 (&k_smem)[2]    = alloc.allocate<st_bf_4x4, 2>(); // 64x64, tic-toc'd (16384)
    st_bf_4x4 (&v_smem)[2][2] = alloc.allocate<st_bf_4x4, 2, 2>(); // 64x128, but easier to work with when split up. (32768)
    st_bf_4x4 (&o_smem)[2]    = alloc.allocate<st_bf_4x4, 2>(); // 64x128, but easier to work with when split up. (16384)

    st_bf_4x4 (&kv_scratch)[2][2] = alloc.allocate<st_bf_4x4, 2, 2>(); // This is scratch for doing wgmma's (32768)

    st_bf_4x4 (&q_map)[2] = alloc.allocate<st_bf_4x4, 2>(); // featurized q (16384)
    st_bf_4x4 (&k_map)[2] = alloc.allocate<st_bf_4x4, 2>(); // featurized k (16384)

    st_fl_4x4 (&kv_state_smem)[2][2] = reinterpret_cast<st_fl_4x4(&)[2][2]>(q_smem); // we can reuse old memory for the writeout at the end

    // Initialize barriers
    __shared__ kittens::barrier inputs_arrived[2], inputs_finished[2], outputs_ready[2];
    if (warpid == 0) {
        init_barrier(inputs_arrived[0], 0, 2); // needs to wait on just one memory transaction, plus confirmation that o is available for writing.
        init_barrier(inputs_arrived[1], 0, 2);
        if(laneid == 0) { // o is available for writing on the initial load, so mark that appropriately
            arrive(inputs_arrived[0]);
        }
        init_barrier(inputs_finished[0], NUM_CONSUMER_WARPS, 0);
        init_barrier(inputs_finished[1], NUM_CONSUMER_WARPS, 0);
        init_barrier(outputs_ready[0],   NUM_CONSUMER_WARPGROUPS, 0);
        init_barrier(outputs_ready[1],   NUM_CONSUMER_WARPGROUPS, 0);
    }
    // Launch first load. No sync needed since thread 0 is doing these, too.
    if(warpid == 0) {
        tma::expect<st_bf_4x4, 8>(inputs_arrived[0]); // register a transaction for q, k, v, and q_map, k_map
        int load_idx = ((batch_id * gridDim.y) + head_id) * n_chunks;
        tma::load_async(q_smem[tic],    tma_q, inputs_arrived[0], load_idx); // launch the initial load
        tma::load_async(k_smem[tic],    tma_k, inputs_arrived[0], load_idx); // launch the initial load
        tma::load_async(v_smem[tic][0], tma_v, inputs_arrived[0], load_idx, 0); // launch the initial load
        tma::load_async(v_smem[tic][1], tma_v, inputs_arrived[0], load_idx, 1); // launch the initial load
        // load q map, k map blocks
        int base_map_idx = (head_id * gridDim.z + state_id) * STATE_PER_SM; // gridDim.z is how many SMs are collaborating on each head
        tma::load_async(q_map[0], tma_q_map, inputs_arrived[0], base_map_idx+0);
        tma::load_async(q_map[1], tma_q_map, inputs_arrived[0], base_map_idx+1);
        tma::load_async(k_map[0], tma_k_map, inputs_arrived[0], base_map_idx+0);
        tma::load_async(k_map[1], tma_k_map, inputs_arrived[0], base_map_idx+1);
    }

    __syncthreads(); // all warps must arrive here, confirming barrier initialization is visible to all threads.

    if(warpgroupid == NUM_CONSUMER_WARPGROUPS) { // last warpgroup is a producer
        warpgroup::decrease_registers<24>();
   
        if(warpid == NUM_CONSUMER_WARPS) { // just need a single warp to handle input loads
            for (int chunk_idx = 0; chunk_idx < n_chunks-1; chunk_idx++, tic=tic^1, toc=toc^1) {
                tma::expect<st_bf_4x4, 4>(inputs_arrived[toc]); // register that another block is coming in
                int next_load_idx = ((batch_id * gridDim.y) + head_id) * n_chunks + chunk_idx + 1;
                tma::load_async(q_smem[toc],    tma_q, inputs_arrived[toc], next_load_idx); // load that block
                tma::load_async(k_smem[toc],    tma_k, inputs_arrived[toc], next_load_idx); // load that block
                tma::load_async(v_smem[toc][0], tma_v, inputs_arrived[toc], next_load_idx, 0); // load that block
                tma::load_async(v_smem[toc][1], tma_v, inputs_arrived[toc], next_load_idx, 1); // load that block
                wait(inputs_finished[tic], (chunk_idx/2)%2); // phase changes at half the rate of the tic/toc
            }
        }
        else if(warpid == NUM_CONSUMER_WARPS + 1) { // responsible for storing outputs
            for (int chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++, tic=tic^1, toc=toc^1) {
                wait(outputs_ready[tic], (chunk_idx/2)%2); // phase changes at half the rate of the tic/toc
                int store_idx = ((batch_id * gridDim.y) + head_id) * n_chunks + chunk_idx;
                tma::store_add_async(tma_o, o_smem[0], store_idx, 0); // store
                tma::store_add_async(tma_o, o_smem[1], store_idx, 1);
                tma::store_commit_group();
                tma::store_async_read_wait();
                if(laneid == 0) arrive(inputs_arrived[toc]); // tell the consumers they can write to o again
            }
        }
    }
    else { // other warpgroups are consumers
        warpgroup::increase_registers<240>(); // 240 registers, no spills!

        rt_fl_1x4<> kv_state[2];
        zero(kv_state[0]);
        zero(kv_state[1]);

        for (int chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++, tic^=1, toc^=1) {
            
            wait(inputs_arrived[tic], (chunk_idx/2)%2); // wait for memory to arrive

            rt_fl_1x4<> o_reg, qf_reg, kf_reg;
            rt_bf_1x4<> qf_reg_bf, kf_reg_bf;
            
            // First thing we need to do is do [ReLU(q_map @ q)] @ kv_state

            warpgroup::mma_fence(qf_reg);
            warpgroup::mm_AB(qf_reg, q_smem[tic], q_map[0]);
            warpgroup::mma_commit_group();
            // INJECTION: warpgroup::mma_async_wait();
            warpgroup::store(kv_scratch[warpgroupid][0], kv_state[0]); // need to store k, v back to smem for wgmma
            warpgroup::mma_async_wait();

            featurize(qf_reg);
            copy(qf_reg_bf, qf_reg); // now q has been featurized the first way

            warpgroup::sync(); // shared memory writes need to have finished by this point.
            warpgroup::mma_fence(o_reg); // launch first kv_state matmul
            warpgroup::mm_AB(o_reg, qf_reg_bf, kv_scratch[warpgroupid][0]); // overwrite o_reg
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            warpgroup::mma_fence(qf_reg); // launch second featurization
            warpgroup::mm_AB(qf_reg, q_smem[tic], q_map[1]);
            warpgroup::mma_commit_group();
            // INJECTION: warpgroup::mma_async_wait();
            warpgroup::store(kv_scratch[warpgroupid][1], kv_state[1]); // need to store k, v back to smem for wgmma
            warpgroup::mma_async_wait();

            featurize(qf_reg);
            copy(qf_reg_bf, qf_reg); // now q has been featurized the second way

            warpgroup::sync(); // shared memory writes need to have finished by this point.
            warpgroup::mma_fence(o_reg); // launch second kv_state matmul
            warpgroup::mma_AB(o_reg, qf_reg_bf, kv_scratch[warpgroupid][1]); // accumulate into o_reg
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait(); // at this point all compute on o is done.

            // NEXT: we need to update the state. We can immediately launch a k_map matmul from shared memory

            warpgroup::mma_fence(kf_reg);
            warpgroup::mm_AtBt(kf_reg, k_map[0], k_smem[tic]); // note we want to transpose k here on its way into registers
            warpgroup::mma_commit_group();
            // INJECTION: warpgroup::mma_async_wait();
            
            // We've now done all the work on o that's required -- send it up to shared memory and let the producer run a cp.reduce.async.bulk.
            warpgroup::store(o_smem[warpgroupid], o_reg); // bf16

            warpgroup::mma_async_wait();

            featurize(kf_reg);
            copy(kf_reg_bf, kf_reg); // now k has been featurized with the first map

            warpgroup::mma_fence(kv_state[0]);
            warpgroup::mma_AB(kv_state[0], kf_reg_bf, v_smem[tic][warpgroupid]); // not pre-transposed so AtB
            warpgroup::mma_commit_group();

            warpgroup::sync(); // o memory arrived in smem!
            if(warpgroup::laneid() == 0) arrive(outputs_ready[tic]); // we've now told the producer it can send o along once it's ready.

            warpgroup::mma_async_wait();

            warpgroup::mma_fence(kf_reg); // launch second featurization
            warpgroup::mm_AtBt(kf_reg, k_map[1], k_smem[tic]); // note we want to transpose k here on its way into registers
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            featurize(kf_reg);
            copy(kf_reg_bf, kf_reg); // now k has been featurized the second way

            warpgroup::mma_fence(kv_state[1]);
            warpgroup::mma_AB(kv_state[1], kf_reg_bf, v_smem[tic][warpgroupid]); // pre-transposed so AB
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            // Finished with inputs
            if(laneid == 0) arrive(inputs_finished[tic]); // we can now start loading the next q, k, v
        }
        kittens::group<8>::sync(); // Everyone must be done with everything before we do this! Otherwise we will overwrite Q, K, V.
        warpgroup::store(kv_state_smem[0][warpgroupid], kv_state[0]);
        warpgroup::sync();
        if(warpgroup::warpid() == 0) {
            int base_kv_idx = ((((batch_id * gridDim.y) + head_id) * gridDim.z) + state_id) * STATE_PER_SM;
            tma::store_async(tma_kv_state, kv_state_smem[0][warpgroupid], base_kv_idx+0, warpgroupid);
            tma::store_commit_group();
        }
        warpgroup::store(kv_state_smem[1][warpgroupid], kv_state[1]);
        warpgroup::sync();
        if(warpgroup::warpid() == 0) {
            int base_kv_idx = ((((batch_id * gridDim.y) + head_id) * gridDim.z) + state_id) * STATE_PER_SM;
            tma::store_async(tma_kv_state, kv_state_smem[1][warpgroupid], base_kv_idx+1, warpgroupid);
            tma::store_commit_group();
        }
        tma::store_async_read_wait();
        warpgroup::sync();
    }
}


// This is a producer+consumer copy kernel that demonstrates the use of TMA to implement a two-stage pipeline.
// __global__ __launch_bounds__(NUM_THREADS, 1)
// void cylon_backwards(int N,
//                     const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v
//                     const CUtensorMap* tma_q_map, const CUtensorMap* tma_k_map,
//                     const CUtensorMap* tma_o_grad,
//                     CUtensorMap* tma_q_grad, CUtensorMap* tma_k_grad, CUtensorMap* tma_v_grad,
//                     CUtensorMap* tma_q_map_grad, CUtensorMap* tma_k_map_grad) {

//     int laneid = kittens::laneid(), warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();
//     int tic = 0, toc = 1; // these are used to track the two-stage pipeline.
//     unsigned int batch_id = blockIdx.x / COLLABORATIVE_SMS; // which batch/head are we handling?
//     unsigned int state_id = blockIdx.x % COLLABORATIVE_SMS; // which part of the KV state are we handling?
//     int n_chunks = N / 64; // this kernel handles 64 rows per chunk.

//     extern __shared__ int __shm[];
//     shared_allocator alloc(&__shm[0]); // allocate shared memory
//     st_bf_4x4 (&q_smem)[2]    = alloc.allocate<st_bf_4x4, 2>(); // 64x64, tic-toc'd (16384)
//     st_bf_4x4 (&k_smem)[2]    = alloc.allocate<st_bf_4x4, 2>(); // 64x64, tic-toc'd (16384)
//     st_bf_4x4 (&v_smem)[2][2] = alloc.allocate<st_bf_4x4, 2, 2>(); // 64x128, but easier to work with when split up. (32768)
//     st_bf_4x4 (&o_smem)[2][2] = alloc.allocate<st_bf_4x4, 2, 2>(); // 64x128, but easier to work with when split up. (32768)

//     st_bf_4x4 (&kv_scratch)[2][2] = alloc.allocate<st_bf_4x4, 2, 2>(); // This is scratch for doing wgmma's (32768)

//     st_bf_4x4 (&q_map)[2] = alloc.allocate<st_bf_4x4, 2>(); // featurized q (16384)
//     st_bf_4x4 (&k_map)[2] = alloc.allocate<st_bf_4x4, 2>(); // featurized k (16384)

//     // Initialize barriers
//     __shared__ kittens::barrier inputs_arrived[2], inputs_finished[2], outputs_ready[2];
//     if (warpid == 0) {
//         init_barrier(inputs_arrived[0], 0, 2); // needs to wait on just one memory transaction, plus confirmation that o is available for writing.
//         init_barrier(inputs_arrived[1], 0, 2);
//         if(laneid == 0) { // o is available for writing on the initial load, so mark that appropriately
//             arrive(inputs_arrived[0]);
//         }
//         init_barrier(inputs_finished[0], NUM_CONSUMER_WARPS, 0);
//         init_barrier(inputs_finished[1], NUM_CONSUMER_WARPS, 0);
//         init_barrier(outputs_ready[0],   NUM_CONSUMER_WARPGROUPS, 0);
//         init_barrier(outputs_ready[1],   NUM_CONSUMER_WARPGROUPS, 0);
//     }
//     // Launch first load. No sync needed since thread 0 is doing these, too.
//     if(warpid == 0) {
//         tma::expect<st_bf_4x4, 8>(inputs_arrived[0]); // register a transaction for q, k, v, and q_map, k_map
//         int load_idx = n_chunks * batch_id;
//         tma::load_async(q_smem[tic],    tma_q, inputs_arrived[0], load_idx); // launch the initial load
//         tma::load_async(k_smem[tic],    tma_k, inputs_arrived[0], load_idx); // launch the initial load
//         tma::load_async(v_smem[tic][0], tma_v, inputs_arrived[0], load_idx, 0); // launch the initial load
//         tma::load_async(v_smem[tic][1], tma_v, inputs_arrived[0], load_idx, 1); // launch the initial load
//         // load q map, k map blocks
//         tma::load_async(q_map[0], tma_q_map, inputs_arrived[0], state_id*2+0);
//         tma::load_async(q_map[1], tma_q_map, inputs_arrived[0], state_id*2+1);
//         tma::load_async(k_map[0], tma_k_map, inputs_arrived[0], state_id*2+0);
//         tma::load_async(k_map[1], tma_k_map, inputs_arrived[0], state_id*2+1);
//     }

//     __syncthreads(); // all warps must arrive here, confirming barrier initialization is visible to all threads.

//     if(warpgroupid == NUM_CONSUMER_WARPGROUPS) { // last warpgroup is a producer
//         warpgroup::decrease_registers<24>();
   
//         if(warpid == NUM_CONSUMER_WARPS) { // just need a single warp to handle input loads
//             for (int chunk_idx = 0; chunk_idx < n_chunks-1; chunk_idx++, tic=tic^1, toc=toc^1) {
//                 tma::expect<st_bf_4x4, 4>(inputs_arrived[toc]); // register that another block is coming in
//                 int next_load_idx = n_chunks * batch_id + chunk_idx + 1;
//                 tma::load_async(q_smem[toc],    tma_q, inputs_arrived[toc], next_load_idx); // load that block
//                 tma::load_async(k_smem[toc],    tma_k, inputs_arrived[toc], next_load_idx); // load that block
//                 tma::load_async(v_smem[toc][0], tma_v, inputs_arrived[toc], next_load_idx, 0); // load that block
//                 tma::load_async(v_smem[toc][1], tma_v, inputs_arrived[toc], next_load_idx, 1); // load that block
//                 wait(inputs_finished[tic], (chunk_idx/2)%2); // phase changes at half the rate of the tic/toc
//             }
//         }
//         else if(warpid == NUM_CONSUMER_WARPS + 1) { // responsible for storing outputs
//             for (int chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++, tic=tic^1, toc=toc^1) {
//                 wait(outputs_ready[tic], (chunk_idx/2)%2); // phase changes at half the rate of the tic/toc
//                 int store_idx = n_chunks * batch_id + chunk_idx;
//                 tma::store_async_read_wait();
//                 tma::store_add_async(tma_o, o_smem[tic][0], store_idx, 0); // store
//                 tma::store_add_async(tma_o, o_smem[tic][1], store_idx, 1);
//                 tma::store_commit_group();
//                 if(laneid == 0) arrive(inputs_arrived[toc]); // tell the consumers they can write to o again
//             }
//         }
//     }
//     else { // other warpgroups are consumers
//         warpgroup::increase_registers<240>(); // 240 registers, no spills!

//         rt_fl_1x4<> kv_state[2];
//         zero(kv_state[0]);
//         zero(kv_state[1]);

//         for (int chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++, tic^=1, toc^=1) {
            
//             wait(inputs_arrived[tic], (chunk_idx/2)%2); // wait for memory to arrive

//             rt_fl_1x4<> o_reg, qf_reg, kf_reg;
//             rt_bf_1x4<> qf_reg_bf, kf_reg_bf;
            
//             // First thing we need to do is do [ReLU(q_map @ q)] @ kv_state

//             warpgroup::mma_fence(qf_reg);
//             warpgroup::mm_AB(qf_reg, q_smem[tic], q_map[0]);
//             warpgroup::mma_commit_group();
//             // INJECTION: warpgroup::mma_async_wait();
//             warpgroup::store(kv_scratch[warpgroupid][0], kv_state[0]); // need to store k, v back to smem for wgmma
//             warpgroup::mma_async_wait();

//             featurize(qf_reg);
//             copy(qf_reg_bf, qf_reg); // now q has been featurized the first way

//             warpgroup::sync(); // shared memory writes need to have finished by this point.
//             warpgroup::mma_fence(o_reg); // launch first kv_state matmul
//             warpgroup::mm_AB(o_reg, qf_reg_bf, kv_scratch[warpgroupid][0]); // overwrite o_reg
//             warpgroup::mma_commit_group();
//             warpgroup::mma_async_wait();

//             warpgroup::mma_fence(qf_reg); // launch second featurization
//             warpgroup::mm_AB(qf_reg, q_smem[tic], q_map[1]);
//             warpgroup::mma_commit_group();
//             // INJECTION: warpgroup::mma_async_wait();
//             warpgroup::store(kv_scratch[warpgroupid][1], kv_state[1]); // need to store k, v back to smem for wgmma
//             warpgroup::mma_async_wait();

//             featurize(qf_reg);
//             copy(qf_reg_bf, qf_reg); // now q has been featurized the second way

//             warpgroup::sync(); // shared memory writes need to have finished by this point.
//             warpgroup::mma_fence(o_reg); // launch second kv_state matmul
//             warpgroup::mma_AB(o_reg, qf_reg_bf, kv_scratch[warpgroupid][1]); // accumulate into o_reg
//             warpgroup::mma_commit_group();
//             warpgroup::mma_async_wait(); // at this point all compute on o is done.

//             // NEXT: we need to update the state. We can immediately launch a k_map matmul from shared memory

//             warpgroup::mma_fence(kf_reg);
//             warpgroup::mm_AtBt(kf_reg, k_map[0], k_smem[tic]); // note we want to transpose k here on its way into registers
//             warpgroup::mma_commit_group();
//             // INJECTION: warpgroup::mma_async_wait();
            
//             // We've now done all the work on o that's required -- send it up to shared memory and let the producer run a cp.reduce.async.bulk.
//             warpgroup::store(o_smem[tic][warpgroupid], o_reg); // bf16

//             warpgroup::mma_async_wait();

//             featurize(kf_reg);
//             copy(kf_reg_bf, kf_reg); // now k has been featurized with the first map

//             warpgroup::mma_fence(kv_state[0]);
//             warpgroup::mma_AB(kv_state[0], kf_reg_bf, v_smem[tic][warpgroupid]); // not pre-transposed so AtB
//             warpgroup::mma_commit_group();

//             warpgroup::sync(); // o memory arrived in smem!
//             if(warpgroup::laneid() == 0) arrive(outputs_ready[tic]); // we've now told the producer it can send o along once it's ready.

//             warpgroup::mma_async_wait();

//             warpgroup::mma_fence(kf_reg); // launch second featurization
//             warpgroup::mm_AtBt(kf_reg, k_map[1], k_smem[tic]); // note we want to transpose k here on its way into registers
//             warpgroup::mma_commit_group();
//             warpgroup::mma_async_wait();

//             featurize(kf_reg);
//             copy(kf_reg_bf, kf_reg); // now k has been featurized the second way

//             warpgroup::mma_fence(kv_state[1]);
//             warpgroup::mma_AB(kv_state[1], kf_reg_bf, v_smem[tic][warpgroupid]); // pre-transposed so AB
//             warpgroup::mma_commit_group();
//             warpgroup::mma_async_wait();

//             // Finished with inputs
//             if(laneid == 0) arrive(inputs_finished[tic]); // we can now start loading the next q, k, v
//         }
//     }
// }

#include "harness.impl"