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
    relu(reg, reg);
}
template<ducks::rt::all RT> __device__ inline void featurize_backwards(RT &a_prefeature, RT &a_grad) {
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

// This is a producer+consumer copy kernel that demonstrates the use of TMA to implement a two-stage pipeline.
// TODO: We do actually have enough shared memory to make it a three-stage pipeline. 
__global__ __launch_bounds__(NUM_THREADS, 1)
void cylon_forwards(int N,
                    const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v,
                    CUtensorMap* tma_o, CUtensorMap* tma_kv_state,
                    const CUtensorMap* tma_q_map, const CUtensorMap* tma_k_map) {

    int laneid = kittens::laneid(), warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();
    int tic = 0, toc = 1; // these are used to track the two-stage pipeline.
    unsigned int batch_id = blockIdx.z; // which batch?
    unsigned int head_id  = blockIdx.y; // which head?
    unsigned int state_id = blockIdx.x; // which part of the KV state are we handling? (Important that this is on x.)
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
        int base_map_idx = (head_id * gridDim.x + state_id) * STATE_PER_SM; // gridDim.x is how many SMs are collaborating on each head
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
            int base_kv_idx = ((((batch_id * gridDim.y) + head_id) * gridDim.x) + state_id) * STATE_PER_SM;
            tma::store_async(tma_kv_state, kv_state_smem[0][warpgroupid], base_kv_idx+0, warpgroupid);
            tma::store_commit_group();
        }
        warpgroup::store(kv_state_smem[1][warpgroupid], kv_state[1]);
        warpgroup::sync();
        if(warpgroup::warpid() == 0) {
            int base_kv_idx = ((((batch_id * gridDim.y) + head_id) * gridDim.x) + state_id) * STATE_PER_SM;
            tma::store_async(tma_kv_state, kv_state_smem[1][warpgroupid], base_kv_idx+1, warpgroupid);
            tma::store_commit_group();
        }
        tma::store_async_read_wait();
        warpgroup::sync();
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

__device__ inline void wg_async_store_add(CUtensorMap* tma_dest, st_fl_4x4 &shared, const rt_fl_1x4<> &reg, const int index, const int col=0) {
    tma::store_async_read_wait();
    warpgroup::sync(warpgroup::groupid()+1);
    warpgroup::store(shared, reg);
    warpgroup::sync(warpgroup::groupid()+1);
    if(warpgroup::warpid() == 0) {
        tma::store_add_async(tma_dest, shared, index, col);
        tma::store_commit_group();
    }
}

// This is a producer+consumer copy kernel that demonstrates the use of TMA to implement a two-stage pipeline.
__global__ __launch_bounds__(NUM_THREADS, 1)
void cylon_backwards(int N,
                    const CUtensorMap* tma_q, const CUtensorMap* tma_k, const CUtensorMap* tma_v, // in fp32
                    const CUtensorMap* tma_q_map, const CUtensorMap* tma_k_map, // in bf16
                    const CUtensorMap* tma_o_grad, const CUtensorMap* tma_kv_state, // o grad in bf16, kv_state in fp32
                    CUtensorMap* tma_q_grad, CUtensorMap* tma_k_grad, CUtensorMap* tma_v_grad, // in fp32
                    CUtensorMap* tma_q_map_grad, CUtensorMap* tma_k_map_grad) { // in fp32

    int laneid = kittens::laneid(), warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();
    int tic = 0, toc = 1; // these are used to track the two-stage pipeline.
    unsigned int batch_id = blockIdx.z; // which batch?
    unsigned int head_id  = blockIdx.y; // which head?
    unsigned int state_id = blockIdx.x; // which part of the KV state are we handling? (Important that this is on x.)
    int n_chunks = N / 64; // this kernel handles 64 rows per chunk.

    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]); // allocate shared memory
    st_bf_4x4 (&q_smem)[2]     = alloc.allocate<st_bf_4x4, 2>(); // 64x64, tic-toc'd (16384)
    st_bf_4x4 (&k_smem)[2]     = alloc.allocate<st_bf_4x4, 2>(); // 64x64, tic-toc'd (16384)
    st_bf_4x4 (&v_smem)[2][2]  = alloc.allocate<st_bf_4x4, 2, 2>(); // 64x128, but easier to work with when split up. (32768)
    st_bf_4x4 (&og_smem)[2][2] = alloc.allocate<st_bf_4x4, 2, 2>(); // 64x128, but easier to work with when split up. (32768)

    // cumulative smem at this point: 16384 + 16384 + 32768 + 32768 = 98304

    st_bf_4x4 (&kv_state_smem)[2]         = alloc.allocate<st_bf_4x4, 2>();
    st_bf_4x4 (&kv_state_grad_smem)[2]    = alloc.allocate<st_bf_4x4, 2>();
    st_bf_4x4 (&qf_smem)                  = alloc.allocate<st_bf_4x4>();
    st_bf_4x4 (&kf_smem)                  = alloc.allocate<st_bf_4x4>();
    st_bf_4x4 (&qf_grad_smem)             = alloc.allocate<st_bf_4x4>();
    st_bf_4x4 (&kf_grad_smem)             = alloc.allocate<st_bf_4x4>();


    // cumulative smem at this point: 98304 + 65536 = 163840

    st_bf_4x4 (&q_map) = alloc.allocate<st_bf_4x4>(); // featurized q (8192)
    st_bf_4x4 (&k_map) = alloc.allocate<st_bf_4x4>(); // featurized k (8192)

    // cumulative smem at this point: 163840 + 8192 + 8192 = 180224

    st_fl_4x4 (&v_grad_smem)[2]        = alloc.allocate<st_fl_4x4, 2>(); // 64x64, tic-toc'd (32768)
    st_fl_4x4 (&q_grad_smem)           = reinterpret_cast<st_fl_4x4&>(v_grad_smem[0]);
    st_fl_4x4 (&k_grad_smem)           = reinterpret_cast<st_fl_4x4&>(v_grad_smem[1]);
    st_fl_4x4 (&kv_state_load_smem)[2] = reinterpret_cast<st_fl_4x4(&)[2]>(v_grad_smem); // we can reuse wgmma scratch for the initial kv load

    // cumulative smem at this point: 180224 + 32768 = 212992

    // Initialize barriers
    __shared__ kittens::barrier inputs_arrived[2], inputs_finished[2], setup_barrier;
    if (warpid == 0) {
        init_barrier(inputs_arrived[0], 0, 1); // needs to wait on just one memory transaction, plus confirmation that o is available for writing.
        init_barrier(inputs_arrived[1], 0, 1);
        init_barrier(setup_barrier, 0, 1); // wait for kv state to load
        init_barrier(inputs_finished[0], NUM_CONSUMER_WARPS, 0);
        init_barrier(inputs_finished[1], NUM_CONSUMER_WARPS, 0);
    }
    // Launch first load. No sync needed since thread 0 is doing these, too.
    if(warpid == 0) {
        tma::expect<st_fl_4x4, 2>(setup_barrier);
        // load kv state
        int kv_idx = (((batch_id * gridDim.y) + head_id) * gridDim.x) + state_id;
        tma::load_async(kv_state_load_smem[0], tma_kv_state, setup_barrier, kv_idx, 0);
        tma::load_async(kv_state_load_smem[1], tma_kv_state, setup_barrier, kv_idx, 1);
        // load q, k, v, q_map, k_map
        tma::expect<st_bf_4x4, 8>(inputs_arrived[0]); // 6 in bf16, 2 in fp32
        int load_idx = ((batch_id * gridDim.y) + head_id) * n_chunks + (n_chunks-1);
        tma::load_async(q_smem[tic],    tma_q, inputs_arrived[0], load_idx); // launch the initial load
        tma::load_async(k_smem[tic],    tma_k, inputs_arrived[0], load_idx); // launch the initial load
        tma::load_async(v_smem[tic][0], tma_v, inputs_arrived[0], load_idx, 0); // launch the initial load
        tma::load_async(v_smem[tic][1], tma_v, inputs_arrived[0], load_idx, 1); // launch the initial load
        tma::load_async(og_smem[tic][0], tma_o_grad, inputs_arrived[0], load_idx, 0); // launch the initial load
        tma::load_async(og_smem[tic][1], tma_o_grad, inputs_arrived[0], load_idx, 1); // launch the initial load
        // load q map, k map blocks
        int base_map_idx = (head_id * gridDim.x) + state_id; // gridDim.x is how many SMs are collaborating on each head
        tma::load_async(q_map, tma_q_map, inputs_arrived[0], base_map_idx);
        tma::load_async(k_map, tma_k_map, inputs_arrived[0], base_map_idx);
    }

    // if(blockIdx.x == 0 && laneid == 0) printf("(warp %d) about to hit syncthreads\n", warpid);
    __syncthreads(); // all warps must arrive here, confirming barrier initialization is visible to all threads.

    if(warpgroupid == NUM_CONSUMER_WARPGROUPS) { // last warpgroup is a producer
        warpgroup::decrease_registers<24>();
   
        if(warpid == NUM_CONSUMER_WARPS) { // just need a single warp to handle input loads
            for (int chunk_idx = 0; chunk_idx < n_chunks-1; chunk_idx++, tic=tic^1, toc=toc^1) {
                tma::expect<st_bf_4x4, 6>(inputs_arrived[toc]); // register that another block is coming in
                int next_load_idx = ((batch_id * gridDim.y) + head_id + 1) * n_chunks - chunk_idx - 2;
                tma::load_async(q_smem[toc],    tma_q, inputs_arrived[toc], next_load_idx); // load that block
                tma::load_async(k_smem[toc],    tma_k, inputs_arrived[toc], next_load_idx); // load that block
                tma::load_async(v_smem[toc][0], tma_v, inputs_arrived[toc], next_load_idx, 0); // load that block
                tma::load_async(v_smem[toc][1], tma_v, inputs_arrived[toc], next_load_idx, 1); // load that block
                tma::load_async(og_smem[toc][0], tma_o_grad, inputs_arrived[toc], next_load_idx, 0); // load that block
                tma::load_async(og_smem[toc][1], tma_o_grad, inputs_arrived[toc], next_load_idx, 1); // load that block
                wait(inputs_finished[tic], (chunk_idx/2)%2); // phase changes at half the rate of the tic/toc
            }
        }
    }
    else { // other warpgroups are consumers
        warpgroup::increase_registers<240>(); // 240 registers, no spills!
        using consumers = group<8>;

        rt_fl_1x4<> kv_state, kv_state_grad;
        zero(kv_state_grad); // zero initial grads
        wait(setup_barrier, 0);
        warpgroup::load(kv_state, kv_state_load_smem[warpgroupid]);
        mul(kv_state, kv_state, -1.f); // this is so that we can do mma's onto it with + later.
        warpgroup::store(kv_state_smem[warpgroupid], kv_state);
        consumers::sync(0);

        if(warpgroupid == 0) {
            rt_fl_1x4<> q_map_grad;
            zero(q_map_grad);    // zero initial grads

            for (int chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++, tic^=1, toc^=1) {
                int global_chunk_idx = ((batch_id * gridDim.y) + head_id + 1) * n_chunks - chunk_idx - 1;
                // if(blockIdx.x == 0 && laneid == 0) printf("(warp %d) starting iteration %d\n", warpid, chunk_idx);
                
                wait(inputs_arrived[tic], (chunk_idx/2)%2); // wait for memory to arrive
                consumers::sync(0); // DEBUGGER

                rt_fl_1x4<> qf, qf_relu;
                warpgroup::mma_fence(qf);
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
                warpgroup::mma_fence(kv_state);
                warpgroup::mma_AtB(kv_state, kf_smem, v_smem[tic][warpgroupid]); // update kv state
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                consumers::sync(0); // DEBUGGER

                warpgroup::store(kv_state_smem[warpgroupid], kv_state);

                warpgroup::mma_fence(kv_state_grad);
                warpgroup::mma_AtB(kv_state_grad, qf_smem, og_smem[tic][warpgroupid]);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                consumers::sync(0); // everyone done, no prior dependencies

                // Next this warpgroup will be responsible for handling q_grad, and q_map_grad
            
                // q_head_chunk_grad = -torch.einsum('nv,kv->nk', o_grad_chunk, kv) * (q_head_prerelu>0)
                rt_fl_1x4<> qf_grad, q_grad;
                warpgroup::mma_fence(qf_grad);
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
                warpgroup::mma_fence(q_grad);
                warpgroup::mm_ABt(q_grad, qf_grad_smem, q_map);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                wg_async_store_add(tma_q_grad, q_grad_smem, q_grad, global_chunk_idx);

                // q_map_grad[s] += torch.einsum('nd,nk->dk', q_chunk, q_head_chunk_grad)
                warpgroup::mma_fence(q_map_grad);
                warpgroup::mma_AtB(q_map_grad, q_smem[tic], qf_grad_smem);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                // everyone must do v_grad:
                rt_fl_1x4<> v_grad;
                warpgroup::mma_fence(v_grad);
                warpgroup::mm_AB(v_grad, kf_smem, kv_state_grad_smem[warpgroupid]);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait(); // v_grad is now finished and in fp32
                
                wg_async_store_add(tma_v_grad, v_grad_smem[warpgroupid], v_grad, global_chunk_idx, warpgroupid); // v is wider

                // Finished with inputs
                if(laneid == 0) arrive(inputs_finished[tic]); // we can now start loading the next q, k, v
            }
            // write out q map grad
            div(q_map_grad, q_map_grad, float(gridDim.z)); // need to take a mean on the batch dimension
            consumers::sync(0); // wait for all warpgroups to finish
            int base_map_idx = (head_id * gridDim.x) + state_id; // gridDim.x is how many SMs are collaborating on each head
            wg_async_store_add(tma_q_map_grad, q_grad_smem, q_map_grad, base_map_idx);
            tma::store_async_wait();
            warpgroup::sync();
        }
        else {
            rt_fl_1x4<> k_map_grad;
            zero(k_map_grad);    // zero initial grads

            for (int chunk_idx = 0; chunk_idx < n_chunks; chunk_idx++, tic^=1, toc^=1) {
                int global_chunk_idx = ((batch_id * gridDim.y) + head_id + 1) * n_chunks - chunk_idx - 1;
                // if(blockIdx.x == 0 && laneid == 0) printf("(warp %d) starting iteration %d\n", warpid, chunk_idx);
                
                wait(inputs_arrived[tic], (chunk_idx/2)%2); // wait for memory to arrive
                consumers::sync(0); // DEBUGGER

                rt_fl_1x4<> kf, kf_relu;
                warpgroup::mma_fence(kf);
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
                warpgroup::mma_fence(kv_state);
                warpgroup::mma_AtB(kv_state, kf_smem, v_smem[tic][warpgroupid]); // update kv state
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                consumers::sync(0); // DEBUGGER

                warpgroup::store(kv_state_smem[warpgroupid], kv_state);

                warpgroup::mma_fence(kv_state_grad);
                warpgroup::mma_AtB(kv_state_grad, qf_smem, og_smem[tic][warpgroupid]);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                consumers::sync(0); // everyone done, no prior dependencies

                // Next this warpgroup will be responsible for handling k_grad, and k_map_grad

                rt_fl_1x4<> kf_grad, k_grad;
                warpgroup::mma_fence(kf_grad);
                warpgroup::mm_ABt(kf_grad, v_smem[tic][0], kv_state_grad_smem[0]);
                warpgroup::mma_ABt(kf_grad, v_smem[tic][1], kv_state_grad_smem[1]);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();
                featurize_backwards(kf, kf_grad);
                warpgroup::store(kf_grad_smem, kf_grad);
                warpgroup::sync(2);

                consumers::sync(0); // apparently necessary

                // k_grad += torch.einsum('nk,dk->nd', k_head_chunk_grad, k_map)
                warpgroup::mma_fence(k_grad);
                warpgroup::mm_ABt(k_grad, kf_grad_smem, k_map);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait(); // k_grad is now finished and in fp32

                wg_async_store_add(tma_k_grad, k_grad_smem, k_grad, global_chunk_idx);

                // k_map_grad[s] += torch.einsum('nd,nk->dk', k_chunk, k_head_chunk_grad)
                warpgroup::mma_fence(k_map_grad);
                warpgroup::mma_AtB(k_map_grad, k_smem[tic], kf_grad_smem);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                // everyone must do v_grad:
                // v_grad += torch.einsum('nk,kv->nv', torch.relu(k_head_prerelu), kv_grad)
                rt_fl_1x4<> v_grad;
                warpgroup::mma_fence(v_grad);
                warpgroup::mm_AB(v_grad, kf_smem, kv_state_grad_smem[warpgroupid]);
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait(); // v_grad is now finished and in fp32

                wg_async_store_add(tma_v_grad, v_grad_smem[warpgroupid], v_grad, global_chunk_idx, warpgroupid); // v is wider

                // Finished with inputs
                if(laneid == 0) arrive(inputs_finished[tic]); // we can now start loading the next q, k, v
            }
            // write out k map grad
            div(k_map_grad, k_map_grad, float(gridDim.z)); // need to take a mean on the batch dimension
            consumers::sync(0); // wait for all warpgroups to finish
            int base_map_idx = (head_id * gridDim.x) + state_id; // gridDim.x is how many SMs are collaborating on each head
            wg_async_store_add(tma_k_map_grad, k_grad_smem, k_map_grad, base_map_idx);
            tma::store_async_wait();
            warpgroup::sync();
        }
    }
}

#include "harness.impl"