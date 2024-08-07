#include "kittens.cuh"
#include "common/debug.cuh"
#include "cooperative_groups.h"
namespace cg = cooperative_groups;
using namespace kittens;

constexpr int CLUSTER_SIZE = 8;
constexpr int NUM_CONSUMER_WARPGROUPS = 2;
constexpr int NUM_CONSUMER_WARPS = NUM_CONSUMER_WARPGROUPS * WARPGROUP_WARPS;
constexpr int NUM_WARPS = NUM_CONSUMER_WARPS + WARPGROUP_WARPS; // producers, too
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

#define COLOR_PRINT(color, format, ...) \
    /* do { \
        if (laneid == 0) { \
            if (block_rank) printf(FG_BRIGHT_YELLOW "(block=%d warp=%d) " color format RESET "\n", block_rank, warpid, ##__VA_ARGS__); \
            else            printf(FG_BRIGHT_RED    "(block=%d warp=%d) " color format RESET "\n", block_rank, warpid, ##__VA_ARGS__); \
        } \
    } while(0) */

template<ducks::rt::all RT> __device__ inline void featurize(RT &reg) {
    // This is a placeholder for whatever featurization we want to do.
    // In this case, I'll just use a ReLU for simplicity.
    relu(reg, reg);
}

// This is a producer+consumer copy kernel that demonstrates the use of TMA to implement a two-stage pipeline.
__global__ __launch_bounds__(NUM_THREADS, 1) __cluster_dims__(CLUSTER_SIZE, 1, 1)
void cylon_linear_attention(int N, const CUtensorMap* tma_q, CUtensorMap* tma_k, CUtensorMap* tma_v, CUtensorMap* tma_o)  {

    int laneid = kittens::laneid(), warpid = kittens::warpid(), warpgroupid = warpgroup::groupid();
    int tic = 0, toc = 1; // these are used to track the two-stage pipeline.
    int n_blocks = N / 64; // this kernel handles 64 rows per block.
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int block_rank = cluster.block_rank();
    unsigned int cluster_rank = blockIdx.x / CLUSTER_SIZE;

    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]); // allocate shared memory
    st_bf_4x4 (&q_smem)[2]    = alloc.allocate<st_bf_4x4, 2>(); // 64x64, tic-toc'd (16384)
    st_bf_4x4 (&k_smem)[2]    = alloc.allocate<st_bf_4x4, 2>(); // 64x64, tic-toc'd (16384)
    st_bf_4x4 (&v_smem)[2][2] = alloc.allocate<st_bf_4x4, 2, 2>(); // 64x128, but easier to work with when split up. (32768)
    // o_smem also needs a double buffer, but it's not really tic toc. Rather, [0] is incoming from another thread, and [1] is outgoing.
    st_fl_4x4 (&o_smem)[2][2] = alloc.allocate<st_fl_4x4, 2, 2>(); // 64x128, but easier to work with when split up. (65536 as fp32)

    st_bf_4x4 (&kv_scratch)[2][2] = alloc.allocate<st_bf_4x4, 2, 2>(); // This is scratch for doing wgmma's

    st_bf_4x4 (&o_smem_out)[2] = reinterpret_cast<st_bf_4x4 (&)[2]>(o_smem[1][0]); // for the final block, the global store should be in bf16, not fp32

    // Initialize barriers
    __shared__ kittens::barrier inputs_arrived[2], inputs_finished[2], outputs_arrived[2], outputs_finished[2];
    __shared__ kittens::barrier ready_for_outputs; // wait to be signaled across the cluster that the next SM is ready to be sent outputs
    if (warpid == 0) {
        init_barrier(inputs_arrived[0], 0, 1); // needs to wait on just one memory transaction, each
        init_barrier(inputs_arrived[1], 0, 1);
        init_barrier(inputs_finished[0], NUM_CONSUMER_WARPS, 0); // needs to wait on just one memory transaction, each
        init_barrier(inputs_finished[1], NUM_CONSUMER_WARPS, 0);
        init_barrier(outputs_arrived[0], 0, 1); // needs to wait on one thread from each consumer warp
        init_barrier(outputs_arrived[1], 0, 1);
        init_barrier(outputs_finished[0], NUM_CONSUMER_WARPS, 0); // needs to wait on one thread from each consumer warp
        init_barrier(outputs_finished[1], NUM_CONSUMER_WARPS, 0);
        init_barrier(ready_for_outputs, NUM_CONSUMER_WARPGROUPS, 0); // both consumer warpgroups need to agree they're ready.
    }
    // Launch first load. No sync needed since thread 0 is doing these, too.
    if(warpid == 0) {
        tma::expect<st_bf_4x4, 4>(inputs_arrived[0]); // register a transaction for q, k, v
        int block_idx = n_blocks * cluster_rank;
        tma::load_async(q_smem[tic],    tma_q, inputs_arrived[0], block_idx, block_rank); // launch the initial load
        tma::load_async(k_smem[tic],    tma_k, inputs_arrived[0], block_idx, block_rank); // launch the initial load
        tma::load_async(v_smem[tic][0], tma_v, inputs_arrived[0], block_idx, 0); // launch the initial load
        tma::load_async(v_smem[tic][1], tma_v, inputs_arrived[0], block_idx, 1); // launch the initial load
    }
    if(warpgroupid > 0) {
        warpgroup::zero(kv_scratch[warpgroupid-1][0]); // zero out all kv state to begin with.
        warpgroup::zero(kv_scratch[warpgroupid-1][1]);
    }

    COLOR_PRINT(FG_BRIGHT_BLUE, "finished init, arriving at cluster sync");
    cluster.sync(); // all warps must arrive here, confirming barrier initialization is visible to all threads.
    COLOR_PRINT(FG_BRIGHT_MAGENTA, "finished cluster sync");

    if(warpgroupid == NUM_CONSUMER_WARPGROUPS) { // last warpgroup is a producer
        warpgroup::decrease_registers<24>();
        COLOR_PRINT(FG_BRIGHT_WHITE, "(DEBUG) decreased registers");
   
        if(warpid == NUM_CONSUMER_WARPS) { // just need a single warp to do this
            for (int block_idx = 0; block_idx < n_blocks-1; block_idx++, tic=tic^1, toc=toc^1) {
                tma::expect<st_bf_4x4, 4>(inputs_arrived[toc]); // register that another block is coming in
                int next_tile_idx = n_blocks * cluster_rank + block_idx + 1;
                tma::load_async(q_smem[toc], tma_q, inputs_arrived[toc], next_tile_idx); // load that block
                tma::load_async(k_smem[toc], tma_k, inputs_arrived[toc], next_tile_idx); // load that block
                tma::load_async(v_smem[toc][0], tma_v, inputs_arrived[toc], next_tile_idx, 0); // load that block
                tma::load_async(v_smem[toc][1], tma_v, inputs_arrived[toc], next_tile_idx, 1); // load that block
                COLOR_PRINT(BG_WHITE FG_BRIGHT_GREEN, "launched input load for iter %d", next_tile_idx);
                wait(inputs_finished[tic], (block_idx/2)%2); // phase changes at half the rate of the tic/toc
            }
        }
        else if(warpid == NUM_CONSUMER_WARPS + 1) {
            for (int block_idx = 0; block_idx < n_blocks; block_idx++, tic=tic^1, toc=toc^1) {
                COLOR_PRINT(BG_WHITE FG_BRIGHT_YELLOW, "output/dsmem producer waiting for outputs finished at iter %d", block_idx);
                wait(outputs_finished[tic], (block_idx/2)%2); // phase changes at half the rate of the tic/toc
                if(block_rank == CLUSTER_SIZE - 1) {
                    // Last block writes to global memory
                    int tile_row = n_blocks * cluster_rank + block_idx;
                    tma::store_async(tma_o, o_smem_out[0], tile_row, 0); // store
                    tma::store_async(tma_o, o_smem_out[1], tile_row, 1);
                    tma::store_commit_group();
                    tma::store_async_read_wait();
                }
                else {
                    COLOR_PRINT(BG_WHITE FG_BRIGHT_YELLOW, "output/dsmem producer waiting for ready signal at iter %d", block_idx);
                    tma::cluster::wait(ready_for_outputs, toc); // should not block at the beginnning, hence wait on toc
                    tma::cluster::expect<st_fl_4x4, 2>(outputs_arrived[tic], block_rank+1); // tell the next cluster to prepare for the next block
                    tma::cluster::store_async(o_smem[0][0], o_smem[1][0], block_rank+1, outputs_arrived[tic]); // send them outputs
                    tma::cluster::store_async(o_smem[0][1], o_smem[1][1], block_rank+1, outputs_arrived[tic]); // send them outputs
                    COLOR_PRINT(BG_WHITE FG_BRIGHT_YELLOW, "dsmem producer sent outputs at iter %d", block_idx);
                }
            }
        }
    }
    else { // other warpgroups are consumers
        constexpr int n_reg = NUM_CONSUMER_WARPGROUPS <= 3 ? 480/NUM_CONSUMER_WARPGROUPS : 480/NUM_CONSUMER_WARPGROUPS-8;
        warpgroup::increase_registers<n_reg>(); // valid up to 6 consumer warpgroups, and I can't imagine wanting more
        COLOR_PRINT(FG_BRIGHT_WHITE, "(DEBUG) increased registers");

        rt_fl_1x4<> kv_state[2];
        zero(kv_state[0]);
        zero(kv_state[1]);

        for (int block_idx = 0; block_idx < n_blocks; block_idx++, tic^=1, toc^=1) {
            COLOR_PRINT(FG_BRIGHT_WHITE, "consumer start loop %d", block_idx);
            
            wait(inputs_arrived[tic], (block_idx/2)%2); // wait for memory to arrive
            COLOR_PRINT(FG_BRIGHT_GREEN, "consumer input memory arrived %d", block_idx);

            rt_fl_1x4<> o_reg;
            if(block_rank > 0) {
                wait(outputs_arrived[tic], (block_idx/2)%2);
                warpgroup::load(o_reg, o_smem[0][warpgroupid]);
            }
            else {
                zero(o_reg); // Just zero to start if this is the first warpgroup
            }
            COLOR_PRINT(FG_BRIGHT_GREEN, "consumer accumulator memory arrived %d", block_idx);

            // We launch first matmul with identity
            warpgroup::sync(); // all shared memory writes must be visible by now, too.
            warpgroup::mma_fence(o_reg);
            warpgroup::mma_AB(o_reg, q_smem[tic], kv_scratch[warpgroupid][0]); // first one is identity
            warpgroup::mma_commit_group();

            // we can now tell the previous SM that it can send outputs. (while matmul.)
            if(block_rank > 0 && warpgroup::laneid() == 0) {
                tma::cluster::arrive(ready_for_outputs, block_rank-1);
                COLOR_PRINT(FG_BRIGHT_CYAN, "consumer signaled ready for outputs, iter=%d to block=%d", block_idx, block_rank-1);
            }

            // Next we load q state into registers
            rt_bf_1x4 q_reg;
            warpgroup::load(q_reg, q_smem[tic]);
            // Do whatever featurization we want -- in this case, I'll just use a ReLU for simplicity.
            featurize(q_reg);

            warpgroup::mma_async_wait(); // Finish MMA

            // Launch the other half
            warpgroup::mma_fence(o_reg);
            warpgroup::mma_AB(o_reg, q_reg, kv_scratch[warpgroupid][1]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();

            // We've now done all the work on o that's required -- send it up to shared memory and let the producer run a cp.async.bulk.
            if(block_rank == CLUSTER_SIZE-1) warpgroup::store(o_smem_out[warpgroupid], o_reg); // store out in bf16
            else                             warpgroup::store(o_smem[1][warpgroupid], o_reg); // fp32
            warpgroup::sync(); // arrived!
            if(laneid == 0) arrive(outputs_finished[tic]); // we've now told the producer it can send o along once it's ready.
            COLOR_PRINT(FG_BRIGHT_CYAN, "consumer signaled outputs finished on iter=%d", block_idx);

            // NEXT: we need to update the state. We can immediately launch a K@V matmul onto kv_state[0] from shared memory using the identity.
            warpgroup::mma_fence(kv_state[0]);
            warpgroup::mma_AtB(kv_state[0], k_smem[tic], v_smem[tic][warpgroupid]); // not pre-transposed so AtB
            warpgroup::mma_commit_group();

            rt_bf_4x1 k_reg;
            rt_bf_1x4 k_reg_trans;
            auto st = subtile_inplace<4,1>(k_smem[tic], 0, warpid%4); // grab the right column of k
            load(k_reg, st); // each loads their subtile
            transpose_sep(k_reg_trans, k_reg); // transpose it into k_reg_trans
            featurize(k_reg_trans); // featurize k

            warpgroup::mma_async_wait();

            warpgroup::mma_fence(kv_state[1]);
            warpgroup::mma_AB(kv_state[1], k_reg_trans, v_smem[tic][warpgroupid]); // pre-transposed so AB
            warpgroup::mma_commit_group();

            warpgroup::store(kv_scratch[warpgroupid][0], kv_state[0]);

            warpgroup::mma_async_wait();

            // Finished with inputs
            if(laneid == 0) arrive(inputs_finished[tic]); // we can now start loading the next q, k, v
            COLOR_PRINT(FG_BRIGHT_CYAN, "consumer signaled inputs finished on iter=%d", block_idx);

            warpgroup::store(kv_scratch[warpgroupid][1], kv_state[1]); // need to store k, v back to smem for wgmma
            COLOR_PRINT(FG_BRIGHT_WHITE, "consumer finished loop %d", block_idx);
        }
    }
    cluster.sync();
}
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line ) {
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
}

#include <iostream>
#include <fstream>
constexpr int B     = 128;
constexpr int N     = 1024;
constexpr int D_QK  = 64;
constexpr int D_VO   = 128;
uint64_t flops(int b, int n, int d_qk, int d_vo, int cluster_size) {
    // how many chunks of 64 need to be done?
    uint64_t flops_per_sm = (64*d_qk*d_vo*2); // a single Q@(state) or K.T@V matmul
    flops_per_sm *= 2; // we have normal and relu versions
    flops_per_sm *= 2; // we also need to do both a Q@state AND a K.T@V matmul in each loop
    flops_per_sm *= n / 64; // how many chunks do we do?
    return flops_per_sm * b * cluster_size;
}
int main() {
    // Create arrays
    bf16* q = new bf16[B * CLUSTER_SIZE * N * D_QK];
    bf16* k = new bf16[B * CLUSTER_SIZE * N * D_QK];
    bf16* v = new bf16[B * N * D_VO];
    bf16* o = new bf16[B * N * D_VO];

    for(int i = 0; i < B * CLUSTER_SIZE * N * D_QK; i++) { // Initialize memory with something memorable.
        q[i] = __float2bfloat16(1.0);
        k[i] = __float2bfloat16(1.0);
    }
    for(int i = 0; i < B * N * D_VO; i++) {
        v[i] = __float2bfloat16(1.0);
    }

    // Create CUDA tensors
    bf16 *d_q, *d_k, *d_v, *d_o;
    cudaMalloc(&d_q, B * CLUSTER_SIZE * N * D_QK * sizeof(bf16));
    cudaMalloc(&d_k, B * CLUSTER_SIZE * N * D_QK * sizeof(bf16));
    cudaMalloc(&d_v, B * N * D_VO * sizeof(bf16));
    cudaMalloc(&d_o, B * N * D_VO * sizeof(bf16));

    // Copy data to GPU
    cudaMemcpy(d_q, q, B * CLUSTER_SIZE * N * D_QK * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k, B * CLUSTER_SIZE * N * D_QK * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, B * N * D_VO * sizeof(bf16), cudaMemcpyHostToDevice);

    // Create tma descriptors on both
    CUtensorMap* q_tma  = tma::allocate_and_create_tensor_map<st_bf_4x4>(d_q, B * N / 64, CLUSTER_SIZE);
    CUtensorMap* k_tma  = tma::allocate_and_create_tensor_map<st_bf_4x4>(d_k, B * N / 64, CLUSTER_SIZE);
    CUtensorMap* v_tma  = tma::allocate_and_create_tensor_map<st_bf_4x4>(d_v, B * N / 64, 2);
    CUtensorMap* o_tma  = tma::allocate_and_create_tensor_map<st_bf_4x4>(d_o, B * N / 64, 2);

    // Launch kernel
    cudaFuncSetAttribute(cylon_linear_attention, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000);
    cudaFuncSetAttribute(cylon_linear_attention, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); // allow 16 thread block

    auto start = std::chrono::high_resolution_clock::now();

    cylon_linear_attention<<<B*CLUSTER_SIZE, NUM_THREADS, 200000>>>(N, q_tma, k_tma, v_tma, o_tma);
    CudaCheckError();

    // Wait for kernel to finish
    cudaDeviceSynchronize();
    CudaCheckError();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Kernel execution time: " << duration.count() << " microseconds" << std::endl;
    uint64_t flops_executed = flops(B, N, D_QK, D_VO, CLUSTER_SIZE);
    std::cout << "FLOPs executed: " << flops_executed/1e12 << "T\n";
    std::cout << "TFLOPs achieved: " << flops_executed / duration.count() / 1e6 << std::endl;

    // Copy data from GPU
    cudaMemcpy(o, d_o, B * N * D_VO * sizeof(bf16), cudaMemcpyDeviceToHost);

    // Dump o to a text file
    std::ofstream outfile("output.txt");
    if (outfile.is_open()) {
        for (int i = 0; i < N * D_VO; i++) { // only dump one head
            outfile << __bfloat162float(o[i]) << " ";
            if ((i + 1) % 128 == 0) { // write out each line of o independently
                outfile << "\n";
            }
        }
        outfile.close();
        std::cout << "Output dumped to output.txt" << std::endl;
    } else {
        std::cerr << "Unable to open output file" << std::endl;
        return 1;
    }

    std::cout << "Not crashed!" << std::endl;
    return 0;
}