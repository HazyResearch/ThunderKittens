#include "kittens.cuh"

using namespace kittens;

constexpr int WORLD_SIZE = 8;
constexpr int DP_SIZE = 1;
constexpr int NUM_EXPERTS = 256;

constexpr int HIDDEN_DIM_SIZE = 7168;
// Force HIDDEN_DIM_SCALE_SIZE to be multiple of 16
constexpr int HIDDEN_DIM_SCALE_SIZE = 64; // HIDDEN_DIM_SIZE / BLOCK_SIZE => HIDDEN_DIM_SIZE / 128
constexpr int TOKEN_DIM = HIDDEN_DIM_SIZE + HIDDEN_DIM_SCALE_SIZE;

constexpr int NUM_WARPS = 10;

using g_pgl = pgl<gl<int, -1, -1, -1, -1>>;
using g_gl = gl<int, -1, -1, -1, -1>;

using g_buffer = pgl<gl<int, -1, -1, -1, -1>>;
using tok_idx_sv = sv<int, NUM_EXPERTS>;

// 736 * 10 = 7360 => Only first TOKEN DIM elements are used though
// using tok_rt = rt<float, 16, 46>;
using tok_rv = rv<int, 736>;
using tok_sv = sv<int, HIDDEN_DIM_SIZE>;

/*
Grid size is 132
*/

struct globals {
    // expertX: numLocalExperts, maxNumTokens * numDPGroups, perTokenBytes
    g_gl expertX; // destination array for token data 
    // g_pgl expertXScale;

    // dpX: numTokens * hiddenDimBytes
    g_gl dpX; // src array for token data to send off 
    // g_pgl dpXScale;

    
    /*
    numTokensBuffer: numDPGroups, numLocalExperts
    */
    g_buffer numTokensBuffer;
    /*
    numRecvBuffer: numDPGroups, numLocalExperts
    */
    g_buffer numRecvBuffer;
    /*
    xBufferOut: 8, 32, 128, TOKEN_DIM
    xBufferOut: numDPGroups, numLocalExperts, maxNumTokens, perTokenBytes
    */
    g_buffer xBufferOut;
    
    
    int *outNumTokensPerExpert; // ensure this is zero'd out before calling t
    // inidices is m * numExpertsPerToken
    // Reasonable to load into a r
    int *indices;
    int *numTokensPerDP; // make non pgl

    // uint32_t *sourceExpert;
    // uint32_t *sourceIndex;
    // uint32_t *sourceOffset;
    // uint32_t *sourceGroup;
    
    size_t maxNumTokens;
    size_t numExperts;
    size_t numExpertsPerToken;

    int numTokens; // num tokens cur device needs to send (m from pplx)
};

__global__ void dispatch_kernel(const __grid_constant__ globals g, int dev_idx) {
    using everyone = kittens::group<NUM_WARPS>;
    using send_group = kittens::group<NUM_WARPS - 1>;
    using recv_group = kittens::group<NUM_WARPS>;
    const unsigned num_local_experts = NUM_EXPERTS / WORLD_SIZE;

    /*
    Send Phase
    */
    // extern __shared__ uint32_t tokenIndex[];
    // for (uint32_t i = threadIdx.x; i < g.numExperts; i += blockDim.x) {
    //   tokenIndex[i] = 0;
    // }
    // __syncthreads();

    extern __shared__ kittens::alignment_dummy __shm[]; 
    kittens::shared_allocator al((int*)&__shm[0]);
    tok_idx_sv &tokenIndex = al.allocate<tok_idx_sv>();
    kittens::zero(tokenIndex);
    everyone::sync(0);

    // Do last warp here so can use "groups" for shared memory loads
    if (kittens::warpid() == NUM_WARPS - 1) {
        /*
        There are 256 experts in total, each device is responsible for 32 experts 

        Last warp needs to determine how many tokens each expert is responsible for
            - To do this, each warp across all devices loop through all 256 experts
                - Each device (in this scenario) has 132 blocks

        Starting g_expert_idx is between 0 and 132:
            - Block Idx 0 handles expert 0, 132, 
            ...
            - Block Idx 124 handles expert 124, 256
        */
        for (size_t g_expert_idx = blockIdx.x; g_expert_idx < NUM_EXPERTS; g_expert_idx += gridDim.x) {
            int dst_gpu = g_expert_idx / num_local_experts;
            int dst_expert_idx = g_expert_idx % num_local_experts;

            int count = 0;
            
            /*
            The loop is looping through all tokens for current device and checking if it 
            is equal to the global expert
                - all threads in a block look through all routings given by indices 
                and if it equals the current expert block is handling, increment count
            

            How do we make this kittens? 
            - load all of indices into an rt of equivalent size
            - want to create tile then that is all zero's unless is equal to dst_expert_idx
            - then 
            - some sort of shared buffer 

            how are we going to 
            - Would be nice if we can just reduce add and send... how can we reduce add to correct 
            location 
            - how large of a tile would that be? 

            have tile that represents all experts => JUST 256
            have local register tile of 256 => keep local count, logic, and then atomic add RT to 

            want to use kittens load functions to optimize loads? 

            
            */
            #pragma unroll
            for (int i = laneid(); i < g.numTokens * g.numExpertsPerToken; i += WARP_THREADS) {
                // int expert = __ldg(&g.indices[i]);
                // if (expert == dst_expert_idx) count++;
            }
            
            count += __shfl_xor_sync(0xffffffff, count, 16);
            count += __shfl_xor_sync(0xffffffff, count, 8);
            count += __shfl_xor_sync(0xffffffff, count, 4);
            count += __shfl_xor_sync(0xffffffff, count, 2);
            count += __shfl_xor_sync(0xffffffff, count, 1);
        
            if (laneid() == 0) {
                // g.numTokensBuffer.mc_vas[g.dev_idx][dst_expert_idx] = count + 1;
                // Could be faster to just store directly instead of multimem here => test difference
                // asm volatile ("{multimem.red.release.sys.global.add.u32 [%0], %1;}" 
                //   :: "l"(&(g.numTokensBuffer.mc_vas[dev_idx][dst_expert_idx])), 
                //      "r"(count + 1) : "memory");
            }
        }
    } else {
        for (int i = 0; i < g.numTokens; ++i) {
            // Each block handles a different token
            // In example, max numTokens is 128 so only 128 / 132 blocks will be active
            if (i % gridDim.x == blockIdx.x) {
                /*
                Here, we want to take data from dpX which is the local token data
                currently on device and place it into xBufferOut 

                Need to send both dpX and dpXScale to the expert

                dpX is hiddenDim size (7168)
                dpXScale is hiddenDimScale (7168 / 128) => 56
                    - dpXScale is hiddenDimScale (7168 / 112) => 64

                Have 10 Warps to load => Just load straight to shared memory

                Use TMA here eventually??
                */
                tok_sv &tok_dst_data = al.allocate<tok_sv>();
                // send_group::load(tok_dst_data, g.dpX, {i});
                send_group::sync(0);

                for (int j = 0; j < g.numExpertsPerToken; ++j) {
                    // int dst_expert = __ldg(&g.indices[i * g.numExpertsPerToken + j]);
                    // int dst_gpu = dst_expert / num_local_experts;
                    // int dst_expert_idx = dst_expert % num_local_experts;
                    
                    // Get the number of tokens that have currently been sent to the expert
                    // int index = tokenIndex.data[dst_expert];

                    // Experiment with just using store here??
                    // send_group::broadcast(g.xBufferOut, tok_dst_data, dev_idx, 
                    //     {dev_idx, dst_expert_idx, index, 0});
                    if (warpid() == 0 && laneid() == 0) {
                        // asm volatile ("{multimem.red.release.sys.global.add.u32 [%0], %1;}" 
                        //     :: "l"(&(g.numRecvBuffer.mc_vas[dev_idx][dst_expert_idx])), "n"(1) : "memory");
                    }
                }

                // Replicate token count across all blocks
                if (warpid() == 0 && laneid() < g.numExpertsPerToken) {
                    // int dst_expert = __ldg(&g.indices[i * g.numExpertsPerToken + laneid()]);
                    // tokenIndex.data[dst_expert]++;
                }
            }


        }


    }

    /*
    Receive Phase
    */
    /*
    first 32 threads of device stall untill device receives all tokens expected
    */
    for (size_t expert = blockIdx.x * blockDim.x + threadIdx.x;
         expert < num_local_experts; expert += blockDim.x * gridDim.x) {
        // int src_local_expert = expert / WORLD_SIZE;
        // // Stall until numTokensBuffer is updated 
        // while (g.numTokensBuffer[dev_idx].raw_ptr[src_local_expert] == 0);
        // size_t numTokens = g.numTokensBuffer[dev_idx].raw_ptr[src_local_expert] - 1;
        // // Stall here until we know that all tokens have been received
        // while (g.numRecvBuffer[dev_idx].raw_ptr[src_local_expert] < numTokens);
        
        // int src_gpu = expert % WORLD_SIZE;
        // int slot = src_local_expert * WORLD_SIZE + src_gpu;
        // g.numTokensPerDP[slot] = numTokens;
        // atomicAdd(&g.outNumTokensPerExpert[src_local_expert], numTokens);
        
        // g.numTokensBuffer[dev_idx].raw_ptr[src_local_expert] = 0;
        // g.numRecvBuffer[dev_idx].raw_ptr[src_local_expert] = 0;
    }
    
    // cg::this_grid().sync();
    int expert = 0;
    int device = 0;
    int offset = 0;
    int start = 0;
    int max_batch_tokens = num_local_experts * WORLD_SIZE * g.maxNumTokens;

    /*
    Each block handles it's own token
    */
    for (int token = blockIdx.x; token < max_batch_tokens; token += gridDim.x) {
        int j = token - offset;
        while (offset + __ldg(&g.numTokensPerDP[expert * WORLD_SIZE + device]) <= token) {
            // Since current token not in current group, add number of tokens in group to offset
            offset += __ldg(&g.numTokensPerDP[expert * WORLD_SIZE + device]);
            j = token - offset;
            if (++device == WORLD_SIZE) {
                device = 0;
                start = offset;
                if (++expert == num_local_experts) {
                break;
                }
            }
            }
            if (expert >= num_local_experts) {
            break;
            }

            // Copy the token to the output buffer.
            int group = expert * WORLD_SIZE + device;
            int loc = token - start;

            tok_sv &tok_data = al.allocate<tok_sv>();
            
            // recv_group::load(tok_data, g.xBufferOut);
            recv_group::sync(0);
            // recv_group::store(g.expertX, tok_data, {expert, loc});
    }
}

template <typename T> __host__ T ceil_div(T x, T y) { return (x + y - 1) / y; }

void run_benchmark(int max_num_tokens, int num_experts, int experts_per_token, int hidden_dim, int scale_block_size) {
    CUCHECK(cuInit(0));
    int device_ids[WORLD_SIZE];
    for (int i = 0; i < WORLD_SIZE; ++i) device_ids[i] = i;
    KittensClub club(device_ids, WORLD_SIZE);

    unsigned long smem_size = MAX_SHARED_MEMORY - 1024; // MAX_SHARED_MEMORY = 227KB for Hopper
    club.execute([smem_size](int dev_idx) {
        CUDACHECK(cudaFuncSetAttribute(dispatch_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, 
                                        smem_size));
    });

    const unsigned numBlocks = std::min( // min( max((256 / 10), (128 * 8)), 132) = 132
        std::max(
            ceil_div(num_experts, NUM_WARPS), (max_num_tokens * experts_per_token)
        ),
        132
    );
    
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(NUM_WARPS * 32, 1, 1);

    int num_local_experts = num_experts / WORLD_SIZE;


    int **device_num_tokens_ptrs = new int*[WORLD_SIZE];
    size_t num_tokens_size = WORLD_SIZE * num_local_experts * sizeof(int);

    int **device_num_recv_ptrs = new int*[WORLD_SIZE];
    size_t num_recv_size = WORLD_SIZE * num_local_experts * sizeof(int);

    int **device_num_x_buffer_ptrs = new int*[WORLD_SIZE];
    size_t num_x_buffer_size = WORLD_SIZE * num_local_experts * sizeof(int);
    for (int dev_idx = 0; dev_idx < WORLD_SIZE; ++dev_idx) {
        pglCudaMalloc<true>(WORLD_SIZE, device_ids, dev_idx, &device_num_tokens_ptrs[dev_idx], num_tokens_size);
        pglCudaMalloc<true>(WORLD_SIZE, device_ids, dev_idx, &device_num_recv_ptrs[dev_idx], num_recv_size);
        pglCudaMalloc<true>(WORLD_SIZE, device_ids, dev_idx, &device_num_x_buffer_ptrs[dev_idx], num_x_buffer_size);
    }
    g_buffer num_tokens_pgl(device_ids, device_num_tokens_ptrs, 1, 1, WORLD_SIZE, num_local_experts);
    g_buffer num_recv_pgl(device_ids, device_num_tokens_ptrs, 1, 1, WORLD_SIZE, num_recv_size);
    g_buffer x_buffer_pgl(device_ids, device_num_tokens_ptrs, 1, 1, WORLD_SIZE, num_x_buffer_size);
    
    int *d_dpx_data, *d_expertx_data;
    CUDACHECK(cudaMalloc(&d_dpx_data, max_num_tokens * HIDDEN_DIM_SIZE));
    CUDACHECK(cudaMalloc(&d_expertx_data, max_num_tokens * HIDDEN_DIM_SIZE));
    // This is more than needed => Can optimize
    g_gl dpX_gl{d_dpx_data, 1, 1, max_num_tokens, HIDDEN_DIM_SIZE};
    g_gl expertX_gl{d_expertx_data, 1, num_local_experts, max_num_tokens * WORLD_SIZE, HIDDEN_DIM_SIZE};

    int *d_out_num_tokens_per_expert;
    CUDACHECK(cudaMalloc(&d_out_num_tokens_per_expert, num_local_experts * sizeof(int)));

    int *d_indices;
    CUDACHECK(cudaMalloc(&d_indices, max_num_tokens * experts_per_token * sizeof(int)));

    int *d_num_tokens_per_dev;
    CUDACHECK(cudaMalloc(&d_num_tokens_per_dev, num_local_experts * WORLD_SIZE * sizeof(int)));
    
    globals G(
        expertX_gl,
        dpX_gl,
        num_tokens_pgl,
        num_recv_pgl,
        x_buffer_pgl,
        d_out_num_tokens_per_expert,
        d_indices,
        d_num_tokens_per_dev,
        max_num_tokens,
        num_experts,
        experts_per_token,
        128
    );

    int NUM_PROFILE_ITERS = 10; 
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_PROFILE_ITERS; ++i) {
        club.execute([&](int dev_idx) {
            dispatch_kernel<<<dimGrid, dimBlock, smem_size>>>(G, dev_idx);
            CUDACHECK(cudaDeviceSynchronize());
        });
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> elapsed = end - start;
    double avg_time = elapsed.count() / NUM_PROFILE_ITERS;
    printf("Finished dispatch kernel\n");
    printf("Avg time for dispatch kernel: %f microseconds\n", avg_time);

}

int main() {
    // num experts, num experts per token, token dim, block size (hidden dim scale size)
    run_benchmark(128, 256, 8, 7168, 128);
}


