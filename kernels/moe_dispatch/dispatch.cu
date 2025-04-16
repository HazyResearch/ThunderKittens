#include "kittens.cuh"
#include <random>

using namespace kittens;

constexpr int WORLD_SIZE = 8;
// constexpr int WORLD_SIZE = 2;
constexpr int DP_SIZE = 1;
constexpr int NUM_EXPERTS = 256;
// constexpr int NUM_EXPERTS = 16;

constexpr int HIDDEN_DIM_SIZE = 7168;
// Force HIDDEN_DIM_SCALE_SIZE to be multiple of 16
constexpr int HIDDEN_DIM_SCALE_SIZE = 64; // HIDDEN_DIM_SIZE / BLOCK_SIZE => HIDDEN_DIM_SIZE / 128
constexpr int TOKEN_DIM = HIDDEN_DIM_SIZE + HIDDEN_DIM_SCALE_SIZE;

constexpr int NUM_WARPS = 10;
// constexpr int NUM_WARPS = 2;

using g_pgl = pgl<gl<int, -1, -1, -1, -1>, WORLD_SIZE, true, false>;
using g_gl = gl<int, -1, -1, -1, -1>;

using g_buffer = pgl<gl<int, -1, -1, -1, -1>, WORLD_SIZE, true, false>;
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
    
    
    int *outNumTokensPerExpert; // Size is numExperts / numPEs 
    // inidices is m * numExpertsPerToken
    // Reasonable to load into a r
    int *indices;
    int *numTokensPerDP; // make non pgl

    // uint32_t *sourceExpert;
    // uint32_t *sourceIndex;
    // uint32_t *sourceOffset;
    // uint32_t *sourceGroup;
    
    int maxNumTokens;
    int numExperts;
    int numExpertsPerToken;

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

    // if (blockIdx.x == 1 && warpid() == 1 && laneid() == 0 && dev_idx == 1) {
    //     for (int i = 0; i < num_local_experts; ++i) {
    //         printf("%d ", g.numTokensBuffer[dev_idx].raw_ptr[i]);
    //     }
    //     printf("\n");
    // }
    // __syncthreads();

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
                int expert = __ldg(&g.indices[i]);
                if (expert == g_expert_idx) {
                    count++;
                }
                
            }
            // printf("Count for expert %d on device %d: %d\n", g_expert_idx, dev_idx, count);
            
            count += __shfl_xor_sync(0xffffffff, count, 16);
            count += __shfl_xor_sync(0xffffffff, count, 8);
            count += __shfl_xor_sync(0xffffffff, count, 4);
            count += __shfl_xor_sync(0xffffffff, count, 2);
            count += __shfl_xor_sync(0xffffffff, count, 1);
            
            if (laneid() == 0) {
                // printf("Count for expert %d on device %d: %d\n", g_expert_idx, dev_idx, count);
                // g.numTokensBuffer.mc_vas[g.dev_idx][dst_expert_idx] = count + 1;
                // Could be faster to just store directly instead of multimem here => test difference
                // printf("Device %d\n", dev_idx);
                // printf("Count for expert: %d\n", count);
                
                // Check if value is correct
                // printf("Before count for expert in numTokensBuffer: %d\n", g.numTokensBuffer[dev_idx].raw_ptr[dst_expert_idx]); 
                // atomicExch(&(g.numTokensBuffer[dst_gpu].raw_ptr[dst_expert_idx * WORLD_SIZE + dev_idx]), count + 1);
                g.numTokensBuffer[dst_gpu].raw_ptr[dst_expert_idx * WORLD_SIZE + dev_idx] = count + 1;

                // atomicAdd(&(g.numTokensBuffer[dst_gpu].raw_ptr[dst_expert_idx * WORLD_SIZE + dev_idx]), count + 1);
                // atomicAdd(&(g.numTokensBuffer[dev_idx].raw_ptr[dst_expert_idx]), 1);
                // __threadfence_system();
                // asm volatile ("{multimem.red.release.sys.global.add.u32 [%0], %1;}" 
                //   :: "l"(&(g.numTokensBuffer.mc_vas[dev_idx][dst_expert_idx])), 
                //      "r"(count + 1) : "memory");
                // printf("Count for expert in numTokensBuffer: %d\n", g.numTokensBuffer[dev_idx].raw_ptr[dst_expert_idx]);
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
                // send_group::load(tok_dst_data, g.dpX, {i}); // This is errors
                // send_group::sync(0);

                for (int j = 0; j < g.numExpertsPerToken; ++j) {
                    int dst_expert = __ldg(&g.indices[i * g.numExpertsPerToken + j]);
                    int dst_gpu = dst_expert / num_local_experts;
                    int dst_expert_idx = dst_expert % num_local_experts;
                    
                    // Get the number of tokens that have currently been sent to the expert
                    int index = tokenIndex.data[dst_expert];

                    send_group::store(g.xBufferOut[dev_idx], tok_dst_data, 
                        {dev_idx, dst_expert_idx, index, 0});
                    if (warpid() == 0 && laneid() == 0) {
                        atomicAdd(&(g.numRecvBuffer[dst_gpu].raw_ptr[dst_expert_idx * WORLD_SIZE + dev_idx]), 1);
                    }
                }

                // Replicate token count across all blocks
                if (warpid() == 0 && laneid() < g.numExpertsPerToken) {
                    int dst_expert = __ldg(&g.indices[i * g.numExpertsPerToken + laneid()]);
                    tokenIndex.data[dst_expert]++;
                }
            }


        }
    }
    // __threadfence_system();

    // __nanosleep(1000);
    // __nanosleep(1000);
    // __nanosleep(1000);
    // __nanosleep(1000);
    // __nanosleep(1000);
    // __nanosleep(1000);

    // if (blockIdx.x == 1 && warpid() == 1 && laneid() == 0 && dev_idx == 1) {
    //     for (int i = 0; i < num_local_experts; ++i) {
    //         printf("%d ", g.numTokensBuffer[dev_idx].raw_ptr[i]);
    //     }
    //     printf("\n");
    // }

    /*
    Receive Phase
    */
    /*
    first 32 threads of device stall untill device receives all tokens expected
    */
    // for (int expert_and_group = blockIdx.x * blockDim.x + threadIdx.x; expert_and_group < num_local_experts * WORLD_SIZE; expert_and_group += gridDim.x * blockDim.x) {
    //     int src_gpu = expert_and_group % WORLD_SIZE;
    //     int src_local_expert = expert_and_group / WORLD_SIZE;
    //     int slot = src_local_expert * WORLD_SIZE + src_gpu;
    //     int global_expert = src_local_expert + (dev_idx * num_local_experts);

    //     while (((volatile int*)g.numTokensBuffer[dev_idx].raw_ptr)[slot] == 0) {
    //     }
    //     size_t numTokens = g.numTokensBuffer[dev_idx].raw_ptr[slot] - 1;
    //     // Stall here until we know that all tokens have been received
    //     while (*(volatile int*)&g.numRecvBuffer[dev_idx].raw_ptr[slot] < numTokens) {
    
    //     }
    //     atomicAdd(&g.outNumTokensPerExpert[src_local_expert], numTokens);
        
    //     g.numTokensBuffer[dev_idx].raw_ptr[slot] = 0;
    //     g.numRecvBuffer[dev_idx].raw_ptr[slot] = 0;
        
    // }
    if (warpid() == NUM_WARPS - 1 && laneid() == 0) {
        for (size_t expert_gpu_group = blockIdx.x; expert_gpu_group < num_local_experts * WORLD_SIZE; expert_gpu_group += gridDim.x) {
            int src_gpu = expert_gpu_group % WORLD_SIZE;
            int src_local_expert = expert_gpu_group / WORLD_SIZE;
            int slot = src_local_expert * WORLD_SIZE + src_gpu;
            int global_expert = src_local_expert + (dev_idx * num_local_experts);

            // Stall until numTokensBuffer is updated 
            // while (g.numTokensBuffer[dev_idx].raw_ptr[slot] == 0) {
            //     // printf("Waiting for numTokensBuffer: %d\n", g.numTokensBuffer[dev_idx].raw_ptr[src_local_expert]);
            //     printf("Waiting for token counts expert: %d\n", global_expert);
            // }
            while (((volatile int*)g.numTokensBuffer[dev_idx].raw_ptr)[slot] == 0) {
            }
            // while (atomicAdd(&(g.numTokensBuffer[dev_idx].raw_ptr[slot]), 0) == 0) {
            // }
            int numTokens = g.numTokensBuffer[dev_idx].raw_ptr[slot] - 1;
            // printf("Finished waiting for expert: %d\n", expert);
            // printf("Num Tokens: %d\n", numTokens);
            // printf("Expert %d has %d tokens\n", expert, dev_idx, numTokens);
            // // Stall here until we know that all tokens have been received
            while (*(volatile int*)&g.numRecvBuffer[dev_idx].raw_ptr[slot] < numTokens) {
                
            }
            
            // if (slot >= num_local_experts * WORLD_SIZE) {
            //     printf("Slot %d is out of bounds\n", slot);
            // }
            // g.numTokensPerDP[slot] = numTokens;
            atomicAdd(&g.outNumTokensPerExpert[src_local_expert], numTokens);
            
            g.numTokensBuffer[dev_idx].raw_ptr[slot] = 0;
            g.numRecvBuffer[dev_idx].raw_ptr[slot] = 0;
        }
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
            
            recv_group::load(tok_data, g.xBufferOut[dev_idx], {group, j});
            recv_group::sync(0);
            recv_group::store(g.expertX, tok_data, {expert, loc});
    }
}

template <typename T> __host__ T ceil_div(T x, T y) { return (x + y - 1) / y; }

void inner_run(int* d_dpx_data, int* d_expertx_data, g_buffer num_tokens_pgl, g_buffer num_recv_pgl, g_buffer x_buffer_pgl, 
               int *out_num_tokens_per_expert, int *indices, int *num_tokens_per_dev,
               int max_num_tokens, int num_experts, int experts_per_token, int dev_idx, dim3 dimGrid, dim3 dimBlock) {
    int num_local_experts = num_experts / WORLD_SIZE;
    g_gl dpX_gl{d_dpx_data, 1, 1, max_num_tokens, HIDDEN_DIM_SIZE};
    g_gl expertX_gl{d_expertx_data, 1, num_local_experts, max_num_tokens * WORLD_SIZE, HIDDEN_DIM_SIZE};

    // printf("num_experts" ": %d, experts_per_token: %d, max_num_tokens: %d\n", num_experts, experts_per_token, max_num_tokens);

    globals G{
        expertX_gl,
        dpX_gl,
        num_tokens_pgl,
        num_recv_pgl,
        x_buffer_pgl,
        out_num_tokens_per_expert,
        indices,
        num_tokens_per_dev,
        max_num_tokens,
        num_experts,
        experts_per_token,
        max_num_tokens
    };

    dispatch_kernel<<<dimGrid, dimBlock, MAX_SHARED_MEMORY - 1024>>>(G, dev_idx);
}

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

    int num_local_experts = NUM_EXPERTS / WORLD_SIZE;
    // Initialize host side data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, NUM_EXPERTS - 1); // Range [0, 128]
    
    // Fill the array with random values
    int h_indices[WORLD_SIZE][max_num_tokens * experts_per_token];
    for (int dev_idx = 0; dev_idx < WORLD_SIZE; ++dev_idx) {
        for (int i = 0; i < max_num_tokens * experts_per_token; ++i) {
            h_indices[dev_idx][i] = dist(gen);
            // h_indices[dev_idx][i] = 1;
            if (dev_idx == 0) {
                printf("%d ", h_indices[dev_idx][i]);
            }
        }
    }
    printf("\n");

    int pgl_data[WORLD_SIZE * num_local_experts];
    for (int i = 0; i < WORLD_SIZE * num_local_experts; ++i) {
        pgl_data[i] = 0;
    }

    // Initialize device side code
    int **device_num_tokens_ptrs = new int*[WORLD_SIZE];
    int **device_num_recv_ptrs = new int*[WORLD_SIZE];
    int **device_num_x_buffer_ptrs = new int*[WORLD_SIZE];
    
    size_t num_tokens_size = WORLD_SIZE * num_local_experts * sizeof(int);
    size_t num_recv_size = WORLD_SIZE * num_local_experts * sizeof(int);
    size_t num_x_buffer_size = WORLD_SIZE * num_local_experts * max_num_tokens * HIDDEN_DIM_SIZE * sizeof(int);

    for (int dev_idx = 0; dev_idx < WORLD_SIZE; ++dev_idx) {
        pglCudaMalloc<true>(WORLD_SIZE, device_ids, dev_idx, &device_num_tokens_ptrs[dev_idx], num_tokens_size);
        pglCudaMalloc<true>(WORLD_SIZE, device_ids, dev_idx, &device_num_recv_ptrs[dev_idx], num_recv_size);
        pglCudaMalloc<true>(WORLD_SIZE, device_ids, dev_idx, &device_num_x_buffer_ptrs[dev_idx], num_x_buffer_size);
    }

    g_buffer num_tokens_pgl(device_ids, device_num_tokens_ptrs, 1, 1, WORLD_SIZE, num_local_experts);
    g_buffer num_recv_pgl(device_ids, device_num_recv_ptrs, 1, 1, WORLD_SIZE, num_local_experts);
    g_buffer x_buffer_pgl(device_ids, device_num_x_buffer_ptrs, WORLD_SIZE, num_local_experts, max_num_tokens, HIDDEN_DIM_SIZE);
    
    int *d_dpx_data[WORLD_SIZE], *d_expertx_data[WORLD_SIZE], *d_out_num_tokens_per_expert[WORLD_SIZE],
        *d_indices[WORLD_SIZE], *d_num_tokens_per_dev[WORLD_SIZE];
    
    // Create CUDA events for timing
    cudaEvent_t start_events[WORLD_SIZE], stop_events[WORLD_SIZE];
    
    for (int dev_idx = 0; dev_idx < WORLD_SIZE; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaMalloc(&d_dpx_data[dev_idx], max_num_tokens * HIDDEN_DIM_SIZE));
        CUDACHECK(cudaMalloc(&d_expertx_data[dev_idx], max_num_tokens * HIDDEN_DIM_SIZE));
        CUDACHECK(cudaMalloc(&d_out_num_tokens_per_expert[dev_idx], num_local_experts * sizeof(int)));
        CUDACHECK(cudaMalloc(&d_indices[dev_idx], max_num_tokens * experts_per_token * sizeof(int)));
        CUDACHECK(cudaMalloc(&d_num_tokens_per_dev[dev_idx], num_local_experts * WORLD_SIZE * sizeof(int)));
        
        // Create CUDA events with timing
        CUDACHECK(cudaEventCreate(&start_events[dev_idx]));
        CUDACHECK(cudaEventCreate(&stop_events[dev_idx]));
    }

    for (int dev_idx = 0; dev_idx < WORLD_SIZE; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaMemcpy(d_indices[dev_idx], h_indices[dev_idx], 
            max_num_tokens * experts_per_token * sizeof(int), cudaMemcpyHostToDevice));

        CUDACHECK(cudaMemcpy(device_num_tokens_ptrs[dev_idx], pgl_data, 
            num_tokens_size, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(device_num_recv_ptrs[dev_idx], pgl_data, 
            num_recv_size, cudaMemcpyHostToDevice));
            
        CUDACHECK(cudaMemset(d_out_num_tokens_per_expert[dev_idx], 0,
            num_local_experts * sizeof(int)));
        CUDACHECK(cudaMemset(d_num_tokens_per_dev[dev_idx], 0,
            num_local_experts * WORLD_SIZE * sizeof(int)));
    }

    const unsigned numBlocks = std::min( // min( max((256 / 10), (128 * 8)), 132) = 132
        std::max(
            ceil_div(NUM_EXPERTS, NUM_WARPS), (max_num_tokens * experts_per_token)
        ),
        132
    );
    dim3 dimGrid(numBlocks, 1, 1);
    dim3 dimBlock(NUM_WARPS * 32, 1, 1);

    int NUM_WARMUP_ITERS = 5;
    // Warmup iterations
    for (int i = 0; i < NUM_WARMUP_ITERS; ++i) {
        club.execute([&](int dev_idx) {
            inner_run(d_dpx_data[dev_idx], d_expertx_data[dev_idx],
                num_tokens_pgl, num_recv_pgl, x_buffer_pgl, d_out_num_tokens_per_expert[dev_idx], 
                d_indices[dev_idx], d_num_tokens_per_dev[dev_idx], max_num_tokens, NUM_EXPERTS, 
                experts_per_token, dev_idx, dimGrid, dimBlock);
            CUDACHECK(cudaDeviceSynchronize());
        });
    }
    
    int NUM_PROFILE_ITERS = 20;
    float total_time = 0.0f;
    
    // Timing iterations
    for (int i = 0; i < NUM_PROFILE_ITERS; ++i) {
        club.execute([&](int dev_idx) {
            CUDACHECK(cudaEventRecord(start_events[dev_idx], 0));
            inner_run(d_dpx_data[dev_idx], d_expertx_data[dev_idx],
                num_tokens_pgl, num_recv_pgl, x_buffer_pgl, d_out_num_tokens_per_expert[dev_idx], 
                d_indices[dev_idx], d_num_tokens_per_dev[dev_idx], max_num_tokens, NUM_EXPERTS, 
                experts_per_token, dev_idx, dimGrid, dimBlock);                
            CUDACHECK(cudaEventRecord(stop_events[dev_idx], 0));
            
            CUDACHECK(cudaDeviceSynchronize());
        });
        
        float elapsed_time = 0.0f;
        for (int dev_idx = 0; dev_idx < WORLD_SIZE; ++dev_idx) {
            CUDACHECK(cudaSetDevice(dev_idx));
            float ms = 0.0f;
            CUDACHECK(cudaEventElapsedTime(&ms, start_events[dev_idx], stop_events[dev_idx]));
            elapsed_time += ms;
        }
        elapsed_time /= WORLD_SIZE;
        total_time += elapsed_time;
    }
    
    // Calculate average kernel execution time
    float avg_time = total_time / NUM_PROFILE_ITERS;
    printf("Avg kernel execution time: %f microseconds\n", avg_time * 1000.0f);

    // Cleanup
    for (int dev_idx = 0; dev_idx < WORLD_SIZE; ++dev_idx) {
        CUDACHECK(cudaSetDevice(dev_idx));
        CUDACHECK(cudaFree(d_dpx_data[dev_idx]));
        CUDACHECK(cudaFree(d_expertx_data[dev_idx]));
        CUDACHECK(cudaFree(d_out_num_tokens_per_expert[dev_idx]));
        CUDACHECK(cudaFree(d_indices[dev_idx]));
        CUDACHECK(cudaFree(d_num_tokens_per_dev[dev_idx]));
        
        // Destroy CUDA events
        CUDACHECK(cudaEventDestroy(start_events[dev_idx]));
        CUDACHECK(cudaEventDestroy(stop_events[dev_idx]));
    }
    for (int dev_idx = 0; dev_idx < WORLD_SIZE; ++dev_idx) {
        pglCudaFree(dev_idx, device_num_tokens_ptrs[dev_idx], num_tokens_size);
        pglCudaFree(dev_idx, device_num_recv_ptrs[dev_idx], num_recv_size);
        pglCudaFree(dev_idx, device_num_x_buffer_ptrs[dev_idx], num_x_buffer_size);
    }
    delete[] device_num_tokens_ptrs;
    delete[] device_num_recv_ptrs;
    delete[] device_num_x_buffer_ptrs;

    pglFree(num_tokens_pgl);
    pglFree(num_recv_pgl);
    pglFree(x_buffer_pgl);
}
int main() {
    // max_num_tokens, num_experts, num experts per token, token dim, block size (hidden dim scale size)
    // run_benchmark(128, 16, 8, 7168, 128);
    // run_benchmark(128, 16, 1, 7168, 128);
    run_benchmark(128, 256, 8, 7168, 128);
}


