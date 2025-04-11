#include "kittens.cuh"
#include <cooperative_groups.h>

using namespace kittens;
namespace cg = cooperative_groups;

constexpr int WORLD_SIZE = 8;
constexpr int DP_SIZE = 1;
constexpr int NUM_EXPERTS = 256;

constexpr int HIDDEN_DIM_SIZE = 7168;
constexpr int HIDDEN_DIM_SCALE_SIZE = HIDDEN_DIM_SIZE / 128; // HIDDEN_DIM_SIZE / BLOCK_SIZE

constexpr int NUM_WARPS = 10;

using g_pgl = pgl<gl<uint32_t, -1, -1, -1, -1>>;
using g_gl = gl<uint32_t, -1, -1, -1, -1>;
using g_buffer = pgl<gl<uint32_t, -1, -1, -1, -1>>;
using s_vec = sv<uint32_t, NUM_EXPERTS>;

/*
Grid size is 132
*/

struct globals {
    g_pgl outNumTokensPerExpert;

    g_pgl expertX;
    g_pgl expertXScale;

    g_pgl dpX;
    g_pgl dpXScale;

    
    g_buffer numTokensBuffer;
    g_buffer numRecvBuffer;
    g_buffer xBufferIn;
    g_buffer xBufferOut;
    
    g_gl numTokensPerDP; 
    g_gl sourceExpert;
    g_gl sourceIndex;
    g_gl sourceOffset;
    g_gl sourceGroup;
    
    // inidices is m * numExpertsPerToken
    // Reasonable to load into a rt
    uint32_t *indices;

    size_t maxNumTokens;
    size_t numExperts;
    size_t numExpertsPerToken;

    int numTokens; // num tokens cur device needs to send
    int dev_idx;
};

__global__ void dispatch_kernel(const __grid_constant__ globals g) {
    const unsigned num_local_experts = NUM_EXPERTS / WORLD_SIZE;
    using everyone = group<NUM_WARPS>;

    /*
    Send Phase
    */
    extern __shared__ kittens::alignment_dummy __shm[]; 
    kittens::shared_allocator al((int*)&__shm[0]);
    s_vec &s_v = al.allocate<s_vec>();
    everyoe::zero(s_v);
    everyone::sync();

    if (kittens::warpid == NUM_WARPS - 1) {
        /*
        In example num of experts responsible for is 32

        Starting g_expert_idx is between 0 and 132:
            - Block Idx 0 handles expert 0, 132, 
            ...
            - Block Idx 124 handles expert 124, 256
        */
        for (size_t g_expert_idx = blockIdx.x; g_expert_idx < NUM_EXPERTS; g_expert_idx += gridDim.x) {
            int dst_gpu = g_expert_idx / num_local_experts;
            int dst_expert_idx = g_expert_idx % num_local_experts;

            size_t count = 0;
            
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
            for (int i = threadIdx.x; i < g.numTokens * g.numExpertsPerToken; i += WARP_THREADS) {
                int expert = __ldg(&g.indices[i]);
                if (expert == dst_expert_idx) count++;
            }

        }
        

    } else {


    }

    /*
    Receive Phase
    */
    for (size_t expert = blockIdx.x * blockDim.x + threadIdx.x;
         expert < num_local_experts; expert += blockDim.x * gridDim.x) {
        
        // Stall until numTokensBuffer is updated 
        // nvshmem_uint64_wait_until(&numTokensBuffer[slot], NVSHMEM_CMP_NE, 0);

        size_t numTokens  = numTokensBuffer[expert] - 1;

        // Stall here until we know that all tokens have been received
        // nvshmem_uint64_wait_until(&numRecvBuffer[slot], NVSHMEM_CMP_EQ, numTokens);
        
        numTokensPerDP[expert] = numTokens;



    }
    
    cg::this_grid().sync();




}