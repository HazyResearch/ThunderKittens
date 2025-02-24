#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

// Modified layout to support LoRA with different ranks
template<int M_BLOCK, int N_BLOCK>
struct matmul_layout {
    using base_tile = st_bf<64, 64>;
    // A is now [M, K] (no batch dim), B is [B, K, r] with varying r
    using A_global_layout = gl<bf16, 1, 1, -1, -1, base_tile>; // [1, 1, M, K]
    using B_global_layout = gl<bf16, -1, 1, -1, -1, base_tile>; // [B, 1, K, r]
    using C_global_layout = gl<bf16, -1, 1, -1, -1, base_tile>; // [B, 1, M, r]
    using ranks_global_layout = gl<int, -1, 1, 1, 1>; // [B, 1, 1, 1]
    
    struct globals {
        A_global_layout A;
        B_global_layout B;
        C_global_layout C;
        ranks_global_layout ranks;
    };
    
    struct input_block {
        base_tile a[M_BLOCK];   // Tiles for A
        base_tile b[N_BLOCK];   // Tiles for B (padded to max rank)
    };
    
    struct finish_block {
        base_tile c[M_BLOCK][N_BLOCK];  // Result tiles
    };
    
    struct common_state {
        int batch;       // Current batch index
        int2 coord;      // Current coordinates in the grid
        int rank;        // Current batch's rank (actual, not padded)
    };
    
    struct consumer_state {
        rt_fl<16, 64> accum[N_BLOCK];  // Accumulator registers
    };
};

template<int _M_BLOCK=2, int _N_BLOCK=4, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = _M_BLOCK, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    using layout = matmul_layout<M_BLOCK, N_BLOCK>;
    using wide_tile = st_bf<64, 64*N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS=M_BLOCK*4, INPUT_PIPE_STAGES=4, PRODUCER_BARRIER_ARRIVALS=1;
    
    // Helper functions for grid dimensions
    template<bool PERSISTENT_GRID=false> __host__ static inline dim3 grid(int B, int M, int N, int K) {
        return dim3(PERSISTENT_GRID ? 132 : B*M*N/(M_BLOCK*N_BLOCK*layout::base_tile::num_elements));
    }
    
    // Common setup function
    __device__ static inline void common_setup(common_setup_args<layout> args) {
        int Rblocks = args.globals.C.rows / (M_BLOCK*64);
        int Cblocks = args.globals.C.cols / (N_BLOCK*64);
        int super_rows = (Rblocks/SUPER_M)*SUPER_M,
            final_rows = Rblocks - super_rows,
            super_repeat = SUPER_M*Cblocks;

        int blocks_per_batch = Rblocks*Cblocks;

        int global_block_id = args.task_iter*gridDim.x + blockIdx.x;
        int batch_idx = global_block_id / blocks_per_batch;
        int block_idx_in_batch = global_block_id % blocks_per_batch;
        
        if (batch_idx >= args.globals.B.batch) {
            args.num_iters = -1;
            return;
        }
        
        args.common.batch = batch_idx;
        
        // Get the rank for this batch
        if (warpid() == 0 && laneid() == 0) {
            args.common.rank = args.globals.ranks[{args.common.batch, 0, 0, 0}];
        }
        __syncwarp();
        
        // Share rank with all threads in the warp
        args.common.rank = __shfl_sync(0xffffffff, args.common.rank, 0);
        
        if (block_idx_in_batch < super_rows * Cblocks) {
            int x = SUPER_M*(block_idx_in_batch/super_repeat) + block_idx_in_batch%SUPER_M;
            args.common.coord = { x, (block_idx_in_batch%super_repeat)/SUPER_M };
        }
        else if (block_idx_in_batch < Rblocks*Cblocks) {
            int remainder_id = block_idx_in_batch - super_rows*Cblocks;
            args.common.coord = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
        }
        else {
            args.num_iters = -1;
            return;
        }
        
        args.num_iters = args.globals.A.cols/64;  // K dimension tile iterations
        
        // Assign work to warps
        int id = warpgroup::groupid() == NUM_CONSUMER_WARPS/4 ? 0 : warpgroup::groupid();
        args.common.coord = { args.common.coord.x*M_BLOCK + id, args.common.coord.y*N_BLOCK };
    }
    
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>(); // decrease registers for producers
        }
        
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                
                // Load tiles from A (common across all batches)
                for(int i = 0; i < M_BLOCK; i++) {
                    tma::load_async(args.input.a[i], args.globals.A,
                                   {0, 0, args.common.coord.x+i, args.iter}, args.inputs_arrived);
                }
                
                // Load tiles from B using the rank info
                // Only load up to the actual rank for this batch (padded with zeros)
                for(int i = 0; i < N_BLOCK; i++) {
                    if (args.common.coord.y + i < args.common.rank) {
                        tma::load_async(args.input.b[i], args.globals.B,
                                      {args.common.batch, 0, args.iter, args.common.coord.y+i}, 
                                      args.inputs_arrived);
                    } else {
                        // Zero out tiles beyond the rank
                        for (int r = 0; r < 64; r++) {
                            for (int c = 0; c < 64; c++) {
                                args.input.b[i][{r, c}] = __float2bfloat16(0.0f);
                            }
                        }
                        if (laneid() == 0) {
                            tma::signals::arrive(args.inputs_arrived);
                        }
                    }
                }
            }
        }
    };
    
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>(); // increase registers for consumers
            for (int n = 0; n < N_BLOCK; n++) {
                zero(args.state.accum[n]);
            }
        }
        
        __device__ static void compute(consumer_compute_args<layout> args) {
            for(int n = 0; n < N_BLOCK; n++) {
                // Only compute if this tile is within the actual rank for this batch
                if(args.common.coord.y + n < args.common.rank) {
                    warpgroup::mma_ABt(
                        args.state.accum[n],
                        args.input.a[warpgroup::groupid()],
                        args.input.b[n]
                    );
                }
            }
            warpgroup::mma_async_wait();
            if(laneid() == 0) arrive(args.inputs_finished);
        }
        
        __device__ static void finish(consumer_finish_args<layout> args) {
            for(int n = 0; n < N_BLOCK; n++) {
                warpgroup::store(args.finish.c[warpgroup::groupid()][n], args.state.accum[n]);
            }
            warpgroup::sync(warpgroup::groupid()+4);
            
            if(warpgroup::warpid() == 0) {
                for(int i = 0; i < N_BLOCK; i++) {
                    // Only store the results if this tile is within the actual rank
                    if(args.common.coord.y + i < args.common.rank) {
                        tma::store_async(args.globals.C, args.finish.c[warpgroup::groupid()][i],
                                      {args.common.batch, 0, args.common.coord.x, args.common.coord.y+i});
                        tma::store_async_read_wait();
                    }
                }
            }

            // Zero the accumulators
            for(int n = 0; n < N_BLOCK; n++) {
                zero(args.state.accum[n]);
            }
            if(laneid() == 0) arrive(args.finish_finished);
        }
    };
};

// Function to run the multi-LoRA matmul from PyTorch
torch::Tensor multi_lora_matmul(torch::Tensor A, torch::Tensor B, torch::Tensor ranks) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(ranks);
    
    // A shape: [M, K]
    // B shape: [B, K, N] (padded to max rank)
    // ranks shape: [B]
    
    TORCH_CHECK(A.dim() == 2, "A must be 2-dimensional [M, K]");
    TORCH_CHECK(B.dim() == 3, "B must be 3-dimensional [B, K, N]");
    TORCH_CHECK(ranks.dim() == 1, "ranks must be 1-dimensional [B]");
    TORCH_CHECK(A.size(1) == B.size(1), "Inner dimensions must match");
    TORCH_CHECK(B.size(0) == ranks.size(0), "Batch size mismatch between B and ranks");
    
    uint M = A.size(0), K = A.size(1);
    uint B_batch = B.size(0), max_rank = B.size(2);
    
    // Create output tensor [B, M, max_rank]
    torch::Tensor C = torch::zeros({B_batch, M, max_rank}, A.options());
    
    // M_BLOCK, N_BLOCK, SUPER_M
    using mmt = matmul_template<2, 4, 8>;
    
    using A_global_layout = typename mmt::layout::A_global_layout;
    using B_global_layout = typename mmt::layout::B_global_layout;
    using C_global_layout = typename mmt::layout::C_global_layout;
    using ranks_global_layout = typename mmt::layout::ranks_global_layout;
    using globals = typename mmt::layout::globals;
    
    // Setup global layouts
    A_global_layout Ag = {reinterpret_cast<bf16*>(A.data_ptr<c10::BFloat16>()), 1, nullptr, M, K};
    B_global_layout Bg = {reinterpret_cast<bf16*>(B.data_ptr<c10::BFloat16>()), B_batch, nullptr, K, max_rank};
    C_global_layout Cg = {reinterpret_cast<bf16*>(C.data_ptr<c10::BFloat16>()), B_batch, nullptr, M, max_rank};
    
    // Convert ranks tensor to int if needed and set up layout
    torch::Tensor ranks_int;
    if (ranks.scalar_type() != torch::kInt) {
        ranks_int = ranks.to(torch::kInt);
    } else {
        ranks_int = ranks;
    }
    ranks_global_layout Rg = {reinterpret_cast<int*>(ranks_int.data_ptr<int>()), B_batch, nullptr, 1, 1};
    
    globals G{Ag, Bg, Cg, Rg};
    
    dim3 grid(mmt::grid(B_batch, M, max_rank, K));
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<mmt>);
    cudaFuncSetAttribute(prototype::lcf::kernel<mmt>, cudaFuncAttributeMaxDynamicSharedMemorySize, MAX_SHARED_MEMORY-1024);
    prototype::lcf::kernel<mmt><<<grid, block, MAX_SHARED_MEMORY-1024>>>(G);
    
    return C;
}
