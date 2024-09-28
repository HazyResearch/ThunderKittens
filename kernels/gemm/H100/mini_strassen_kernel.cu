#include "kittens.cuh"

using namespace kittens;

using tile = st_hf<64, 64>;
constexpr int M_BLOCK = 1, N_BLOCK = 2, M_TILE = 2*M_BLOCK, N_TILE = 2*N_BLOCK;
constexpr int SUPER_M = 12;
constexpr int NUM_CONSUMER_WARPS = 8;
constexpr int NUM_THREADS = (NUM_CONSUMER_WARPS+4)*32;
using  input_layout   = gl<half, 1, 7, -1, -1, tile>;
using  output_layout  = gl<half, 1, 1, -1, -1, tile>;
struct globals        { input_layout A, B; output_layout C; };

__device__ inline void load(tile (&a_s)[7],
                            tile (&b_s)[2][7],
                            const input_layout &A,
                            const input_layout &B,
                            barrier (&inputs_arrived)[7],
                            barrier (&inputs_finished)[7],
                            const coord &block_coords, int iter) {
    int phase = (iter%2)^1;
    #pragma unroll
    for(int j = 0; j < 7; j++) {
        wait(inputs_finished[j], phase);
        tma::expect_bytes(inputs_arrived[j], sizeof(tile)*(N_BLOCK+1));
        tma::load_async(a_s[j], A, {j, block_coords.r, iter}, inputs_arrived[j]);
        #pragma unroll
        for(int i = 0; i < N_BLOCK; i++) {
            tma::load_async(b_s[i][j], B, {j, iter, block_coords.c+i}, inputs_arrived[j]);
        }
    }
}

__device__ inline void work(rt_hf<16, 64> (&c_r)[2][2],
                            wgmma::descriptor<tile, 0> (&a_s)[7],
                            wgmma::descriptor<tile, 1> (&b_s)[7],
                            barrier (&inputs_arrived)[7],
                            barrier (&inputs_finished)[7],
                            int iter) {
    int phase = iter%2;
    rt_hf<16, 64> accum[2];
    wait(inputs_arrived[0], phase); // M1 arrives
    warpgroup::mm_AB(accum[0], a_s[0], b_s[0]); // M1
    wait(inputs_arrived[1], phase); // M2 arrives
    warpgroup::mm_AB(accum[1], a_s[1], b_s[1]); // M2
    warpgroup::mma_async_wait<1>();
    arrive(inputs_finished[0]); // finished M1
    add(c_r[0][0], c_r[0][0], accum[0]); // M1
    add(c_r[1][1], c_r[1][1], accum[0]);
    wait(inputs_arrived[2], phase); // M3 arrives
    warpgroup::mm_AB(accum[0], a_s[2], b_s[2]); // M3
    warpgroup::mma_async_wait<1>();
    arrive(inputs_finished[1]); // finished M2
    add(c_r[1][0], c_r[1][0], accum[1]); // M2
    sub(c_r[1][1], c_r[1][1], accum[1]);
    wait(inputs_arrived[3], phase); // M4 arrives
    warpgroup::mm_AB(accum[1], a_s[3], b_s[3]); // M4
    warpgroup::mma_async_wait<1>();
    arrive(inputs_finished[2]); // finished M3
    add(c_r[0][1], c_r[0][1], accum[0]); // M3
    add(c_r[1][1], c_r[1][1], accum[0]);
    wait(inputs_arrived[4], phase); // M5 arrives
    warpgroup::mm_AB(accum[0], a_s[4], b_s[4]); // M5
    warpgroup::mma_async_wait<1>();
    arrive(inputs_finished[3]); // finished M4
    add(c_r[0][0], c_r[0][0], accum[1]); // M4
    add(c_r[1][0], c_r[1][0], accum[1]);
    wait(inputs_arrived[5], phase); // M6 arrives
    warpgroup::mma_AB(c_r[1][1], a_s[5], b_s[5]); // M6, accumulate into C[1][1]
    warpgroup::mma_async_wait<1>();
    arrive(inputs_finished[4]); // finished M5
    sub(c_r[0][0], c_r[0][0], accum[0]); // M5
    add(c_r[0][1], c_r[0][1], accum[0]);
    wait(inputs_arrived[6], phase); // M7 arrives
    warpgroup::mma_AB(c_r[0][0], a_s[6], b_s[6]); // M7, accumulate into C[0][0]
    warpgroup::mma_async_wait<1>();
    arrive(inputs_finished[5]); // finished M6
    warpgroup::mma_async_wait();
    arrive(inputs_finished[6]); // finished M7
}
// __device__ inline void work(rt_hf<16, 64> (&c_r)[2][2],
//                             tile (&a_s)[7],
//                             tile (&b_s)[7],
//                             barrier (&inputs_arrived)[7],
//                             barrier (&inputs_finished)[7],
//                             int iter) {
//     int phase = iter%2;
//     rt_hf<16, 64> accum[2];
//     wait(inputs_arrived[0], phase); // M1 arrives
//     warpgroup::mm_AB(accum[0], a_s[0], b_s[0]); // M1
//     wait(inputs_arrived[1], phase); // M2 arrives
//     warpgroup::mm_AB(accum[1], a_s[1], b_s[1]); // M2
//     warpgroup::mma_async_wait<1>();
//     arrive(inputs_finished[0]); // finished M1
//     add(c_r[0][0], c_r[0][0], accum[0]); // M1
//     add(c_r[1][1], c_r[1][1], accum[0]);
//     wait(inputs_arrived[2], phase); // M3 arrives
//     warpgroup::mm_AB(accum[0], a_s[2], b_s[2]); // M3
//     warpgroup::mma_async_wait<1>();
//     arrive(inputs_finished[1]); // finished M2
//     add(c_r[1][0], c_r[1][0], accum[1]); // M2
//     sub(c_r[1][1], c_r[1][1], accum[1]);
//     wait(inputs_arrived[3], phase); // M4 arrives
//     warpgroup::mm_AB(accum[1], a_s[3], b_s[3]); // M4
//     warpgroup::mma_async_wait<1>();
//     arrive(inputs_finished[2]); // finished M3
//     add(c_r[0][1], c_r[0][1], accum[0]); // M3
//     add(c_r[1][1], c_r[1][1], accum[0]);
//     wait(inputs_arrived[4], phase); // M5 arrives
//     warpgroup::mm_AB(accum[0], a_s[4], b_s[4]); // M5
//     warpgroup::mma_async_wait<1>();
//     arrive(inputs_finished[3]); // finished M4
//     add(c_r[0][0], c_r[0][0], accum[1]); // M4
//     add(c_r[1][0], c_r[1][0], accum[1]);
//     wait(inputs_arrived[5], phase); // M6 arrives
//     warpgroup::mma_AB(c_r[1][1], a_s[5], b_s[5]); // M6, accumulate into C[1][1]
//     warpgroup::mma_async_wait<1>();
//     arrive(inputs_finished[4]); // finished M5
//     sub(c_r[0][0], c_r[0][0], accum[0]); // M5
//     add(c_r[0][1], c_r[0][1], accum[0]);
//     wait(inputs_arrived[6], phase); // M7 arrives
//     warpgroup::mma_AB(c_r[0][0], a_s[6], b_s[6]); // M7, accumulate into C[0][0]
//     warpgroup::mma_async_wait<1>();
//     arrive(inputs_finished[5]); // finished M6
//     warpgroup::mma_async_wait();
//     arrive(inputs_finished[6]); // finished M7
// }

__device__ static void finish(const output_layout &C,
                              tile (&c_s)[2][2],
                              rt_hf<16, 64> (&c_r)[2][2],
                              const coord &tile_coords) {
    for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
        warpgroup::store(c_s[i][j], c_r[i][j]);
        warpgroup::sync();
        if(warpgroup::warpid() == 0) {
            tma::store_async(C, c_s[i][j],
                            {tile_coords.r+i, tile_coords.c+j});
        }
    }
}

__device__ static inline int num_iters(const globals &g) { return g.A.cols / 64; }
__host__ static inline dim3 get_grid(int M, int N, int K) {
    return dim3(M*N/(M_TILE*N_TILE*tile::num_elements));
}
__device__ static inline void get_coords(kittens::coord &coords, const globals &g, int id) {
    int Rblocks = g.C.rows / (M_TILE*64), Cblocks = g.C.cols / (N_TILE*64);
    int super_rows = (Rblocks/SUPER_M)*SUPER_M,
    final_rows = Rblocks - super_rows,
    super_repeat = SUPER_M*Cblocks;
    if (blockIdx.x < super_rows * Cblocks)
        coords = { SUPER_M*(blockIdx.x/super_repeat) + blockIdx.x%SUPER_M,
                    (blockIdx.x%super_repeat)/SUPER_M };
    else {
        int remainder_id = blockIdx.x - super_rows*Cblocks;
        coords = { super_rows + (remainder_id%final_rows), remainder_id/final_rows };
    }
    if(id == -1) coords = { coords.r*M_BLOCK, coords.c*N_BLOCK };
    else 		 coords = { coords.r*M_TILE, coords.c*N_TILE + id*2 };
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void mini_matmul(const __grid_constant__ globals g) {

    extern __shared__ int __shm[];
    shared_allocator alloc(&__shm[0]); // allocate shared memory
    tile (&a_s)[7] = alloc.allocate<tile, 7>();
    tile (&b_s)[2][7] = alloc.allocate<tile, 2, 7>();
    tile (&c_s)[2][2][2] = reinterpret_cast<tile(&)[2][2][2]>(a_s);

    // Initialize barriers. This is constant for all two-stage producer-consumer kernels.
    __shared__ kittens::barrier inputs_arrived[7], inputs_finished[7];
    if (warpid() == 0) { // a single warp (in fact a single thread) does these.
        for(int i = 0; i < 7; i++) {
            init_barrier(inputs_arrived[i], 1, 0); // needs to wait on each producer warp
            init_barrier(inputs_finished[i], 8, 0); // needs to wait on one thread from each consumer warp
        }
    }
    int iters = num_iters(g);

    __syncthreads(); // all warps must arrive here, confirming barrier initialization is visible to all threads.

    if(warpid() >= NUM_CONSUMER_WARPS) { // last warpgroup is a producer
        warpgroup::producer_registers(); // decrease registers for producers
        coord block_coords;
        get_coords(block_coords, g, -1);
        for(int load_iter = 0; load_iter < iters; load_iter++) {
            if(warpgroup::warpid() == 0) {
                load(a_s, b_s, g.A, g.B, inputs_arrived, inputs_finished, block_coords, load_iter);
            }
        }
    }
    else { // other warpgroups are consumers
        warpgroup::consumer_registers<2>(); // increase registers for 2 consumer warpgroups
        rt_hf<16, 64> c_r[2][2];
        zero(c_r[0][0]); zero(c_r[0][1]); zero(c_r[1][0]); zero(c_r[1][1]);
        coord tile_coords;
        get_coords(tile_coords, g, warpgroup::groupid());
        wgmma::descriptor<tile, 0> a_descs[7] = {
            wgmma::descriptor<tile, 0>(a_s[0]),
            wgmma::descriptor<tile, 0>(a_s[1]),
            wgmma::descriptor<tile, 0>(a_s[2]),
            wgmma::descriptor<tile, 0>(a_s[3]),
            wgmma::descriptor<tile, 0>(a_s[4]),
            wgmma::descriptor<tile, 0>(a_s[5]),
            wgmma::descriptor<tile, 0>(a_s[6])
        };
        wgmma::descriptor<tile, 1> b_descs[7] = {
            wgmma::descriptor<tile, 1>(b_s[warpgroup::groupid()][0]),
            wgmma::descriptor<tile, 1>(b_s[warpgroup::groupid()][1]),
            wgmma::descriptor<tile, 1>(b_s[warpgroup::groupid()][2]),
            wgmma::descriptor<tile, 1>(b_s[warpgroup::groupid()][3]),
            wgmma::descriptor<tile, 1>(b_s[warpgroup::groupid()][4]),
            wgmma::descriptor<tile, 1>(b_s[warpgroup::groupid()][5]),
            wgmma::descriptor<tile, 1>(b_s[warpgroup::groupid()][6])
        };
        for(int work_iter = 0; work_iter < iters; work_iter++) {
            work(c_r, a_descs, b_descs, inputs_arrived, inputs_finished, work_iter);
            // work(c_r, a_s, b_s[warpgroup::groupid()], inputs_arrived, inputs_finished, work_iter);
        }
        group<NUM_CONSUMER_WARPS>::sync(0);
        finish(g.C, c_s[warpgroup::groupid()], c_r, tile_coords);
    }
}

// __device__ static inline int num_iters(const globals &g) { return 2; }
// __host__ static inline dim3 get_grid(int M, int N, int K) {
//     return dim3(1);
// }
// __global__ __launch_bounds__(NUM_THREADS, 1)
// void dummy_kernel(const __grid_constant__ globals g) {

//     // Initialize barriers. This is constant for all two-stage producer-consumer kernels.
//     __shared__ kittens::barrier inputs_arrived[7], inputs_finished[7];
//     if (warpid() == 0) { // a single warp (in fact a single thread) does these.
//         for(int i = 0; i < 7; i++) {
//             init_barrier(inputs_arrived[i], 4, 0); // needs to wait on each producer warp
//             init_barrier(inputs_finished[i], 8, 0); // needs to wait on each consumer warp
//         }
//     }
//     int iters = 2;

//     __syncthreads(); // all warps must arrive here, confirming barrier initialization is visible to all threads.

//     if(warpid() >= NUM_CONSUMER_WARPS) { // last warpgroup is a producer
//         for(int load_iter = 0; load_iter < iters; load_iter++) {
//             for(int i = 0; i < 7; i++) {
//                 if(load_iter>0) wait(inputs_finished[i], (load_iter%2)^1);
//                 if(laneid() == 0) arrive(inputs_arrived[i]);
//                 __syncwarp();
//             }
//         }
//     }
//     else { // other warpgroups are consumers
//         for(int work_iter = 0; work_iter < iters; work_iter++) {
//             for(int i = 0; i < 7; i++) {
//                 wait(inputs_arrived[i], work_iter%2);
//                 if(laneid() == 0) arrive(inputs_finished[i]);
//                 __syncwarp();
//             }
//         }
//     }
//     if(laneid() == 0) printf("%d %d finished\n", blockIdx.x, warpid());
//     __syncthreads();
// }

// #define mini_matmul dummy_kernel