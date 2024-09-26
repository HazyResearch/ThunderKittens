#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
template<int N_BLOCK=2>
struct matmul_layout {
    using  base_tile      = st_hf<64, 64>;
    using  input_layout   = gl<half, 1, 7, -1, -1, base_tile>;
    using  output_layout  = gl<half, 1, 1, -1, -1, base_tile>;
    struct globals        { input_layout A, B; output_layout C; };
    struct input_block    { base_tile a[7], b[N_BLOCK][7]; };
    struct finish_block   { base_tile c[N_BLOCK][2][2]; };
    struct producer_state { kittens::coord coords; };
    struct consumer_state { kittens::coord coords;
                            rt_hf<16, 64> c[2][2]; };
};
template<int _N_BLOCK=2, int _SUPER_M=12>
struct matmul_template {
    static constexpr int M_BLOCK = 1, N_BLOCK = _N_BLOCK, SUPER_M = _SUPER_M;
    static constexpr int M_TILE = 2*M_BLOCK, N_TILE = 2*N_BLOCK;
    using layout    = matmul_layout<N_BLOCK>;
    static constexpr int NUM_CONSUMER_WARPS = N_BLOCK*layout::base_tile::rows / TILE_DIM;
      // Helper functions
    __host__ static inline dim3 grid(int M, int N, int K) {
        return dim3(M*N/(M_TILE*N_TILE*layout::base_tile::num_elements));
    }
    __device__ static inline void get_coords(kittens::coord &coords, const typename layout::globals &g, int id) {
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
    // ThunderKittens template functions
    __device__ static inline int iters(const typename layout::globals &g) { return g.A.cols / 64; }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers(); // decrease registers for producers
            get_coords(args.state.coords, args.globals, -1);
        }
        __device__ static void load(producer_load_args<layout> args) {
            if(warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for(int j = 0; j < 7; j++) {
                    tma::load_async(args.input.a[j], args.globals.A,
                                    {j, args.state.coords.r, args.iter}, args.inputs_arrived);
                }
                for(int i = 0; i < N_BLOCK; i++) {
                    for(int j = 0; j < 7; j++) {
                        tma::load_async(args.input.b[i][j], args.globals.B,
                                        {j, args.iter, args.state.coords.c+i}, args.inputs_arrived);
                    }
                }
                arrive(args.inputs_arrived, 3);
            }
        }
    };
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS/4>(); // increase registers for consumers
            get_coords(args.state.coords, args.globals, warpgroup::groupid());
            zero(args.state.c[0][0]);
            zero(args.state.c[0][1]);
            zero(args.state.c[1][0]);
            zero(args.state.c[1][1]);
        }
        __device__ static void work(consumer_work_args<layout> args) {
            st_hf<64, 64> (&a_s)[7] = args.input.a, (&b_s)[7] = args.input.b[warpgroup::groupid()];
            rt_hf<16, 64> accum[2];
            warpgroup::mm_AB(accum[0], a_s[0], b_s[0]); // M1
            warpgroup::mm_AB(accum[1], a_s[1], b_s[1]); // M2
            warpgroup::mma_async_wait<1>();
            add(args.state.c[0][0], args.state.c[0][0], accum[0]); // M1
            add(args.state.c[1][1], args.state.c[1][1], accum[0]);
            warpgroup::mm_AB(accum[0], a_s[2], b_s[2]); // M3
            warpgroup::mma_async_wait<1>();
            add(args.state.c[1][0], args.state.c[1][0], accum[1]); // M2
            sub(args.state.c[1][1], args.state.c[1][1], accum[1]);
            warpgroup::mm_AB(accum[1], a_s[3], b_s[3]); // M4
            warpgroup::mma_async_wait<1>();
            add(args.state.c[0][1], args.state.c[0][1], accum[0]); // M3
            add(args.state.c[1][1], args.state.c[1][1], accum[0]);
            warpgroup::mm_AB(accum[0], a_s[4], b_s[4]); // M5
            warpgroup::mma_async_wait<1>();
            add(args.state.c[0][0], args.state.c[0][0], accum[1]); // M4
            add(args.state.c[1][0], args.state.c[1][0], accum[1]);
            warpgroup::mma_AB(args.state.c[1][1], a_s[5], b_s[5]); // M6, accumulate into C[1][1]
            warpgroup::mma_async_wait<1>();
            sub(args.state.c[0][0], args.state.c[0][0], accum[0]); // M5
            add(args.state.c[0][1], args.state.c[0][1], accum[0]);
            warpgroup::mma_AB(args.state.c[0][0], a_s[6], b_s[6]); // M7, accumulate into C[0][0]
            warpgroup::mma_async_wait();
            if(warpgroup::laneid() == 0) arrive(args.inputs_finished, 4);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            st_hf<64, 64> (&c_s)[2][2] = args.finish.c[warpgroup::groupid()];
            for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
                warpgroup::store(c_s[i][j], args.state.c[i][j]);
                warpgroup::sync();
                if(warpgroup::warpid() == 0) {
                    tma::store_async(args.globals.C, c_s[i][j],
                                    {args.state.coords.r+i, args.state.coords.c+j});
                }
            }
        }
    };
};