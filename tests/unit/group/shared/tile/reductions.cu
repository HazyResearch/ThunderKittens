#include "reductions.cuh"

#ifdef TEST_GROUP_SHARED_TILE_REDUCTIONS

struct normalize_row {
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = "shared_norm_row";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
        std::vector<kittens::bf16> i_ref(i_ref_f.size());
        std::vector<kittens::bf16> o_ref(o_ref_f.size());
        for(int i = 0; i < i_ref.size(); i++) i_ref[i] = __float2bfloat16(i_ref_f[i]);
        for(int i = 0; i < H*16; i++) {
            kittens::bf16 row_sum = 0;
            for(int j = 0; j < W*16; j++) {
                o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                row_sum         += i_ref[i*W*16+j];
            }
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] /= row_sum;
        }
        for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = __bfloat162float(o_ref[i]);
    }
    template<int H, int W, int NW> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st_bf<H, W> &shared_tile = al.allocate<kittens::st_bf<H, W>>();
        __shared__ kittens::col_vec<typeof(shared_tile)> accum;
        kittens::load(shared_tile, input, W*16);
        __syncthreads();
        kittens::row_sum(accum, shared_tile);
        __syncthreads();
        kittens::div_row(shared_tile, shared_tile, accum);
        __syncthreads();
        kittens::store(output, shared_tile, W*16);
    }
};
struct normalize_col {
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = "shared_norm_col";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
        std::vector<kittens::bf16> i_ref(i_ref_f.size());
        std::vector<kittens::bf16> o_ref(o_ref_f.size());
        for(int i = 0; i < i_ref.size(); i++) i_ref[i] = __float2bfloat16(i_ref_f[i]);
        for(int i = 0; i < W*16; i++) {
            kittens::bf16 col_sum = 0;
            for(int j = 0; j < H*16; j++) {
                o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                col_sum         += i_ref[i+j*W*16];
            }
            for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] /= col_sum;
        }
        for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = __bfloat162float(o_ref[i]);
    }
    template<int H, int W, int NW> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st_bf<H, W> &shared_tile = al.allocate<kittens::st_bf<H, W>>();
        __shared__ kittens::row_vec<typeof(shared_tile)> accum;
        kittens::load(shared_tile, input, W*16);
        __syncthreads();
        kittens::col_sum(accum, shared_tile);
        __syncthreads();
        kittens::div_col(shared_tile, shared_tile, accum);
        __syncthreads();
        kittens::store(output, shared_tile, W*16);
    }
};
struct broadcast_row {
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = "shared_broadcast_row";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
        std::vector<kittens::bf16> i_ref(i_ref_f.size());
        std::vector<kittens::bf16> o_ref(o_ref_f.size());
        for(int i = 0; i < i_ref.size(); i++) i_ref[i] = __float2bfloat16(i_ref_f[i]);
        for(int i = 0; i < H*16; i++) {
            kittens::bf16 row_sum = 0;
            for(int j = 0; j < W*16; j++) {
                o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                row_sum         += i_ref[i*W*16+j];
            }
            for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] = row_sum;
        }
        for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = __bfloat162float(o_ref[i]);
    }
    template<int H, int W, int NW> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st_bf<H, W> &shared_tile = al.allocate<kittens::st_bf<H, W>>();
        __shared__ kittens::col_vec<typeof(shared_tile)> accum;
        kittens::load(shared_tile, input, W*16);
        __syncthreads();
        kittens::row_sum(accum, shared_tile);
        __syncthreads();
        kittens::broadcast_row(shared_tile, accum);
        __syncthreads();
        kittens::store(output, shared_tile, W*16);
    }
};
struct broadcast_col {
    template<int H, int W, int NW> using valid = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = "shared_broadcast_col";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
        std::vector<kittens::bf16> i_ref(i_ref_f.size());
        std::vector<kittens::bf16> o_ref(o_ref_f.size());
        for(int i = 0; i < i_ref.size(); i++) i_ref[i] = __float2bfloat16(i_ref_f[i]);
        for(int i = 0; i < W*16; i++) {
            kittens::bf16 col_sum = 0;
            for(int j = 0; j < H*16; j++) {
                o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                col_sum         += i_ref[i+j*W*16];
            }
            for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] = col_sum;
        }
        for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = __bfloat162float(o_ref[i]);
    }
    template<int H, int W, int NW> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st_bf<H, W> &shared_tile = al.allocate<kittens::st_bf<H, W>>();
        __shared__ kittens::row_vec<typeof(shared_tile)> accum;
        G::load(shared_tile, input, W*16);
        __syncthreads();
        G::col_sum(accum, shared_tile);
        __syncthreads();
        G::broadcast_col(shared_tile, accum);
        __syncthreads();
        G::store(output, shared_tile, W*16);
    }
};

void group::shared::tile::reductions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/shared/tile/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_2d<normalize_row, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<normalize_col, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<broadcast_row, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<broadcast_col, SIZE, SIZE, 2>::run(results);


    if constexpr (TEST_INTENSITY > 1) {

        sweep_size_2d<normalize_row, SIZE, SIZE, 4>::run(results);
        sweep_size_2d<normalize_col, SIZE, SIZE, 4>::run(results);
        sweep_size_2d<broadcast_row, SIZE, SIZE, 4>::run(results);
        sweep_size_2d<broadcast_col, SIZE, SIZE, 4>::run(results);

        if constexpr (TEST_INTENSITY > 3) {

            sweep_size_2d<normalize_row, 12, 5, 12>::run(results);
            sweep_size_2d<normalize_col, 12, 5, 12>::run(results);
            sweep_size_2d<broadcast_row, 12, 5, 12>::run(results);
            sweep_size_2d<broadcast_col, 12, 5, 12>::run(results);

        }
    }
}

#endif