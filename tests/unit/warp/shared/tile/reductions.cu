#include "reductions.cuh"

#ifdef TEST_WARP_SHARED_TILE_REDUCTIONS

template<typename T>
struct normalize_row {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_norm_row_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_norm_row_gmem=half" :
                                                                                         "shared_norm_row_gmem=float";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
        if constexpr(std::is_same_v<dtype, kittens::bf16>) {
            std::vector<kittens::bf16> i_ref(i_ref_f.size());
            std::vector<kittens::bf16> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = __float2bfloat16(i_ref_f[i]);
            for(int i = 0; i < H*16; i++) {
                kittens::bf16 row_sum = 0;
                for(int j = 0; j < W*16; j++) {
                    o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                    row_sum         += i_ref[i*W*16+j];
                }
                row_sum = __habs(row_sum)+__float2bfloat16(1.f);
                for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] /= row_sum;
            }
            for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = __bfloat162float(o_ref[i]);
        }
        else if constexpr(std::is_same_v<dtype, float>) {
            std::vector<float> i_ref(i_ref_f.size());
            std::vector<float> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = i_ref_f[i];
            for(int i = 0; i < H*16; i++) {
                float row_sum = 0;
                for(int j = 0; j < W*16; j++) {
                    o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                    row_sum         += i_ref[i*W*16+j];
                }
                row_sum = abs(row_sum)+1.f;
                for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] /= row_sum;
            }
            for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = o_ref[i];
        }
        else {
            std::vector<kittens::half> i_ref(i_ref_f.size());
            std::vector<kittens::half> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = __float2half(i_ref_f[i]);
            for(int i = 0; i < H*16; i++) {
                kittens::half row_sum = 0;
                for(int j = 0; j < W*16; j++) {
                    o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                    row_sum         += i_ref[i*W*16+j];
                }
                row_sum = __habs(row_sum)+__float2half(1.f);
                for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] /= row_sum;
            }
            for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = __half2float(o_ref[i]);
        }
    }
    template<int H, int W, int NW> __device__ static void device_func(const dtype *input, dtype *output) {
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st<dtype, H, W> &shared_tile = al.allocate<kittens::st<dtype, H, W>>();
        __shared__ kittens::col_vec<typeof(shared_tile)> accum;
        kittens::load(shared_tile, input, W*16);
        kittens::row_sum(accum, shared_tile);
        kittens::abs(accum, accum);
        kittens::add(accum, accum, kittens::base_types::constants<dtype>::one());
        kittens::div_row(shared_tile, shared_tile, accum);
        kittens::store(output, shared_tile, W*16);
    }
};
template<typename T>
struct normalize_col {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_norm_col_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_norm_col_gmem=half" :
                                                                                         "shared_norm_col_gmem=float";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
        if constexpr(std::is_same_v<dtype, kittens::bf16>) {
            std::vector<kittens::bf16> i_ref(i_ref_f.size());
            std::vector<kittens::bf16> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = __float2bfloat16(i_ref_f[i]);
            for(int i = 0; i < W*16; i++) {
                kittens::bf16 col_sum = 0;
                for(int j = 0; j < H*16; j++) {
                    o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                    col_sum         += i_ref[i+j*W*16];
                }
                col_sum = __habs(col_sum)+__float2bfloat16(1.f);
                for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] /= col_sum;
            }
            for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = __bfloat162float(o_ref[i]);
        }
        else if constexpr(std::is_same_v<dtype, float>) {
            std::vector<float> i_ref(i_ref_f.size());
            std::vector<float> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = i_ref_f[i];
            for(int i = 0; i < W*16; i++) {
                float col_sum = 0;
                for(int j = 0; j < H*16; j++) {
                    o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                    col_sum         += i_ref[i+j*W*16];
                }
                col_sum = abs(col_sum)+1.f;
                for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] /= col_sum;
            }
            for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = o_ref[i];
        }
        else {
            std::vector<kittens::half> i_ref(i_ref_f.size());
            std::vector<kittens::half> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = __float2half(i_ref_f[i]);
            for(int i = 0; i < W*16; i++) {
                kittens::half col_sum = 0;
                for(int j = 0; j < H*16; j++) {
                    o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                    col_sum         += i_ref[i+j*W*16];
                }
                col_sum = __habs(col_sum)+__float2half(1.f);
                for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] /= col_sum;
            }
            for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = __half2float(o_ref[i]);
        }
    }
    template<int H, int W, int NW> __device__ static void device_func(const dtype *input, dtype *output) {
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st<dtype, H, W> &shared_tile = al.allocate<kittens::st<dtype, H, W>>();
        __shared__ kittens::row_vec<typeof(shared_tile)> accum;
        kittens::load(shared_tile, input, W*16);
        kittens::col_sum(accum, shared_tile);
        kittens::abs(accum, accum);
        kittens::add(accum, accum, kittens::base_types::constants<dtype>::one());
        kittens::div_col(shared_tile, shared_tile, accum);
        kittens::store(output, shared_tile, W*16);
    }
};
template<typename T>
struct broadcast_row {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_broadcast_row_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_broadcast_row_gmem=half" :
                                                                                         "shared_broadcast_row_gmem=float";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
        if constexpr(std::is_same_v<dtype, kittens::bf16>) {
            std::vector<kittens::bf16> i_ref(i_ref_f.size());
            std::vector<kittens::bf16> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = __float2bfloat16(i_ref_f[i]/8.f);
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
        else if constexpr(std::is_same_v<dtype, float>) {
            std::vector<float> i_ref(i_ref_f.size());
            std::vector<float> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = i_ref_f[i]/8.f;
            for(int i = 0; i < H*16; i++) {
                float row_sum = 0;
                for(int j = 0; j < W*16; j++) {
                    o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                    row_sum         += i_ref[i*W*16+j];
                }
                for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] = row_sum;
            }
            for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = o_ref[i];
        }
        else {
            std::vector<kittens::half> i_ref(i_ref_f.size());
            std::vector<kittens::half> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = __float2half(i_ref_f[i]/8.f);
            for(int i = 0; i < H*16; i++) {
                kittens::half row_sum = 0;
                for(int j = 0; j < W*16; j++) {
                    o_ref[i*W*16+j]  = i_ref[i*W*16+j];
                    row_sum         += i_ref[i*W*16+j];
                }
                for(int j = 0; j < W*16; j++) o_ref[i*W*16+j] = row_sum;
            }
            for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = __half2float(o_ref[i]);
        }
    }
    template<int H, int W, int NW> __device__ static void device_func(const dtype *input, dtype *output) {
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st<dtype, H, W> &shared_tile = al.allocate<kittens::st<dtype, H, W>>();
        __shared__ kittens::col_vec<typeof(shared_tile)> accum;
        kittens::load(shared_tile, input, W*16);
        kittens::mul(shared_tile, shared_tile, kittens::base_types::convertor<dtype, float>::convert(0.125f));
        kittens::row_sum(accum, shared_tile);
        kittens::broadcast_row(shared_tile, accum);
        kittens::store(output, shared_tile, W*16);
    }
};

template<typename T>
struct broadcast_col {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_broadcast_col_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_broadcast_col_gmem=half" :
                                                                                         "shared_broadcast_col_gmem=float";
    template<int H, int W, int NW> __host__ static void host_func(const std::vector<float> &i_ref_f, std::vector<float> &o_ref_f) {
        if constexpr(std::is_same_v<dtype, kittens::bf16>) {
            std::vector<kittens::bf16> i_ref(i_ref_f.size());
            std::vector<kittens::bf16> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = __float2bfloat16(i_ref_f[i]/8.f);
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
        else if constexpr(std::is_same_v<dtype, float>) {
            std::vector<float> i_ref(i_ref_f.size());
            std::vector<float> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = i_ref_f[i]/8.f;
            for(int i = 0; i < W*16; i++) {
                float col_sum = 0;
                for(int j = 0; j < H*16; j++) {
                    o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                    col_sum         += i_ref[i+j*W*16];
                }
                for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] = col_sum;
            }
            for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = o_ref[i];
        }
        else {
            std::vector<kittens::half> i_ref(i_ref_f.size());
            std::vector<kittens::half> o_ref(o_ref_f.size());
            for(int i = 0; i < i_ref.size(); i++) i_ref[i] = __float2half(i_ref_f[i]/8.f);
            for(int i = 0; i < W*16; i++) {
                kittens::half col_sum = 0;
                for(int j = 0; j < H*16; j++) {
                    o_ref[i+j*W*16]  = i_ref[i+j*W*16];
                    col_sum         += i_ref[i+j*W*16];
                }
                for(int j = 0; j < H*16; j++) o_ref[i+j*W*16] = col_sum;
            }
            for(int i = 0; i < o_ref.size(); i++) o_ref_f[i] = __half2float(o_ref[i]);
        }
    }
    template<int H, int W, int NW> __device__ static void device_func(const dtype *input, dtype *output) {
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st<dtype, H, W> &shared_tile = al.allocate<kittens::st<dtype, H, W>>();
        __shared__ kittens::row_vec<typeof(shared_tile)> accum;
        kittens::load(shared_tile, input, W*16);
        kittens::mul(shared_tile, shared_tile, kittens::base_types::convertor<dtype, float>::convert(0.125f));
        kittens::col_sum(accum, shared_tile);
        kittens::broadcast_col(shared_tile, accum);
        kittens::store(output, shared_tile, W*16);
    }
};

void warp::shared::tile::reductions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/tile/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_gmem_type_2d_warp<normalize_row, SIZE, SIZE>::run(results);
    sweep_gmem_type_2d_warp<normalize_col, SIZE, SIZE>::run(results);
    sweep_gmem_type_2d_warp<broadcast_row, SIZE, SIZE>::run(results);
    sweep_gmem_type_2d_warp<broadcast_col, SIZE, SIZE>::run(results);
}

#endif