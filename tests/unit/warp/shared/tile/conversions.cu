#include "conversions.cuh"

#ifdef TEST_WARP_SHARED_TILE_CONVERSIONS

// unlike most checks, this one is broken up for compile time versions. (See conversions_2.cu for more details.)

template<typename T>
struct test_swap_layout {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_swaplayout_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_swaplayout_gmem=half" :
                                                                                         "shared_swaplayout_gmem=float";
    template<int H, int W, int NW, gl_t GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, gl_t GL> __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st<dtype, 16*H, 16*W> &t1 = al.allocate<kittens::st<dtype, 16*H, 16*W>>();
        kittens::st<dtype, 16*H, 16*W> &t2 = al.allocate<kittens::st<dtype, 16*H, 16*W>>();
        kittens::load(t2, input, {});
        kittens::copy(t1, t2);
        kittens::store(output, t1, {});
    }
};

template<typename T>
struct test_subtile {
    using dtype = T;
    template<int H, int W, int NW, typename _ST_H, typename _ST_W> using valid = std::bool_constant<NW == 1 && W*H<=64
        && (H % _ST_H::value == 0 && W % _ST_W::value == 0)>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_subtile_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_subtile_gmem=half" :
                                                                                         "shared_subtile_gmem=float";
    template<int H, int W, int NW, gl_t GL, typename _ST_H, typename _ST_W> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int ST_H = _ST_H::value, ST_W = _ST_W::value;
        for(int i = 0; i < H*16; i++)
            for(int j = 0; j < W*16; j++)
                o_ref[i*W*16 + j] = i_ref[i*W*16 + j] * float(i/(ST_H*16)) + float(j/(ST_W*16));
    }
    template<int H, int W, int NW, gl_t GL, typename _ST_H, typename _ST_W> __device__ static void device_func(const GL &input, const GL &output) {
        constexpr int ST_H = _ST_H::value, ST_W = _ST_W::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st<dtype, 16*H, 16*W> &t = al.allocate<kittens::st<dtype, 16*H, 16*W>>();
        kittens::load(t, input, {});
        __syncwarp();
        for(int i = 0; i < H/ST_H; i++) {
            for(int j = 0; j < W/ST_W; j++) {
                auto ref = kittens::subtile_inplace<16*ST_H, 16*ST_W>(t, i, j);
                kittens::rt_fl<16*ST_H, 16*ST_W> reg;
                kittens::load(reg, ref);
                kittens::mul(reg, reg, float(i));
                kittens::add(reg, reg, float(j));
                kittens::store(ref, reg);
            }
        }
        __syncwarp();
        kittens::store(output, t, {});
    }
};

void warp::shared::tile::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 6  :
                         INTENSITY_4 ? 8 : -1;

    sweep_gmem_type_2d_warp<test_swap_layout, SIZE, SIZE>::run(results);
                         
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 1>, std::integral_constant<int, 1>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 1>, std::integral_constant<int, 3>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 1>, std::integral_constant<int, 4>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 1>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 2>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 3>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 4>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 1>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 3>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 4>>::run(results);
}

#endif