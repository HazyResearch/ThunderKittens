#include "conversions.cuh"

#ifdef TEST_GROUP_SHARED_TILE_CONVERSIONS

struct test_shared_copy {
    template<int H, int W, int NW> using valid = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = "shared_copy";
    template<int H, int W, int NW, gl_t GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, gl_t GL> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st_bf<16*H, 16*W> &t1 = al.allocate<kittens::st_bf<16*H, 16*W>>();
        kittens::st_bf<16*H, 16*W> &t2 = al.allocate<kittens::st_bf<16*H, 16*W>>();
        G::load(t2, input, {});
        G::sync(0);
        G::copy(t1, t2);
        G::sync(0);
        G::store(output, t1, {});
    }
};

template<typename T>
struct test_subtile {
    using dtype = T;
    template<int H, int W, int NW, typename _ST_H, typename _ST_W> using valid = std::bool_constant<(
        NW == 1 && W*H<=64
        && (H % _ST_H::value == 0 && W % _ST_W::value == 0 ) 
        && sizeof(dtype) != 1
    )>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_subtile_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_subtile_gmem=half" :
                                                      #ifdef KITTENS_HOPPER
                                                      std::is_same_v<T, kittens::fp8e4m3> ? "shared_subtile_gmem=fp8e4m3" :
                                                      std::is_same_v<T, kittens::fp8e5m2> ? "shared_subtile_gmem=fp8e5m2" :
                                                      #endif
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
        kittens::warp::load(t, input, {});
        __syncwarp();
        for(int i = 0; i < H/ST_H; i++) {
            for(int j = 0; j < W/ST_W; j++) {
                auto ref = t.template subtile<16*ST_H, 16*ST_W>({i, j});
                kittens::rt_fl<16*ST_H, 16*ST_W> reg;
                kittens::warp::load(reg, ref);
                kittens::mul(reg, reg, float(i));
                kittens::add(reg, reg, float(j));
                kittens::warp::store(ref, reg);
            }
        }
        __syncwarp();
        kittens::warp::store(output, t, {});
    }
};

void group::shared::tile::conversions::tests(test_data &results) {
    std::cout << " ----- Starting ops/group/shared/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_2d<test_shared_copy, SIZE, SIZE, 1>::run(results);
    sweep_size_2d<test_shared_copy, SIZE, SIZE, 2>::run(results);

    if constexpr (TEST_INTENSITY > 1) {

        sweep_size_2d<test_shared_copy, SIZE, SIZE, 4>::run(results);

        if constexpr (TEST_INTENSITY > 3) {

            sweep_size_2d<test_shared_copy, 12, 4, 12>::run(results);

        }
    }
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 1>, std::integral_constant<int, 1>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 1>, std::integral_constant<int, 2>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 1>, std::integral_constant<int, 3>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 1>, std::integral_constant<int, 4>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 2>, std::integral_constant<int, 1>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 2>, std::integral_constant<int, 2>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 2>, std::integral_constant<int, 3>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 2>, std::integral_constant<int, 4>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 1>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 2>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 3>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 3>, std::integral_constant<int, 4>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 1>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 2>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 3>>::run(results);
    sweep_gmem_type_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 4>, std::integral_constant<int, 4>>::run(results);
    std::cout << std::endl;
}

#endif