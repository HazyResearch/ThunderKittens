#include "testing_flags.cuh"

#ifdef TEST_WARP_SHARED_CONVERSIONS

#include "testing_commons.cuh"

namespace warp {
namespace shared {
namespace conversions {

struct swap_layout {
    template<int H, int W, int NW, kittens::ducks::st_layout::all L1, kittens::ducks::st_layout::all L2> using valid = std::bool_constant<NW == 1 && W*H<=64 &&
        (!(std::is_same_v<L1, kittens::ducks::st_layout::tma_swizzle> || std::is_same_v<L2, kittens::ducks::st_layout::tma_swizzle>) || W == 1 || W == 2 || W == 4)>; // this is warp-level
    static inline const std::string test_identifier = "shared_swaplayout";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L1, kittens::ducks::st_layout::all L2> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L1, kittens::ducks::st_layout::all L2> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st_bf<H, W, L1> &t1 = al.allocate<kittens::st_bf<H, W, L1>>();
        kittens::st_bf<H, W, L2> &t2 = al.allocate<kittens::st_bf<H, W, L2>>();
        kittens::load(t2, input, W*16);
        kittens::copy(t1, t2);
        kittens::store(output, t1, W*16);
    }
};

struct subtile {
    template<int H, int W, int NW, kittens::ducks::st_layout::all L, typename _ST_H, typename _ST_W> using valid = std::bool_constant<NW == 1 && W*H<=64 &&
        (!(std::is_same_v<L, kittens::ducks::st_layout::tma_swizzle>) || W == 1 || W == 2 || W == 4)
        && (H % _ST_H::value == 0 && W % _ST_W::value == 0)>;
    static inline const std::string test_identifier = "shared_subtile";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L, typename _ST_H, typename _ST_W> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        constexpr int ST_H = _ST_H::value, ST_W = _ST_W::value;
        for(int i = 0; i < H*16; i++)
            for(int j = 0; j < W*16; j++)
                o_ref[i*W*16 + j] = i_ref[i*W*16 + j] * float(i/(ST_H*16)) + float(j/(ST_W*16));
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L, typename _ST_H, typename _ST_W> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        constexpr int ST_H = _ST_H::value, ST_W = _ST_W::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st_bf<H, W, L> &t = al.allocate<kittens::st_bf<H, W, L>>();
        kittens::load(t, input, W*16);
        for(int i = 0; i < H/ST_H; i++) {
            for(int j = 0; j < W/ST_W; j++) {
                auto ref = kittens::subtile_inplace<ST_H, ST_W>(t, i, j);
                kittens::rt_fl<ST_H, ST_W> reg;
                kittens::load(reg, ref);
                kittens::mul(reg, reg, float(i));
                kittens::add(reg, reg, float(j));
                kittens::store(ref, reg);
            }
        }
        kittens::store(output, t, W*16);
    }
};

void tests(test_data &results);

}
}
}

#endif