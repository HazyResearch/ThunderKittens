#include "testing_flags.cuh"

#ifdef TEST_WARPGROUP_SHARED_MAPS

#include "testing_commons.cuh"

namespace warpgroup {
namespace shared {
namespace maps {
struct test_exp {
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> using valid = std::bool_constant<NW == 4 && W*H<=64>;
    static inline const std::string test_identifier = "warpgroup_exp";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) o_ref[i] = ::expf(i_ref[i]); // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 

        kittens::st_bf<H, W, L> &shared_tile = al.allocate<kittens::st_bf<H, W, L>>();

        kittens::warpgroup::load(shared_tile, input, W*16);
        __syncthreads(); 
        kittens::warpgroup::exp(shared_tile, shared_tile);
        __syncthreads();
        kittens::warpgroup::store(output, shared_tile, W*16);
    }
};

void tests(test_data &results);

}
}
}

#endif