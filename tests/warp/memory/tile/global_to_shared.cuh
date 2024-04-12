#include "testing_flags.cuh"

#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED

#include "testing_commons.cuh"

namespace warp {
namespace memory {
namespace tile {
namespace global_to_shared {

struct load_store {
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> using valid = std::bool_constant<NW == 1 &&
        (!std::is_same_v<L, kittens::ducks::st_layout::tma_swizzle> || W == 1 || W == 2 || W == 4) && W*H<=64>;
    static inline const std::string test_identifier = "shared_loadstore";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::st_bf<H, W, L> &shared_tile = al.allocate<kittens::st_bf<H, W, L>>();
        kittens::load(shared_tile, input, W*16);
        kittens::store(output, shared_tile, W*16);
    }
};
struct load_store_async {
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> using valid = std::bool_constant<NW == 1 &&
        (!std::is_same_v<L, kittens::ducks::st_layout::tma_swizzle> || W == 1 || W == 2 || W == 4)>;
    static inline const std::string test_identifier = "shared_loadstore_async";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 

        auto block = cooperative_groups::this_thread_block();
        __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
        if (threadIdx.x == 0) {init(&barrier, block.size());}
        block.sync();
        
        kittens::st_bf<H, W, L> &shared_tile = al.allocate<kittens::st_bf<H, W, L>>();

        block.sync();
        kittens::load_async(shared_tile, input, W*16, barrier);
        barrier.arrive_and_wait();

        kittens::store_async(output, shared_tile, W*16, barrier);
        barrier.arrive_and_wait();
    }
};

void tests(test_data &results);

}
}
}
}

#endif