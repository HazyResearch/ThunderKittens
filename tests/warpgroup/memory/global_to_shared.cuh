#include "testing_flags.cuh"

#ifdef TEST_WARPGROUP_MEMORY_GLOBAL_TO_SHARED

#include "testing_commons.cuh"

namespace warpgroup {
namespace memory {
namespace global_to_shared {

struct load_store {
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> using valid = std::bool_constant<H%4 == 0 && NW == 4 &&
        (!std::is_same_v<L, kittens::ducks::st_layout::tma_swizzle> || W == 1 || W == 2 || W == 4) && W*H<=64>;
    static inline const std::string test_identifier = "wg_shared_loadstore";
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
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> using valid = std::bool_constant<NW == 4 &&
        (!std::is_same_v<L, kittens::ducks::st_layout::tma_swizzle> || W == 1 || W == 2 || W == 4)>;
    static inline const std::string test_identifier = "wg_shared_loadstore_async";
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
        kittens::warpgroup::load_async(shared_tile, input, W*16, barrier);
        barrier.arrive_and_wait();

        kittens::warpgroup::store_async(output, shared_tile, W*16, barrier);
        barrier.arrive_and_wait();
    }
};

struct vec_load_store {
    template<int S, int NW> using valid = std::bool_constant<NW == 4 && S<=64>;
    static inline const std::string test_identifier = "wg_shared_vec_loadstore";
    template<int S, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::col_vec<kittens::st_bf<S, S>> &shared_vec = al.allocate<kittens::col_vec<kittens::st_bf<S, S>>>();
        kittens::warpgroup::load(shared_vec, input);
        kittens::warpgroup::store(output, shared_vec);
    }
};

void tests(test_data &results);

}
}
}

#endif