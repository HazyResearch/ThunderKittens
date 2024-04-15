#include "global_to_shared.cuh"

#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_SHARED

struct load_store {
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> using valid = std::bool_constant<H%NW==0 &&
        (!std::is_same_v<L, kittens::ducks::st_layout::xor_swizzle> || W == 1 || W == 2 || W == 4 || W == 8 || W == 16) && W*H<=64>;
    static inline const std::string test_identifier = "shared_loadstore";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::st_bf<H, W, L> &shared_tile = al.allocate<kittens::st_bf<H, W, L>>();
        G::load(shared_tile, input, W*16);
        G::store(output, shared_tile, W*16);
    }
};
struct load_store_async {
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> using valid = std::bool_constant<H%NW==0 &&
        (!std::is_same_v<L, kittens::ducks::st_layout::xor_swizzle> || W == 1 || W == 2 || W == 4 || W == 8 || W == 16) && H*W<=64>;
    static inline const std::string test_identifier = "shared_loadstore_async";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 

        auto block = cooperative_groups::this_thread_block();
        __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
        if (threadIdx.x == 0) {init(&barrier, block.size());}
        block.sync();
        
        kittens::st_bf<H, W, L> &shared_tile = al.allocate<kittens::st_bf<H, W, L>>();

        block.sync();
        G::load_async(shared_tile, input, W*16, barrier);
        barrier.arrive_and_wait();

        G::store_async(output, shared_tile, W*16, barrier);
        barrier.arrive_and_wait();
    }
};

void group::memory::tile::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/tile/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_st_layout_size_2d<load_store, SIZE, SIZE, 2>::run(results);
    sweep_st_layout_size_2d<load_store, SIZE, SIZE, 4>::run(results);
    sweep_st_layout_size_2d<load_store, SIZE, 4, 12>::run(results);

    sweep_st_layout_size_2d<load_store_async, SIZE, SIZE, 2>::run(results);
    sweep_st_layout_size_2d<load_store_async, SIZE, SIZE, 4>::run(results);
    sweep_st_layout_size_2d<load_store_async, SIZE, 4, 12>::run(results);
}

#endif