#include "global_to_shared.cuh"

#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED

template<typename T>
struct load_store {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> using valid = std::bool_constant<NW == 1 &&
        (!std::is_same_v<L, kittens::ducks::st_layout::swizzle> || W == 1 || W == 2 || W == 4 || W == 8 || W == 16) && W*H<=64>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_loadstore_gmem=half" :
                                                                                         "shared_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __device__ static void device_func(const T *input, T *output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::st<T, H, W, L> &shared_tile = al.allocate<kittens::st<T, H, W, L>>();
        kittens::load(shared_tile, input, W*16);
        kittens::store(output, shared_tile, W*16);
    }
};
template<typename T>
struct load_store_async {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> using valid = std::bool_constant<NW == 1 &&
        (!std::is_same_v<L, kittens::ducks::st_layout::swizzle> || W == 1 || W == 2 || W == 4 || W == 8 || W == 16) && W*H<=64>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_loadstore_async_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_loadstore_async_gmem=half" :
                                                                                         "shared_loadstore_async_gmem=float";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __device__ static void device_func(const T *input, T *output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 

        auto block = cooperative_groups::this_thread_block();
        __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier;
        if (threadIdx.x == 0) {init(&barrier, block.size());}
        block.sync();
        
        kittens::st<T, H, W, L> &shared_tile = al.allocate<kittens::st<T, H, W, L>>();

        block.sync();
        kittens::load_async(shared_tile, input, W*16, barrier);
        barrier.arrive_and_wait();

        kittens::store_async(output, shared_tile, W*16, barrier);
        barrier.arrive_and_wait();
    }
};

void warp::memory::tile::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_gmem_type_2d_warp<load_store, SIZE, SIZE>::run(results);
    sweep_gmem_type_2d_warp<load_store_async, SIZE, SIZE>::run(results);
}

#endif