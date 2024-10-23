#include "global_to_shared.cuh"

#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_SHARED

template<typename T>
struct group_shared_load_store {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<H%NW==0 &&
        W*H<=64>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_shared_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_shared_loadstore_gmem=half" :
                                                                                         "group_shared_loadstore_gmem=float";
    template<int H, int W, int NW, gl_t GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, gl_t GL> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::st<dtype, 16*H, 16*W> &shared_tile = al.allocate<kittens::st<dtype, 16*H, 16*W>>();
        G::load(shared_tile, input, {});
        G::store(output, shared_tile, {});
    }
};
template<typename T>
struct group_shared_load_store_async {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<H%NW==0 &&
        H*W<=64>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_shared_loadstore_async_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_shared_loadstore_async_gmem=half" :
                                                                                         "group_shared_loadstore_async_gmem=float";
    template<int H, int W, int NW, gl_t GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, gl_t GL> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        __syncthreads();
        
        kittens::st<dtype, 16*H, 16*W> &shared_tile = al.allocate<kittens::st<dtype, 16*H, 16*W>>();

        G::load_async(shared_tile, input, {});
        G::load_async_wait(0);
        G::store(output, shared_tile, {});
    }
};

void group::memory::tile::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/tile/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_2d<group_shared_load_store<float>, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_shared_load_store<float>, SIZE, SIZE, 4>::run(results);
    sweep_size_2d<group_shared_load_store<float>, SIZE, 4, 12>::run(results);
    sweep_size_2d<group_shared_load_store_async<float>, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_shared_load_store_async<float>, SIZE, SIZE, 4>::run(results);
    sweep_size_2d<group_shared_load_store_async<float>, SIZE, 4, 12>::run(results);

    sweep_size_2d<group_shared_load_store<kittens::bf16>, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_shared_load_store<kittens::bf16>, SIZE, SIZE, 4>::run(results);
    sweep_size_2d<group_shared_load_store<kittens::bf16>, SIZE, 4, 12>::run(results);
    sweep_size_2d<group_shared_load_store_async<kittens::bf16>, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_shared_load_store_async<kittens::bf16>, SIZE, SIZE, 4>::run(results);
    sweep_size_2d<group_shared_load_store_async<kittens::bf16>, SIZE, 4, 12>::run(results);

    sweep_size_2d<group_shared_load_store<kittens::half>, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_shared_load_store<kittens::half>, SIZE, SIZE, 4>::run(results);
    sweep_size_2d<group_shared_load_store<kittens::half>, SIZE, 4, 12>::run(results);
    sweep_size_2d<group_shared_load_store_async<kittens::half>, SIZE, SIZE, 2>::run(results);
    sweep_size_2d<group_shared_load_store_async<kittens::half>, SIZE, SIZE, 4>::run(results);
    sweep_size_2d<group_shared_load_store_async<kittens::half>, SIZE, 4, 12>::run(results);
}

#endif