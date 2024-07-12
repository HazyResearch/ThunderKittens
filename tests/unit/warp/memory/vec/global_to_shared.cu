#include "global_to_shared.cuh"

#ifdef TEST_WARP_MEMORY_VEC_GLOBAL_TO_SHARED

template<typename T>
struct shared_vec_load_store {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<NW == 1 && S<=64>;
    static inline const std::string test_identifier = std::is_same_v<dtype, kittens::bf16> ? "shared_vec_loadstore_gmem=bf16" :
                                                      std::is_same_v<dtype, kittens::half> ? "shared_vec_loadstore_gmem=half" :
                                                                                             "shared_vec_loadstore_gmem=float";
    template<int S, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW> __device__ static void device_func(const dtype *input, dtype *output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::col_vec<kittens::st<dtype, S, S>> &shared_vec = al.allocate<kittens::col_vec<kittens::st<dtype, S, S>>>();
        kittens::load(shared_vec, input);
        kittens::store(output, shared_vec);
    }
};

void warp::memory::vec::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_gmem_type_1d_warp<shared_vec_load_store, SIZE>::run(results);
}

#endif