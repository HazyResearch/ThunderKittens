#include "global_to_shared.cuh"

#ifdef TEST_GROUP_MEMORY_VEC_GLOBAL_TO_SHARED

template<typename T>
struct vec_load_store {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<S%NW==0 && S<=64>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_shared_vec_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_shared_vec_loadstore_gmem=half" :
                                                                                         "group_shared_vec_loadstore_gmem=float";
    template<int S, int NW, gl_t GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, gl_t GL> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::col_vec<kittens::st<dtype, 16*S, 16*S>> &shared_vec = al.allocate<kittens::col_vec<kittens::st<dtype, 16*S, 16*S>>>();
        G::load(shared_vec, input, {});
        G::store(output, shared_vec, {});
    }
};

template<typename T>
struct vec_async_load_store {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<S%NW==0 && S<=64>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_shared_vec_loadstore_async_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_shared_vec_loadstore_async_gmem=half" :
                                                                                         "group_shared_vec_loadstore_async_gmem=float";
    template<int S, int NW, gl_t GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, gl_t GL> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::col_vec<kittens::st<dtype, 16*S, 16*S>> &shared_vec = al.allocate<kittens::col_vec<kittens::st<dtype, 16*S, 16*S>>>();
        G::load_async(shared_vec, input, {});
        G::load_async_wait(0);
        G::store(output, shared_vec, {});
    }
};

void group::memory::vec::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/vec/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_1d<vec_load_store<float>, SIZE, 2>::run(results);
    sweep_size_1d<vec_load_store<float>, SIZE, 4>::run(results);
    sweep_size_1d<vec_load_store<float>, SIZE, 12>::run(results);
    sweep_size_1d<vec_load_store<kittens::bf16>, SIZE, 2>::run(results);
    sweep_size_1d<vec_load_store<kittens::bf16>, SIZE, 4>::run(results);
    sweep_size_1d<vec_load_store<kittens::bf16>, SIZE, 12>::run(results);
    sweep_size_1d<vec_load_store<kittens::half>, SIZE, 2>::run(results);
    sweep_size_1d<vec_load_store<kittens::half>, SIZE, 4>::run(results);
    sweep_size_1d<vec_load_store<kittens::half>, SIZE, 12>::run(results);
                         
    sweep_size_1d<vec_async_load_store<float>, SIZE, 2>::run(results);
    sweep_size_1d<vec_async_load_store<float>, SIZE, 4>::run(results);
    sweep_size_1d<vec_async_load_store<float>, SIZE, 12>::run(results);
    sweep_size_1d<vec_async_load_store<kittens::bf16>, SIZE, 2>::run(results);
    sweep_size_1d<vec_async_load_store<kittens::bf16>, SIZE, 4>::run(results);
    sweep_size_1d<vec_async_load_store<kittens::bf16>, SIZE, 12>::run(results);
    sweep_size_1d<vec_async_load_store<kittens::half>, SIZE, 2>::run(results);
    sweep_size_1d<vec_async_load_store<kittens::half>, SIZE, 4>::run(results);
    sweep_size_1d<vec_async_load_store<kittens::half>, SIZE, 12>::run(results);
}

#endif