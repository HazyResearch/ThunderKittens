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

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
// fp8 types use sv<T,N> directly: TILE_COL_DIM<fp8>=32 makes st<fp8,16*S,16*S> invalid for odd S
template<typename T>
struct vec_fp8_load_store {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<S%NW==0 && S<=64>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::fp8e4m3> ? "group_shared_vec_loadstore_gmem=fp8e4m3" :
                                                      std::is_same_v<T, kittens::fp8e5m2> ? "group_shared_vec_loadstore_gmem=fp8e5m2" :
                                                                                             "group_shared_vec_loadstore_gmem=fp8e8m0";
    template<int S, int NW, gl_t GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref;
    }
    template<int S, int NW, gl_t GL> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator<16> al((int*)&__shm[0]);
        kittens::sv<T, 16*S> &shared_vec = al.allocate<kittens::sv<T, 16*S>>();
        G::load(shared_vec, input, {});
        G::store(output, shared_vec, {});
    }
};

template<typename T>
struct vec_fp8_async_load_store {
    using dtype = T;
    template<int S, int NW> using valid = std::bool_constant<S%NW==0 && S<=64>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::fp8e4m3> ? "group_shared_vec_loadstore_async_gmem=fp8e4m3" :
                                                      std::is_same_v<T, kittens::fp8e5m2> ? "group_shared_vec_loadstore_async_gmem=fp8e5m2" :
                                                                                             "group_shared_vec_loadstore_async_gmem=fp8e8m0";
    template<int S, int NW, gl_t GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref;
    }
    template<int S, int NW, gl_t GL> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator<16> al((int*)&__shm[0]);
        kittens::sv<T, 16*S> &shared_vec = al.allocate<kittens::sv<T, 16*S>>();
        G::load_async(shared_vec, input, {});
        G::load_async_wait(0);
        G::store(output, shared_vec, {});
    }
};
#endif

void group::memory::vec::global_to_shared::tests(test_data &results) {
    std::cout << " ----- Starting ops/group/memory/vec/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_1d<vec_load_store<float>, SIZE, 1>::run(results);
    sweep_size_1d<vec_load_store<float>, SIZE, 2>::run(results);
    sweep_size_1d<vec_load_store<float>, SIZE, 4>::run(results);
    sweep_size_1d<vec_load_store<float>, SIZE, 12>::run(results);
    sweep_size_1d<vec_load_store<kittens::bf16>, SIZE, 1>::run(results);
    sweep_size_1d<vec_load_store<kittens::bf16>, SIZE, 2>::run(results);
    sweep_size_1d<vec_load_store<kittens::bf16>, SIZE, 4>::run(results);
    sweep_size_1d<vec_load_store<kittens::bf16>, SIZE, 12>::run(results);
    sweep_size_1d<vec_load_store<kittens::half>, SIZE, 1>::run(results);
    sweep_size_1d<vec_load_store<kittens::half>, SIZE, 2>::run(results);
    sweep_size_1d<vec_load_store<kittens::half>, SIZE, 4>::run(results);
    sweep_size_1d<vec_load_store<kittens::half>, SIZE, 12>::run(results);
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    sweep_size_1d<vec_fp8_load_store<kittens::fp8e4m3>, SIZE, 1>::run(results);
    sweep_size_1d<vec_fp8_load_store<kittens::fp8e4m3>, SIZE, 2>::run(results);
    sweep_size_1d<vec_fp8_load_store<kittens::fp8e4m3>, SIZE, 4>::run(results);
    sweep_size_1d<vec_fp8_load_store<kittens::fp8e4m3>, SIZE, 12>::run(results);
    sweep_size_1d<vec_fp8_load_store<kittens::fp8e5m2>, SIZE, 1>::run(results);
    sweep_size_1d<vec_fp8_load_store<kittens::fp8e5m2>, SIZE, 2>::run(results);
    sweep_size_1d<vec_fp8_load_store<kittens::fp8e5m2>, SIZE, 4>::run(results);
    sweep_size_1d<vec_fp8_load_store<kittens::fp8e5m2>, SIZE, 12>::run(results);
#endif

#if defined(KITTENS_BLACKWELL)
    sweep_size_1d<vec_fp8_load_store<kittens::fp8e8m0>, SIZE, 1>::run(results);
    sweep_size_1d<vec_fp8_load_store<kittens::fp8e8m0>, SIZE, 2>::run(results);
    sweep_size_1d<vec_fp8_load_store<kittens::fp8e8m0>, SIZE, 4>::run(results);
    sweep_size_1d<vec_fp8_load_store<kittens::fp8e8m0>, SIZE, 12>::run(results);
#endif

    sweep_size_1d<vec_async_load_store<float>, SIZE, 1>::run(results);
    sweep_size_1d<vec_async_load_store<float>, SIZE, 2>::run(results);
    sweep_size_1d<vec_async_load_store<float>, SIZE, 4>::run(results);
    sweep_size_1d<vec_async_load_store<float>, SIZE, 12>::run(results);
    sweep_size_1d<vec_async_load_store<kittens::bf16>, SIZE, 1>::run(results);
    sweep_size_1d<vec_async_load_store<kittens::bf16>, SIZE, 2>::run(results);
    sweep_size_1d<vec_async_load_store<kittens::bf16>, SIZE, 4>::run(results);
    sweep_size_1d<vec_async_load_store<kittens::bf16>, SIZE, 12>::run(results);
    sweep_size_1d<vec_async_load_store<kittens::half>, SIZE, 1>::run(results);
    sweep_size_1d<vec_async_load_store<kittens::half>, SIZE, 2>::run(results);
    sweep_size_1d<vec_async_load_store<kittens::half>, SIZE, 4>::run(results);
    sweep_size_1d<vec_async_load_store<kittens::half>, SIZE, 12>::run(results);
#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    sweep_size_1d<vec_fp8_async_load_store<kittens::fp8e4m3>, SIZE, 1>::run(results);
    sweep_size_1d<vec_fp8_async_load_store<kittens::fp8e4m3>, SIZE, 2>::run(results);
    sweep_size_1d<vec_fp8_async_load_store<kittens::fp8e4m3>, SIZE, 4>::run(results);
    sweep_size_1d<vec_fp8_async_load_store<kittens::fp8e4m3>, SIZE, 12>::run(results);
    sweep_size_1d<vec_fp8_async_load_store<kittens::fp8e5m2>, SIZE, 1>::run(results);
    sweep_size_1d<vec_fp8_async_load_store<kittens::fp8e5m2>, SIZE, 2>::run(results);
    sweep_size_1d<vec_fp8_async_load_store<kittens::fp8e5m2>, SIZE, 4>::run(results);
    sweep_size_1d<vec_fp8_async_load_store<kittens::fp8e5m2>, SIZE, 12>::run(results);
#endif

#if defined(KITTENS_BLACKWELL)
    sweep_size_1d<vec_fp8_async_load_store<kittens::fp8e8m0>, SIZE, 1>::run(results);
    sweep_size_1d<vec_fp8_async_load_store<kittens::fp8e8m0>, SIZE, 2>::run(results);
    sweep_size_1d<vec_fp8_async_load_store<kittens::fp8e8m0>, SIZE, 4>::run(results);
    sweep_size_1d<vec_fp8_async_load_store<kittens::fp8e8m0>, SIZE, 12>::run(results);
#endif
    std::cout << std::endl;
}

#endif