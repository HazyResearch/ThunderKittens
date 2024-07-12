#include "shared_to_register.cuh"

#ifdef TEST_GROUP_MEMORY_VEC_SHARED_TO_REGISTER

template<typename T>
struct vec_load_store {
    using dtype = T;
    template<int S, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<S%NW==0 && S<=64>; // this is group-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_shared_reg_vec_loadstore=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_shared_reg_vec_loadstore=half" :
                                                                                         "group_shared_reg_vec_loadstore=float";
    template<int S, int NW, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) o_ref[i] = i_ref[i] + 1.f; // just a dummy op to prevent optimization away
    }
    template<int S, int NW, kittens::ducks::rt_layout::all L> __device__ static void device_func(const dtype *input, dtype *output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::col_vec<kittens::st<dtype, S, S>> &shared_vec = al.allocate<kittens::col_vec<kittens::st<dtype, S, S>>>();
        kittens::col_vec<kittens::rt<dtype, S/NW, S/NW, L>> reg_vec;
        G::load(shared_vec, input);
        __syncthreads();
        G::load(reg_vec, shared_vec);
        kittens::add(reg_vec, reg_vec, dtype(1.)); // TODO: CHANGE HOST TOO
        G::store(shared_vec, reg_vec);
        __syncthreads();
        G::store(output, shared_vec);
    }
};

void group::memory::vec::shared_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/memory/vec/shared_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_1d<vec_load_store, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);

    sweep_size_1d<vec_load_store, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);

    // 12 is a nice number because it is not a power of 2, so it has a good shot of breaking things.
    sweep_size_1d<vec_load_store, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_1d<vec_load_store, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
}

#endif