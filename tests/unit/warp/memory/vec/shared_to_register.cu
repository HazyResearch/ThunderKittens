#include "shared_to_register.cuh"

#ifdef TEST_WARP_MEMORY_VEC_SHARED_TO_REGISTER

template<typename T>
struct vec_load_store {
    using dtype = T;
    template<int S, int NW, kittens::ducks::rv_layout::all L> using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<dtype, kittens::bf16> ? "shared_reg_vec_loadstore_gmem=bf16" :
                                                      std::is_same_v<dtype, kittens::half> ? "shared_reg_vec_loadstore_gmem=half" :
                                                                                             "shared_reg_vec_loadstore_gmem=float";
    template<int S, int NW, kittens::ducks::gl::all GL, kittens::ducks::rv_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < i_ref.size(); i++) o_ref[i] = i_ref[i] + 1.f; // just a dummy op to prevent optimization away
    }
    template<int S, int NW, kittens::ducks::gl::all GL, kittens::ducks::rv_layout::all L> __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::sv<dtype, 16*S> &shared_vec = al.allocate<kittens::sv<dtype, 16*S>>();
        kittens::rv_fl<16*S, L> reg_vec;
        kittens::load(shared_vec, input, {});
        __syncthreads();
        kittens::load(reg_vec, shared_vec);
        kittens::add(reg_vec, reg_vec, float(1.)); // TODO: CHANGE HOST TOO
        kittens::store(shared_vec, reg_vec);
        __syncthreads();
        kittens::store(output, shared_vec, {});
    }
};

void warp::memory::vec::shared_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/shared_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_gmem_type_1d_warp<vec_load_store, SIZE, kittens::ducks::rv_layout::naive>::run(results);
    sweep_gmem_type_1d_warp<vec_load_store, SIZE, kittens::ducks::rv_layout::ortho>::run(results);
    sweep_gmem_type_1d_warp<vec_load_store, SIZE, kittens::ducks::rv_layout::align>::run(results);
}

#endif