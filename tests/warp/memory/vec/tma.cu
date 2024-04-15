#include "tma.cuh"

#ifdef TEST_WARP_MEMORY_VEC_TMA

struct test_load { // load with TMA, write out normally
    template<int S, int NW> using valid = std::bool_constant<NW == 1 && S<=64 && S%4==0>; // S%4 ensures alignment
    static inline const std::string test_identifier = "tma_load_vec";
    template<int S, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW>
    __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_allocator al((int*)&__shm[0]); 
        kittens::row_vec<kittens::st_bf<S, S>> (&shared_vec)[4] = al.allocate<kittens::row_vec<kittens::st_bf<S, S>>, 4>();
        
        __shared__ kittens::tma::barrier smem_barrier; 
        kittens::tma::init_barrier<typeof(shared_vec[0]), 4>(smem_barrier);
        for(int i = 0; i < 4; i++) {
            kittens::tma::load_async(shared_vec[i], tma_desc_input, smem_barrier, i);
        }
        kittens::tma::arrive_and_wait(smem_barrier, 0);
        kittens::store(output, shared_vec[0]);
        kittens::store(output + shared_vec[0].length, shared_vec[1]);
        kittens::store(output + 2*shared_vec[0].length, shared_vec[2]);
        kittens::store(output + 3*shared_vec[0].length, shared_vec[3]);
    }
};
struct test_store { // load normally, store with TMA
    template<int S, int NW> using valid = std::bool_constant<NW == 1 && S<=64 && S%4==0>; // S%4 ensures alignment
    static inline const std::string test_identifier = "tma_store_vec";
    template<int S, int NW> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW>
    __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_allocator al((int*)&__shm[0]); 
        kittens::row_vec<kittens::st_bf<S, S>> (&shared_vec)[4] = al.allocate<kittens::row_vec<kittens::st_bf<S, S>>, 4>();
        
        kittens::load(shared_vec[0], input);
        kittens::load(shared_vec[1], input + shared_vec[0].length);
        kittens::load(shared_vec[2], input + 2*shared_vec[0].length);
        kittens::load(shared_vec[3], input + 3*shared_vec[0].length);
        __syncwarp();
        for(int i = 0; i < 4; i++) {
            kittens::tma::store_async(tma_desc_output, shared_vec[i], i);
        }
        kittens::tma::store_commit_group();
        kittens::tma::store_async_wait<0>();
    }
};

template<typename Ker, int S, int NW, typename... args>
static __global__ void tma_global_wrapper_1d(const kittens::bf16 *input, kittens::bf16 *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
    Ker::template device_func<S, NW, args...>(input, output, tma_desc_input, tma_desc_output);
}
template<typename test, int S, int NUM_WORKERS, typename... args>
struct tma_wrapper_1d {
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<S,NUM_WORKERS, args...>(test::test_identifier);
        if constexpr (test::template valid<S, NUM_WORKERS, args...>::value) {
            constexpr int SIZE = S*16 * 4; // 4 for additional TMA dimension
            // initialize
            kittens::bf16 *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // initialize TMA descriptors
            CUtensorMap *i_desc = kittens::tma::allocate_and_create_tensor_map<kittens::row_vec<kittens::st_bf<S, S>>, 4>(d_i);
            CUtensorMap *o_desc = kittens::tma::allocate_and_create_tensor_map<kittens::row_vec<kittens::st_bf<S, S>>, 4>(d_o);
            // run kernel
            cudaFuncSetAttribute(
                tma_global_wrapper_1d<test, S, NUM_WORKERS, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            tma_global_wrapper_1d<test, S, NUM_WORKERS, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(d_i, d_o, i_desc, o_desc);
            // fill in correct results on cpu
            test::template host_func<S, NUM_WORKERS, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, S*16);
            cudaFree(i_desc);
            cudaFree(o_desc);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_S=8, typename... args>
using tma_sweep_size_1d_warp = loop_s<tma_wrapper_1d, test, MAX_S, 1, MAX_S, args...>;

void warp::memory::vec::tma::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/tma tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    tma_sweep_size_1d_warp<test_load,  SIZE>::run(results);
    tma_sweep_size_1d_warp<test_store, SIZE>::run(results);
}

#endif