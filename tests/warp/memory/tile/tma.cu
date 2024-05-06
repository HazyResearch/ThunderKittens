#include "tma.cuh"

#ifdef TEST_WARP_MEMORY_TILE_TMA

struct test_load { // load with TMA, write out normally
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>;
    static inline const std::string test_identifier = "tma_load";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L>
    __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st_bf<H, W, L> (&shared_tile)[2][2] = al.allocate<kittens::st_bf<H, W, L>, 2, 2>();
        
        __shared__ kittens::tma::barrier smem_barrier; 
        kittens::tma::init_barrier<typeof(shared_tile[0][0]), 2, 2>(smem_barrier);
        for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
            kittens::tma::load_async(shared_tile[i][j], tma_desc_input, smem_barrier, i, j);
        }
        kittens::tma::arrive_and_wait(smem_barrier, 0);

        kittens::store(output, shared_tile[0][0], 2*W*16);
        kittens::store(output + shared_tile[0][0].cols, shared_tile[0][1], 2*W*16);
        kittens::store(output + 2*shared_tile[0][0].num_elements, shared_tile[1][0], 2*W*16);
        kittens::store(output + 2*shared_tile[0][0].num_elements + shared_tile[0][0].cols, shared_tile[1][1], 2*W*16);
    }
};
struct test_store { // load normally, store with TMA
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>;
    static inline const std::string test_identifier = "tma_store";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L>
    __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::st_bf<H, W, L> (&shared_tile)[2][2] = al.allocate<kittens::st_bf<H, W, L>, 2, 2>();
        kittens::load(shared_tile[0][0], input, 2*W*16);
        kittens::load(shared_tile[0][1], input + shared_tile[0][0].cols, 2*W*16);
        kittens::load(shared_tile[1][0], input + 2*shared_tile[0][0].num_elements, 2*W*16);
        kittens::load(shared_tile[1][1], input + 2*shared_tile[0][0].num_elements + shared_tile[0][0].cols, 2*W*16);
        __syncwarp();
        for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) {
            kittens::tma::store_async(tma_desc_output, shared_tile[i][j], i, j);
        }
        kittens::tma::store_commit_group();
        kittens::tma::store_async_wait<0>();
    }
};

template<typename Ker, int H, int W, int NW, typename... args>
static __global__ void tma_global_wrapper_2d(const kittens::bf16 *input, kittens::bf16 *output, CUtensorMap* tma_desc_input, CUtensorMap* tma_desc_output) {
    Ker::template device_func<H, W, NW, args...>(input, output, tma_desc_input, tma_desc_output);
}
template<typename test, int H, int W, int NUM_WORKERS, kittens::ducks::st_layout::all L, typename... args>
struct tma_wrapper_2d {
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS, L, args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, L, args...>::value) {
            constexpr int SIZE = H*W*256 * 2*2; // 2*2 for additional TMA dimensions
            // initialize
            kittens::bf16 *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize<initializers::ARANGE>(&d_i, &d_o, i_ref, o_ref);
            // initialize TMA descriptors
            CUtensorMap *i_desc = kittens::tma::allocate_and_create_tensor_map<kittens::st_bf<H, W, L>, 2, 2>(d_i);
            CUtensorMap *o_desc = kittens::tma::allocate_and_create_tensor_map<kittens::st_bf<H, W, L>, 2, 2>(d_o);
            // run kernel
            cudaFuncSetAttribute(
                tma_global_wrapper_2d<test, H, W, NUM_WORKERS, L, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            tma_global_wrapper_2d<test, H, W, NUM_WORKERS, L, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(d_i, d_o, i_desc, o_desc);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, L, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, 2*W*16);
            cudaFree(i_desc);
            cudaFree(o_desc);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
using tma_sweep_size_2d = loop_h<tma_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
struct tma_sweep_st_layout_size_2d {
    static void run(test_data &results) {
        tma_sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::naive, args...>::run(results);
        tma_sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::swizzle, args...>::run(results);
        tma_sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::wgmma_swizzle, args...>::run(results);
        tma_sweep_size_2d<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::wgmma_interleave, args...>::run(results);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, typename... args>
using tma_sweep_st_layout_size_2d_warp = tma_sweep_st_layout_size_2d<test, MAX_H, MAX_W, 1, args...>;

void warp::memory::tile::tma::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/tma tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    tma_sweep_st_layout_size_2d_warp<test_load,  SIZE, SIZE>::run(results);
    tma_sweep_st_layout_size_2d_warp<test_store, SIZE, SIZE>::run(results);
}

#endif