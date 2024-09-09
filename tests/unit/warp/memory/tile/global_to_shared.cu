#include "global_to_shared.cuh"

#ifdef TEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED

template<typename Ker, typename T, int H, int W, int NW, kittens::ducks::gl::all GL, typename... args>
static __global__ void g2s_global_wrapper_2d(const GL input, GL output) {
    Ker::template device_func<H, W, NW, GL, args...>(input, output);
}
template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct g2s_wrapper_2d {
    using dtype = gmem_dtype<test>; // defaults to bf16 in global memory if the test doesn't specify.
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {
            constexpr int B = 3, D = 1, R = 4, C = 5;
            constexpr int SIZE = H*W*256 * B * D * R * C;
            // initialize
            dtype *d_i, *d_o;
            std::vector<float> i_ref(SIZE);
            std::vector<float> o_ref(SIZE);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GL = typename kittens::gl<dtype, -1, D, -1, 16*C>;
            GL input(d_i, B, nullptr, 16*R, nullptr);
            GL output(d_o, B, nullptr, 16*R, nullptr);
            // run kernel
            cudaFuncSetAttribute(
                global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GL, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            global_wrapper_2d<test, dtype, H, W, NUM_WORKERS, GL, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(input, output);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, GL, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*16);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using g2s_sweep_size_2d = loop_h<g2s_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using g2s_sweep_size_2d_warp = g2s_sweep_size_2d<test, MAX_H, MAX_W, 1, args...>;


template<typename T>
struct st_load_store {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_loadstore_gmem=half" :
                                                                                         "shared_loadstore_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL> __device__ static void device_func(const GL input, GL output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::st<T, H, W> &shared_tile = al.allocate<kittens::st<T, H, W>>();
        for(int i = 0; i < input.batch; i++) for(int j = 0; j < input.depth; j++) for(int k = 0; k < input.rows/shared_tile.rows; k++) for(int l = 0; l < input.cols/shared_tile.cols; l++) {
            kittens::load(shared_tile, input, {i, j, k, l});
            kittens::store(output, shared_tile, {i, j, k, l});
        }
    }
};

template<typename T>
struct st_load_store_async {
    using dtype = T;
    template<int H, int W, int NW> using valid = std::bool_constant<NW == 1 && W*H<=64>;
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_loadstore_async_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_loadstore_async_gmem=half" :
                                                                                         "shared_loadstore_async_gmem=float";
    template<int H, int W, int NW, kittens::ducks::gl::all GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::gl::all GL> __device__ static void device_func(const GL input, GL output) {
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 
        kittens::st<T, H, W> &shared_tile = al.allocate<kittens::st<T, H, W>>();

        __syncthreads();
        
        for(int i = 0; i < input.batch; i++) for(int j = 0; j < input.depth; j++) for(int k = 0; k < input.rows/shared_tile.rows; k++) for(int l = 0; l < input.cols/shared_tile.cols; l++) {
            kittens::load_async(shared_tile, input, {i, j, k, l});
            kittens::load_async_wait();
            kittens::store(output, shared_tile, {i, j, k, l});
        }
    }
};

void warp::memory::tile::global_to_shared::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/tile/global_to_shared tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    sweep_gmem_type_2d_warp<st_load_store, SIZE, SIZE>::run(results);
    sweep_gmem_type_2d_warp<st_load_store_async, SIZE, SIZE>::run(results);
}

#endif