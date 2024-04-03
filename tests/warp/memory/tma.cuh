#include "testing_flags.cuh"

#ifdef TEST_WARP_MEMORY_TMA

#include "testing_commons.cuh"

namespace warp {
namespace memory {
namespace tma {

template<kittens::ducks::st_layout::all layout, int TMA_HEIGHT, int TMA_WIDTH, int workers>
__global__ void
test_tmaload_ker(const kittens::bf16 *input, kittens::bf16 *output, CUtensorMap* tma_desc_input) {
    using namespace kittens;
    auto warpid = kittens::warpid(); 
    auto lane   = kittens::laneid(); 

    CUtensorMap* input_tma_descriptor  = tma_desc_input;

    extern __shared__ int __shm[];
    shared_allocator<1024> al((int*)&__shm[0]); 

    // using worker_type = kittens::st_bf<TMA_HEIGHT, TMA_WIDTH, layout>;
    kittens::st_bf<TMA_HEIGHT, TMA_WIDTH, layout> (&input_tile)[workers]  = al.allocate<kittens::st_bf<TMA_HEIGHT, TMA_WIDTH, layout>, workers>();

    auto block = cooperative_groups::this_thread_block();
    __shared__ uint64_t smem_barrier[workers]; 
    constexpr int size_bytes = sizeof(bf16) * input_tile[warpid].num_elements;

    kittens::tma::prefetch(input_tile[warpid], input_tma_descriptor, 0);
    kittens::tma::init_barrier<typeof(input_tile[warpid])>(smem_barrier[warpid], 1); 

    block.sync();

    for (int tile_idx = 0; tile_idx < 4; tile_idx++) {
        kittens::tma::load_async(input_tile[warpid], input_tma_descriptor, tile_idx, smem_barrier[warpid]);
        // load(input_tile, input, TMA_WIDTH * 16);

        int kPhaseBit = 0; 
        kittens::tma::arrive_and_wait(smem_barrier[warpid], kPhaseBit);

        kittens::tma::init_barrier<typeof(input_tile[warpid])>(smem_barrier[warpid], 1); 
        kittens::store(output + (input_tile[warpid].num_elements * tile_idx), input_tile[warpid], TMA_WIDTH * 16); 
        // tma::store_async(output_tma_descriptor, input_tile, tile_idx);
    }
}

template<kittens::ducks::st_layout::all layout, int TMA_HEIGHT, int TMA_WIDTH, int workers>
__global__ void
test_tmastore_ker(const kittens::bf16 *input, kittens::bf16 *output, CUtensorMap* tma_desc_output) {
    using namespace kittens;
    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();

    CUtensorMap* output_tma_descriptor = tma_desc_output;

    extern __shared__ int __shm[];
    kittens::shared_allocator<1024> al((int*)&__shm[0]); 
    kittens::st_bf<TMA_HEIGHT, TMA_WIDTH, layout> (&input_tile)[workers] = al.allocate<kittens::st_bf<TMA_HEIGHT, TMA_WIDTH, layout>, workers>();

    auto block = cooperative_groups::this_thread_block();
    __shared__ uint64_t smem_barrier; 
    constexpr int size_bytes = sizeof(bf16) * input_tile[warpid].num_elements;

    block.sync();

    for (int tile_idx = 0; tile_idx < 4; tile_idx++) {
        // tma::load_async(input_tile, input_tma_descriptor, tile_idx, smem_barrier);
        kittens::load(input_tile[warpid], input + (input_tile[warpid].num_elements * tile_idx), TMA_WIDTH * 16); 

        // store(output, input_tile, TMA_WIDTH * 16);
        kittens::tma::store_async(output_tma_descriptor, input_tile[warpid], tile_idx);
    
        kittens::tma::store_commit_group();

        kittens::tma::store_async_wait<0>();
    }
}

template<kittens::ducks::st_layout::all layout, bool store_test, int TMA_HEIGHT, int TMA_WIDTH, int workers=1>
void test_tma(test_data &results) {
    using namespace kittens;
    // initailize
    bf16 *d_i, *d_o;
    std::vector<float> i_ref(TMA_HEIGHT * TMA_WIDTH * 1024);
    for(int i = 0; i < TMA_HEIGHT * TMA_WIDTH * 1024; i++) i_ref[i] = float(i);
    std::vector<float> o_ref(TMA_HEIGHT * TMA_WIDTH * 1024); 
    initialize(&d_i, &d_o, i_ref, o_ref);

    CUtensorMap tma_desc_input = {};
    kittens::tma::create_tensor_map<st_bf<TMA_HEIGHT, TMA_WIDTH, layout>, 4>(&tma_desc_input, d_i); 

    CUtensorMap tma_desc_output = {};
    kittens::tma::create_tensor_map<st_bf<TMA_HEIGHT, TMA_WIDTH, layout>, 4>(&tma_desc_output, d_o);

    CUtensorMap* tma_desc_input_d;
    CUtensorMap* tma_desc_output_d;

    cudaMalloc(&tma_desc_input_d,                     sizeof(CUtensorMap));
    cudaMalloc(&tma_desc_output_d,                    sizeof(CUtensorMap));
    cudaMemcpy(tma_desc_input_d,    &tma_desc_input,  sizeof(CUtensorMap), cudaMemcpyHostToDevice);
    cudaMemcpy(tma_desc_output_d,   &tma_desc_output, sizeof(CUtensorMap), cudaMemcpyHostToDevice);

    unsigned long mem_size = 100000; 

    // run kernel
    if constexpr (!store_test) {
        cudaFuncSetAttribute(test_tmaload_ker<layout, TMA_HEIGHT, TMA_WIDTH, workers>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        CudaCheckError();

        test_tmaload_ker<layout, TMA_HEIGHT, TMA_WIDTH, workers><<<1, workers * 32, mem_size>>>(d_i, d_o, tma_desc_input_d);
        CudaCheckError();
    }
    else {
        cudaFuncSetAttribute(test_tmastore_ker<layout, TMA_HEIGHT, TMA_WIDTH, workers>, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
        CudaCheckError();

        test_tmastore_ker<layout, TMA_HEIGHT, TMA_WIDTH, workers><<<1, workers * 32, mem_size>>>(d_i, d_o, tma_desc_output_d);
        CudaCheckError();
    }

    // output identical to input
    for(int i = 0; i < TMA_HEIGHT * TMA_WIDTH * 1024; i++) o_ref[i] = i_ref[i];
    std::string load_str = "tma_["+layout_name<layout>()+(store_test ? "]_[store]" : "]_[load]");
    load_str += "_["+std::to_string(TMA_HEIGHT)+"x"+std::to_string(TMA_WIDTH); 
    load_str += "]_["+std::to_string(workers)+"]";

    test_info info;
    info.result = validate(d_i, d_o, i_ref, o_ref, load_str, 16 * TMA_WIDTH);
    info.label = load_str;

    results.push_back(info);
}

template<kittens::ducks::st_layout::all layout, bool store_test, bool swizzled=false>
int tma_dim_test(test_data &results) {
    using namespace kittens;
    int failures = 0; 

    
    test_tma<layout, store_test, 1, 1>(results);
    test_tma<layout, store_test, 1, 2>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 1, 3>(results);
    test_tma<layout, store_test, 1, 4>(results);

    test_tma<layout, store_test, 1, 1, 2>(results); 
    test_tma<layout, store_test, 1, 2, 2>(results); 
    if constexpr (!swizzled) test_tma<layout, store_test, 1, 3, 2>(results); 
    test_tma<layout, store_test, 1, 4, 2>(results); 

    test_tma<layout, store_test, 1, 1, 3>(results);
    test_tma<layout, store_test, 1, 2, 3>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 1, 3, 3>(results);
    test_tma<layout, store_test, 1, 4, 3>(results);

    test_tma<layout, store_test, 1, 1, 4>(results);
    test_tma<layout, store_test, 1, 2, 4>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 1, 3, 4>(results);
    test_tma<layout, store_test, 1, 4, 4>(results);

    test_tma<layout, store_test, 2, 1>(results);
    test_tma<layout, store_test, 2, 2>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 2, 3>(results);
    test_tma<layout, store_test, 2, 4>(results);

    test_tma<layout, store_test, 2, 1, 2>(results);
    test_tma<layout, store_test, 2, 2, 2>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 2, 3, 2>(results);
    test_tma<layout, store_test, 2, 4, 2>(results);

    test_tma<layout, store_test, 2, 1, 3>(results);
    test_tma<layout, store_test, 2, 2, 3>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 2, 3, 3>(results);
    test_tma<layout, store_test, 2, 4, 3>(results);

    test_tma<layout, store_test, 2, 1, 4>(results);
    test_tma<layout, store_test, 2, 2, 4>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 2, 3, 4>(results);
    test_tma<layout, store_test, 2, 4, 4>(results);

    test_tma<layout, store_test, 3, 1>(results);
    test_tma<layout, store_test, 3, 2>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 3, 3>(results);
    test_tma<layout, store_test, 3, 4>(results);

    test_tma<layout, store_test, 3, 1, 2>(results);
    test_tma<layout, store_test, 3, 2, 2>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 3, 3, 2>(results);
    test_tma<layout, store_test, 3, 4, 2>(results);

    test_tma<layout, store_test, 3, 1, 3>(results);
    test_tma<layout, store_test, 3, 2, 3>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 3, 3, 3>(results);
    test_tma<layout, store_test, 3, 4, 3>(results);

    test_tma<layout, store_test, 3, 1, 4>(results);
    test_tma<layout, store_test, 3, 2, 4>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 3, 3, 4>(results);
    test_tma<layout, store_test, 3, 4, 4>(results);

    test_tma<layout, store_test, 4, 1>(results);
    test_tma<layout, store_test, 4, 2>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 4, 3>(results);
    test_tma<layout, store_test, 4, 4>(results);

    test_tma<layout, store_test, 4, 1, 2>(results);
    test_tma<layout, store_test, 4, 2, 2>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 4, 3, 2>(results);
    test_tma<layout, store_test, 4, 4, 2>(results);

    test_tma<layout, store_test, 4, 1, 3>(results);
    test_tma<layout, store_test, 4, 2, 3>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 4, 3, 3>(results);
    test_tma<layout, store_test, 4, 4, 3>(results);

    test_tma<layout, store_test, 4, 1, 4>(results);
    test_tma<layout, store_test, 4, 2, 4>(results);
    if constexpr (!swizzled) test_tma<layout, store_test, 4, 3, 4>(results);
    test_tma<layout, store_test, 4, 4, 4>(results);
    
    return failures;
}

void tests(test_data &results);
}
}
}

#endif