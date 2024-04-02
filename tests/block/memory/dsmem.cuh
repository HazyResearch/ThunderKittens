#include "testing_flags.cuh"

#ifdef TEST_BLOCK_MEMORY_DSMEM

#include "testing_commons.cuh"

namespace block {
namespace memory {
namespace dsmem {

constexpr int cluster_size = 16; 

struct nextneighbor {
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> using valid = std::bool_constant<H%4 == 0 && NW == 8 &&
        (!std::is_same_v<L, kittens::ducks::st_layout::tma_swizzle> || W == 1 || W == 2 || W == 4) && W*H<=64>;
    static inline const std::string test_identifier = "block_shared_nextneighbor";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {

        for (int i = 0; i < 256 * H * W * NW; i++) {
            o_ref[i] = cluster_size - 1;
        }
        for (int i = 256 * H * W * NW; i < 256 * H * W * NW * cluster_size; i++) {
            o_ref[i] = static_cast<int>(i_ref[i] - 1) % cluster_size;
        }

    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L> 
    __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        using namespace kittens;

        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator<16> al((int*)&__shm[0]); 

        kittens::st_bf<H, W, L> (&input_tile)[NW] = al.allocate<kittens::st_bf<H, W, L>, NW>();
        kittens::st_bf<H, W, L> (&output_tile)[NW] = al.allocate<kittens::st_bf<H, W, L>, NW>();

        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        unsigned int cs = cluster.num_blocks();
        unsigned int cluster_idx = blockIdx.x % cs;

        const bf16* block_start = input + (cluster_idx * (input_tile[0].num_elements * NW));

        auto block = cooperative_groups::this_thread_block();
        auto warpid = kittens::warpid(); 
        
        kittens::block<NW>::load(input_tile[warpid], block_start + (warpid * input_tile[0].num_elements), input_tile[0].cols);

        __shared__ uint64_t smem_barrier[1];
        constexpr int size_bytes = sizeof(bf16) * input_tile[0].num_elements * NW; 
        kittens::block<NW>::dsmem::init_barrier(smem_barrier[0], 1); 
        kittens::block<NW>::dsmem::set_barrier_bytes(smem_barrier[0], size_bytes);

        block.sync();
        cluster.sync();

        int neighbor_idx = (cluster_idx + 1) % cs;
        
        kittens::block<NW>::dsmem::tile_distribute_smem(output_tile[warpid], input_tile[warpid], cs, neighbor_idx, size_bytes, smem_barrier[0]);

        constexpr int kPhaseBit = 0;
        kittens::block<NW>::dsmem::distribution_wait(smem_barrier[0], kPhaseBit);

        cluster.sync();

        // write out the results from output_tile to global memory
        bf16* output_block_start = output + (cluster_idx * output_tile[0].num_elements * NW);
        kittens::block<NW>::store(output_block_start + (warpid * output_tile[0].num_elements), output_tile[warpid], output_tile[0].cols);
    }
};

// ----- DSMEM Wrapper -----
template<typename Ker, int H, int W, int NW, typename... args>
static __global__ void __cluster_dims__(cluster_size, 1, 1) dsmem_wrapper(const kittens::bf16 *input, kittens::bf16 *output) {
    Ker::template device_func<H, W, NW, args...>(input, output);
}

template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct wrapper_dsmem {
    static void run(test_data& results) {
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {

            constexpr int tile_elements  = 256 * H * W * NUM_WORKERS; 
            constexpr int total_elements = 256 * H * W * NUM_WORKERS * cluster_size; 

            // initialize
            kittens::bf16 *d_i, *d_o;
            std::vector<float> i_ref(total_elements);
            std::vector<float> o_ref(total_elements);

            for (int i = 0; i < total_elements; i++) {
                i_ref[i] = i/tile_elements;
            }

            initialize<initializers::NONE>(&d_i, &d_o, i_ref, o_ref);

            // run kernel
            cudaFuncSetAttribute(
                dsmem_wrapper<test, H, W, NUM_WORKERS, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            cudaFuncSetAttribute(
                dsmem_wrapper<test, H, W, NUM_WORKERS, args...>,
                cudaFuncAttributeNonPortableClusterSizeAllowed, 
                cluster_size
            );
            dsmem_wrapper<test, H, W, NUM_WORKERS, args...><<<cluster_size, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(d_i, d_o);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, args...>(i_ref, o_ref);

            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*16);
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int H, int W, typename... args> using wrapper_dsmem_block = wrapper_dsmem<test, H, W, 8, args...>;

template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using sweep_size_dsmem       = loop_h<wrapper_dsmem, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args>                    using sweep_size_dsmem_block = sweep_size_dsmem<test, MAX_H, MAX_W, 8, args...>;

// Loop over st_layouts too, since this is needed by a bunch of tests.
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args>
struct sweep_st_layout_dsmem {
    static void run(test_data &results) {
        sweep_size_dsmem<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::naive, args...>::run(results);
        sweep_size_dsmem<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::tma_swizzle, args...>::run(results);
        sweep_size_dsmem<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::xor_swizzle, args...>::run(results);
        sweep_size_dsmem<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::wgmma_row_0b, args...>::run(results);
        sweep_size_dsmem<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::wgmma_row_32b, args...>::run(results);
        sweep_size_dsmem<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::wgmma_col_t_0b, args...>::run(results);
        sweep_size_dsmem<test, MAX_H, MAX_W, NUM_WORKERS, kittens::ducks::st_layout::wgmma_col_t_32b, args...>::run(results);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using sweep_st_layout_dsmem_block = sweep_st_layout_dsmem<test, MAX_H, MAX_W, 8, args...>;

void tests(test_data &results);

}
}
}

#endif