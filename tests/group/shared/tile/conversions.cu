#include "conversions.cuh"

#ifdef TEST_GROUP_SHARED_TILE_CONVERSIONS

struct test_swap_layout {
    template<int H, int W, int NW, kittens::ducks::st_layout::all L1, kittens::ducks::st_layout::all L2> using valid = std::bool_constant<H%NW==0 && W*H<=64 &&
        (!(std::is_same_v<L1, kittens::ducks::st_layout::xor_swizzle> || std::is_same_v<L2, kittens::ducks::st_layout::xor_swizzle>) || W == 1 || W == 2 || W == 4 || W == 8 || W == 16)>; // this is group-level
    static inline const std::string test_identifier = "shared_swaplayout";
    template<int H, int W, int NW, kittens::ducks::st_layout::all L1, kittens::ducks::st_layout::all L2> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, kittens::ducks::st_layout::all L1, kittens::ducks::st_layout::all L2> __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st_bf<H, W, L1> &t1 = al.allocate<kittens::st_bf<H, W, L1>>();
        kittens::st_bf<H, W, L2> &t2 = al.allocate<kittens::st_bf<H, W, L2>>();
        G::load(t2, input, W*16);
        __syncthreads();
        G::copy(t1, t2);
        __syncthreads();
        G::store(output, t1, W*16);
    }
};

void group::shared::tile::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/shared/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_st_layout_size_2d<test_swap_layout, SIZE, SIZE, 2, kittens::ducks::st_layout::naive>::run(results);
    sweep_st_layout_size_2d<test_swap_layout, SIZE, SIZE, 2, kittens::ducks::st_layout::xor_swizzle>::run(results);
    sweep_st_layout_size_2d<test_swap_layout, SIZE, SIZE, 2, kittens::ducks::st_layout::wgmma_row_0b>::run(results);
    sweep_st_layout_size_2d<test_swap_layout, SIZE, SIZE, 2, kittens::ducks::st_layout::wgmma_row_32b>::run(results);
    sweep_st_layout_size_2d<test_swap_layout, SIZE, SIZE, 2, kittens::ducks::st_layout::wgmma_col_t_0b>::run(results);
    sweep_st_layout_size_2d<test_swap_layout, SIZE, SIZE, 2, kittens::ducks::st_layout::wgmma_col_t_32b>::run(results);

    if constexpr (TEST_INTENSITY > 1) {

        sweep_st_layout_size_2d<test_swap_layout, SIZE, SIZE, 4, kittens::ducks::st_layout::naive>::run(results);
        sweep_st_layout_size_2d<test_swap_layout, SIZE, SIZE, 4, kittens::ducks::st_layout::xor_swizzle>::run(results);
        sweep_st_layout_size_2d<test_swap_layout, SIZE, SIZE, 4, kittens::ducks::st_layout::wgmma_row_0b>::run(results);
        sweep_st_layout_size_2d<test_swap_layout, SIZE, SIZE, 4, kittens::ducks::st_layout::wgmma_row_32b>::run(results);
        sweep_st_layout_size_2d<test_swap_layout, SIZE, SIZE, 4, kittens::ducks::st_layout::wgmma_col_t_0b>::run(results);
        sweep_st_layout_size_2d<test_swap_layout, SIZE, SIZE, 4, kittens::ducks::st_layout::wgmma_col_t_32b>::run(results);

        if constexpr (TEST_INTENSITY > 3) {

            sweep_st_layout_size_2d<test_swap_layout, 12, 4, 12, kittens::ducks::st_layout::naive>::run(results);
            sweep_st_layout_size_2d<test_swap_layout, 12, 4, 12, kittens::ducks::st_layout::xor_swizzle>::run(results);
            sweep_st_layout_size_2d<test_swap_layout, 12, 4, 12, kittens::ducks::st_layout::wgmma_row_0b>::run(results);
            sweep_st_layout_size_2d<test_swap_layout, 12, 4, 12, kittens::ducks::st_layout::wgmma_row_32b>::run(results);
            sweep_st_layout_size_2d<test_swap_layout, 12, 4, 12, kittens::ducks::st_layout::wgmma_col_t_0b>::run(results);
            sweep_st_layout_size_2d<test_swap_layout, 12, 4, 12, kittens::ducks::st_layout::wgmma_col_t_32b>::run(results);

        }
    }
}

#endif