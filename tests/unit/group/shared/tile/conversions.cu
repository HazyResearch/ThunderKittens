#include "conversions.cuh"

#ifdef TEST_GROUP_SHARED_TILE_CONVERSIONS

struct test_shared_copy {
    template<int H, int W, int NW> using valid = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = "shared_copy";
    template<int H, int W, int NW, gtl_t GTL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, gtl_t GTL> __device__ static void device_func(const GTL &input, GTL &output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::st_bf<H, W> &t1 = al.allocate<kittens::st_bf<H, W>>();
        kittens::st_bf<H, W> &t2 = al.allocate<kittens::st_bf<H, W>>();
        G::load(t2, input, {});
        G::sync();
        G::copy(t1, t2);
        G::sync();
        G::store(output, t1, {});
    }
};

void group::shared::tile::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/group/shared/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_2d<test_shared_copy, SIZE, SIZE, 2>::run(results);

    if constexpr (TEST_INTENSITY > 1) {

        sweep_size_2d<test_shared_copy, SIZE, SIZE, 4>::run(results);

        if constexpr (TEST_INTENSITY > 3) {

            sweep_size_2d<test_shared_copy, 12, 4, 12>::run(results);

        }
    }
}

#endif