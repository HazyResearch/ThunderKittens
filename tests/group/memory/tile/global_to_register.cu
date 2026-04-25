#include "global_to_register.cuh"

#ifdef TEST_GROUP_MEMORY_TILE_GLOBAL_TO_REGISTER

template<typename T>
struct group_load_store {
    using dtype = T;
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<H%NW==0 && W*H<=64>; // this is group-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "group_reg_loadstore_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "group_reg_loadstore_gmem=half" :
                                                      std::is_same_v<T, char> ? "group_reg_loadstore_gmem=char" :
                                                      std::is_same_v<T, unsigned char> ? "group_reg_loadstore_gmem=unsigned_char" :
                                                      std::is_same_v<T, short> ? "group_reg_loadstore_gmem=short" :
                                                      std::is_same_v<T, unsigned short> ? "group_reg_loadstore_gmem=unsigned_short" :
                                                      std::is_same_v<T, int> ? "group_reg_loadstore_gmem=int" :
                                                      std::is_same_v<T, uint> ? "group_reg_loadstore_gmem=uint" :
                                                      std::is_same_v<T, int64_t> ? "group_reg_loadstore_gmem=int64_t" :
                                                      std::is_same_v<T, uint64_t> ? "group_reg_loadstore_gmem=uint64_t" :
                                                      std::is_same_v<T, double> ? "group_reg_loadstore_gmem=double" :
                                                                                         "group_reg_loadstore_gmem=float";
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL &input, const GL &output) {
        if constexpr (kittens::ducks::base_types::T1<dtype>) {
            using G = kittens::group<NW>;
            kittens::rt<dtype, 16*H/NW, 16*W, L> reg_tile;
            G::load(reg_tile, input, {});
            G::store(output, reg_tile, {});
        }
        else {
            size_t numel = input.numel();
            for(size_t idx = blockIdx.x*blockDim.x + threadIdx.x; idx < numel; idx += blockDim.x*gridDim.x) {
                int c = idx % input.cols();
                int r = (idx / input.cols()) % input.rows();
                int d = (idx / (input.cols()*input.rows())) % input.depth();
                int b = idx / (input.cols()*input.rows()*input.depth());
                output[{b, d, r, c}] = input[{b, d, r, c}];
            }
        }
    }
};

void group::memory::tile::global_to_register::tests(test_data &results) {
    std::cout << " ----- Starting ops/group/memory/tile/global_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_2d<group_load_store<float>, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<float>, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<float>, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<float>, SIZE, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<float>, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<float>, SIZE, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<float>, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<float>, SIZE, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<kittens::bf16>, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<kittens::bf16>, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<kittens::bf16>, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<kittens::bf16>, SIZE, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<kittens::bf16>, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<kittens::bf16>, SIZE, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<kittens::bf16>, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<kittens::bf16>, SIZE, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<kittens::half>, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<kittens::half>, SIZE, SIZE, 1, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<kittens::half>, SIZE, SIZE, 2, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<kittens::half>, SIZE, SIZE, 2, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<kittens::half>, SIZE, SIZE, 4, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<kittens::half>, SIZE, SIZE, 4, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<kittens::half>, SIZE, SIZE, 12, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<kittens::half>, SIZE, SIZE, 12, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d<group_load_store<char>, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<unsigned char>, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<short>, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<unsigned short>, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<int>, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<uint>, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<int64_t>, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<uint64_t>, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d<group_load_store<double>, SIZE, SIZE, 1, kittens::ducks::rt_layout::row>::run(results);
    std::cout << std::endl;
}

#endif