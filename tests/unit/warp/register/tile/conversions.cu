#include "conversions.cuh"

#ifdef TEST_WARP_REGISTER_TILE_CONVERSIONS

// Transpose happens to need its own wrapper, as it has a different shape input and output.
template<typename Ker, typename T, int H, int W, int NW, gl_t GTL_I, gl_t GTL_O, typename... args>
static __global__ void transpose_global_wrapper_2d(const GTL_I input, GTL_O output) {
    Ker::template device_func<H, W, NW, GTL_I, GTL_O, args...>(input, output);
}
template<typename test, int H, int W, int NUM_WORKERS, typename... args>
struct transpose_wrapper_2d {
    static void run(test_data& results) {
        using namespace kittens;
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, args...>::value) {
            // initialize
            kittens::bf16 *d_i, *d_o;
            std::vector<float> i_ref(H*W*256);
            std::vector<float> o_ref(H*W*256);
            initialize(&d_i, &d_o, i_ref, o_ref);
            // make descriptors
            using GTL_I = typename kittens::gl<kittens::bf16, 1, 1, H*16, W*16>;
            using GTL_O = typename kittens::gl<kittens::bf16, 1, 1, W*16, H*16>;
            GTL_I input (d_i, nullptr, nullptr, nullptr, nullptr);
            GTL_O output(d_o, nullptr, nullptr, nullptr, nullptr);
            // run kernel
            cudaFuncSetAttribute(
                transpose_global_wrapper_2d<test, kittens::bf16, H, W, NUM_WORKERS, GTL_I, GTL_O, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            transpose_global_wrapper_2d<test, kittens::bf16, H, W, NUM_WORKERS, GTL_I, GTL_O, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(input, output);
            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, GTL_I, GTL_O, args...>(i_ref, o_ref);
            // check and cleanup
            this_result.result = validate(d_i, d_o, i_ref, o_ref, this_result.label, W*16, 0.02); // mma's sometimes produce small errors. this appears to be hardware.
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int MAX_H=8, int MAX_W=8, int NUM_WORKERS=1, typename... args> using transpose_sweep_size = loop_h<transpose_wrapper_2d, test, MAX_H, MAX_W, NUM_WORKERS, MAX_H, args...>;
template<typename test, int MAX_H=8, int MAX_W=8, typename... args> using transpose_sweep_size_warp = transpose_sweep_size<test, MAX_H, MAX_W, 1, args...>;


struct test_swap_layout {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_swaplayout";
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt_bf<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        auto &reg_tile_other_layout = kittens::swap_layout_inplace(reg_tile);
        kittens::store(output, reg_tile_other_layout, {});
    }
};
struct test_transpose {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_transpose";
    template<int H, int W, int NW, gl_t GTL_I, gl_t GTL_O, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++)
            for(int j = 0; j < W*16; j++)
                o_ref[i+j*H*16] = i_ref[i*W*16+j];
    }
    template<int H, int W, int NW, gl_t GTL_I, gl_t GTL_O, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GTL_I input, GTL_O output) {
        kittens::rt_bf<16*H, 16*W, L> reg_tile;
        kittens::rt_bf<16*W, 16*H, L> reg_tile_transpose;
        kittens::load(reg_tile, input, {});
        kittens::transpose_sep(reg_tile_transpose, reg_tile);
        kittens::store(output, reg_tile_transpose, {});
    }
};
struct test_type_convert {
    template<int H, int W, int NW, typename T2, typename U2> using valid = std::bool_constant<NW == 1 && W*H<=32>; // this is warp-level
        static inline const std::string test_identifier = "reg_typeconvert";
        template<int H, int W, int NW, gl_t GL, typename T2, typename U2> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, gl_t GL, typename T2, typename U2> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt<U2, 16*H, 16*W> reg_tile_U2;
        kittens::rt<T2, 16*H, 16*W> reg_tile_T2;
        kittens::load(reg_tile_U2, input, {});
        kittens::copy(reg_tile_T2, reg_tile_U2);
        kittens::store(output, reg_tile_T2, {});
    }
};

#ifdef KITTENS_HOPPER
struct test_type_convert_typed { 
    template<int H, int W, int NW, typename T2, typename U2> using valid = std::bool_constant<NW == 1 && W*H<=32 && (
        ( ( 
            !std::is_same_v<kittens::fp8e4m3, T2> && !std::is_same_v<kittens::fp8e4m3, U2> &&
            !std::is_same_v<kittens::fp8e5m2, T2> && !std::is_same_v<kittens::fp8e5m2, U2>
        ) || W%2 == 0 )
    )>; // this is warp-level
        static inline const std::string test_identifier = "reg_typeconvert";
        template<int H, int W, int NW, gl_t GL, typename T2, typename U2> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int H, int W, int NW, gl_t GL, typename T2, typename U2> __device__ static void device_func(const GL input, const GL output) {
        if constexpr (std::is_same_v<U2, kittens::fp8e4m3> || std::is_same_v<U2, kittens::fp8e5m2>) { 
            // fp8 to float
            extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
            kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
            kittens::st<U2, 16*H, 16*W> &st_tile_U2 = al.allocate<kittens::st<U2, 16*H, 16*W>>(); 
            kittens::rt<float, 16*H, 16*W> reg_tile_float;                     
            kittens::rt<U2, 16*H, 16*W> reg_tile_U2;
            kittens::rt<T2, 16*H, 16*W> reg_tile_T2;

            kittens::load(reg_tile_float, input, {}); // due to lack of direct global to register fp8
            kittens::copy(reg_tile_U2, reg_tile_float);
            kittens::copy(reg_tile_T2, reg_tile_U2);
            kittens::store(output, reg_tile_T2, {});

        } else if constexpr (std::is_same_v<T2, kittens::fp8e4m3> || std::is_same_v<T2, kittens::fp8e5m2>) {
            // float to fp8
            // printf("float to fp8\n");
            extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
            kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
            kittens::st<kittens::bf16, 16*H, 16*W> &st_tile_bf = al.allocate<kittens::st<kittens::bf16, 16*H, 16*W>>();   
            kittens::st<U2, 16*H, 16*W> &st_tile_U2 = al.allocate<kittens::st<U2, 16*H, 16*W>>();    
            kittens::rt<T2, 16*H, 16*W> reg_tile_T2;
            kittens::rt<U2, 16*H, 16*W> reg_tile_U2;

            kittens::load(reg_tile_U2, input, {});
            kittens::copy(reg_tile_T2, reg_tile_U2);
            kittens::copy(reg_tile_U2, reg_tile_T2);
            kittens::store(st_tile_U2, reg_tile_U2);
            kittens::copy(st_tile_bf, st_tile_U2);
            kittens::store(output, st_tile_bf, {}); // leverage register to global conversions
        }
    }
};
#endif

struct test_subtile {
    template<int H, int W, int NW, typename ST_H> using valid = std::bool_constant<NW == 1 && (H%(ST_H::value))==0 && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_subtile";
    template<int H, int W, int NW, gl_t GL, typename ST_H> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++)
            for(int j = 0; j < W*16; j++)
                o_ref[i*W*16 + j] = i_ref[i*W*16 + j] + float(i/(ST_H::value*16));
    }
    template<int H, int W, int NW, gl_t GL, typename _ST_H> __device__ static void device_func(const GL input, const GL output) {
        constexpr int ST_H = _ST_H::value;
        kittens::rt_fl<16*H, 16*W> reg_tile;
        kittens::load(reg_tile, input, {});
        #pragma unroll
        for(int i = 0; i < H/ST_H; i++) {
            auto &ref = kittens::subtile_inplace<ST_H*16>(reg_tile, i);
            kittens::add(ref, ref, float(i));
        }
        kittens::store(output, reg_tile, {});
    }
};
struct test_make_causal {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_make_causal";
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for(int i = 0; i < H*16; i++)
            for(int j = 0; j < W*16; j++)
                o_ref[i*W*16 + j] = j<=i ? i_ref[i*W*16 + j] : 0;
    }
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::make_causal(reg_tile, reg_tile);
        kittens::store(output, reg_tile, {});
    }
};
struct test_tril {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_tril";
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // triangular lower, with diagonal starting at row_idx 4
        for(int i = 0; i < H*16; i++)
            for(int j = 0; j < W*16; j++)
                o_ref[i*W*16 + j] = j<=i+(4*H) ? i_ref[i*W*16 + j] : 0;
    }
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::tril(reg_tile, reg_tile, 4*H);
        kittens::store(output, reg_tile, {});
    }
};
struct test_triu {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_triu";
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // triangular upper, with diagonal starting at row_idx 4
        for(int i = 0; i < H*16; i++)
            for(int j = 0; j < W*16; j++)
                o_ref[i*W*16 + j] = j>=i+(4*H) ? i_ref[i*W*16 + j] : 0;
    }
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::triu(reg_tile, reg_tile, 4*H);
        kittens::store(output, reg_tile, {});
    }
};
struct test_right_fill {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_right_fill";
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // here, set everything to from and right of col_idx 8 is set to zero
        for(int i = 0; i < H*16; i++) 
            for(int j = 0; j < W*16; j++) 
            o_ref[i*W*16 + j] = (j < (8 * W)) ? i_ref[i*W*16 + j] : 0;
    }
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::right_fill(reg_tile, reg_tile, 8 * W);
        kittens::store(output, reg_tile, {});
    }
};
struct test_left_fill {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_left_fill";
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // here, set everything to from and left of col_idx 8 is set to zero
        for(int i = 0; i < H*16; i++) 
            for(int j = 0; j < W*16; j++) 
                o_ref[i*W*16 + j] = (j >= (8 * W)) ? i_ref[i*W*16 + j] : 0;
    }
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::left_fill(reg_tile, reg_tile, 8 * W);
        kittens::store(output, reg_tile, {});
    }
};
struct test_lower_fill {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_lower_fill";
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // here, set everything to from and lower of row_idx 8 is set to zero
        for(int i = 0; i < H*16; i++) 
            for(int j = 0; j < W*16; j++) 
                o_ref[i*W*16 + j] = (i < (8 * H)) ? i_ref[i*W*16 + j] : 0;
    }
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::lower_fill(reg_tile, reg_tile, 8 * H);
        kittens::store(output, reg_tile, {});
    }
};
struct test_upper_fill {
    template<int H, int W, int NW, kittens::ducks::rt_layout::all L> using valid = std::bool_constant<NW == 1 && H==W && W*H<=64>; // this is warp-level
    static inline const std::string test_identifier = "reg_upper_fill";
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // here, set everything to from and upper of row_idx 8 is set to zero
        for(int i = 0; i < H*16; i++) 
            for(int j = 0; j < W*16; j++) 
                o_ref[i*W*16 + j] = (i >= ((8 * H))) ? i_ref[i*W*16 + j] : 0;
    }
    template<int H, int W, int NW, gl_t GL, kittens::ducks::rt_layout::all L> __device__ static void device_func(const GL input, const GL output) {
        kittens::rt_fl<16*H, 16*W, L> reg_tile;
        kittens::load(reg_tile, input, {});
        kittens::upper_fill(reg_tile, reg_tile, 8 * H);
        kittens::store(output, reg_tile, {});
    }
};

void warp::reg::tile::conversions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/register/tile/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
    // sweep_size_2d_warp<test_swap_layout, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    // sweep_size_2d_warp<test_swap_layout, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);

    // transpose_sweep_size_warp<test_transpose, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    // transpose_sweep_size_warp<test_transpose, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);

    // sweep_size_2d_warp<test_type_convert, SIZE, SIZE, float, kittens::bf16>::run(results);
    // sweep_size_2d_warp<test_type_convert, SIZE, SIZE, kittens::bf16, float>::run(results);
    // sweep_size_2d_warp<test_type_convert, SIZE, SIZE, float, kittens::half>::run(results);
    // sweep_size_2d_warp<test_type_convert, SIZE, SIZE, kittens::half, float>::run(results);
    // sweep_size_2d_warp<test_type_convert, SIZE, SIZE, kittens::half, kittens::bf16>::run(results);
    // sweep_size_2d_warp<test_type_convert, SIZE, SIZE, kittens::bf16, kittens::half>::run(results);

    // #ifdef KITTENS_HOPPER
    // sweep_size_2d_warp<test_type_convert_typed, SIZE, SIZE, float, kittens::fp8e4m3>::run(results); // fp8 
    // sweep_size_2d_warp<test_type_convert_typed, SIZE, SIZE, kittens::fp8e4m3, float>::run(results);
    // sweep_size_2d_warp<test_type_convert_typed, SIZE, SIZE, float, kittens::fp8e5m2>::run(results); 
    // sweep_size_2d_warp<test_type_convert_typed, SIZE, SIZE, kittens::fp8e5m2, float>::run(results);
    // sweep_size_2d_warp<test_type_convert_typed, SIZE, SIZE, kittens::fp8e4m3, kittens::bf16>::run(results);
    // sweep_size_2d_warp<test_type_convert_typed, SIZE, SIZE, kittens::fp8e5m2, kittens::bf16>::run(results);
    // sweep_size_2d_warp<test_type_convert_typed, SIZE, SIZE, kittens::bf16, kittens::fp8e4m3>::run(results);
    // sweep_size_2d_warp<test_type_convert_typed, SIZE, SIZE, kittens::bf16, kittens::fp8e5m2>::run(results);
    // sweep_size_2d_warp<test_type_convert_typed, SIZE, SIZE, kittens::fp8e4m3, kittens::half>::run(results);
    // sweep_size_2d_warp<test_type_convert_typed, SIZE, SIZE, kittens::fp8e5m2, kittens::half>::run(results);
    // sweep_size_2d_warp<test_type_convert_typed, SIZE, SIZE, kittens::half, kittens::fp8e4m3>::run(results);
    // sweep_size_2d_warp<test_type_convert_typed, SIZE, SIZE, kittens::half, kittens::fp8e5m2>::run(results);
    // #endif

    // sweep_size_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 1>>::run(results);
    // sweep_size_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 2>>::run(results);
    // sweep_size_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 3>>::run(results);
    // sweep_size_2d_warp<test_subtile, SIZE, SIZE, std::integral_constant<int, 4>>::run(results);

    // sweep_size_2d_warp<test_right_fill, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    // sweep_size_2d_warp<test_right_fill, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    // sweep_size_2d_warp<test_left_fill,  SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    // sweep_size_2d_warp<test_left_fill,  SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    // sweep_size_2d_warp<test_lower_fill, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    // sweep_size_2d_warp<test_lower_fill, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    // sweep_size_2d_warp<test_upper_fill, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    // sweep_size_2d_warp<test_upper_fill, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);

    sweep_size_2d_warp<test_tril, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_triu, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    sweep_size_2d_warp<test_tril, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);
    sweep_size_2d_warp<test_triu, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results);

    sweep_size_2d_warp<test_make_causal, SIZE, SIZE, kittens::ducks::rt_layout::row>::run(results);
    // sweep_size_2d_warp<test_make_causal, SIZE, SIZE, kittens::ducks::rt_layout::col>::run(results); NOT YET SUPPORTED
}

#endif