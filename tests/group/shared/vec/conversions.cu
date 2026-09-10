#include "conversions.cuh"

#ifdef TEST_GROUP_SHARED_VEC_CONVERSIONS

struct vec_copy {
    template<int S, int NW> using valid = std::bool_constant<S%NW==0 && S<=64>; // this is group-level
    static inline const std::string test_identifier = "shared_vec_convert";
    template<int S, int NW, gl_t GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<int S, int NW, gl_t GL> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        __shared__ kittens::col_vec<kittens::st_bf<16*S, 16*S>> vec1;
        __shared__ kittens::col_vec<kittens::st_bf<16*S, 16*S>> vec2;
        G::load(vec1, input, {});
        G::sync(0);
        G::copy(vec2, vec1);
        G::sync(0);
        G::store(output, vec2, {});
    }
};

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)

struct vec_convert_fp8e4m3 {
    using dtype = kittens::bf16;
    template<int S, int NW> using valid = std::bool_constant<S%NW==0 && S<=64>;
    static inline const std::string test_identifier = "shared_vec_convert_e4m3_roundtrip";
    template<int S, int NW, gl_t GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int i = 0; i < (int)o_ref.size(); i++)
            o_ref[i] = float(kittens::fp8e4m3(i_ref[i]));
    }
    template<int S, int NW, gl_t GL> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator al((int*)&__shm[0]);
        kittens::sv_bf<16*S>      &sv_b1  = al.allocate<kittens::sv_bf<16*S>>();
        kittens::sv_fp8e4m3<16*S> &sv_fp8 = al.allocate<kittens::sv_fp8e4m3<16*S>>();
        kittens::sv_bf<16*S>      &sv_b2  = al.allocate<kittens::sv_bf<16*S>>();
        G::load(sv_b1, input, {});
        G::sync(0);
        G::copy(sv_fp8, sv_b1);
        G::sync(0);
        G::copy(sv_b2, sv_fp8);
        G::sync(0);
        G::store(output, sv_b2, {});
    }
};

struct vec_convert_fp8e5m2 {
    using dtype = kittens::bf16;
    template<int S, int NW> using valid = std::bool_constant<S%NW==0 && S<=64>;
    static inline const std::string test_identifier = "shared_vec_convert_e5m2_roundtrip";
    template<int S, int NW, gl_t GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int i = 0; i < (int)o_ref.size(); i++)
            o_ref[i] = float(kittens::fp8e5m2(i_ref[i]));
    }
    template<int S, int NW, gl_t GL> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator al((int*)&__shm[0]);
        kittens::sv_bf<16*S>      &sv_b1  = al.allocate<kittens::sv_bf<16*S>>();
        kittens::sv_fp8e5m2<16*S> &sv_fp8 = al.allocate<kittens::sv_fp8e5m2<16*S>>();
        kittens::sv_bf<16*S>      &sv_b2  = al.allocate<kittens::sv_bf<16*S>>();
        G::load(sv_b1, input, {});
        G::sync(0);
        G::copy(sv_fp8, sv_b1);
        G::sync(0);
        G::copy(sv_b2, sv_fp8);
        G::sync(0);
        G::store(output, sv_b2, {});
    }
};

#endif // KITTENS_HOPPER || KITTENS_BLACKWELL

#if defined(KITTENS_BLACKWELL)

struct vec_convert_fp8e8m0 {
    using dtype = kittens::bf16;
    template<int S, int NW> using valid = std::bool_constant<S%NW==0 && S<=64>;
    static inline const std::string test_identifier = "shared_vec_convert_e8m0_roundtrip";
    template<int S, int NW, gl_t GL> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        for (int i = 0; i < (int)o_ref.size(); i++)
            o_ref[i] = float(kittens::fp8e8m0(i_ref[i]));
    }
    template<int S, int NW, gl_t GL> __device__ static void device_func(const GL &input, const GL &output) {
        using G = kittens::group<NW>;
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator al((int*)&__shm[0]);
        kittens::sv_bf<16*S>      &sv_b1  = al.allocate<kittens::sv_bf<16*S>>();
        kittens::sv_fp8e8m0<16*S> &sv_fp8 = al.allocate<kittens::sv_fp8e8m0<16*S>>();
        kittens::sv_bf<16*S>      &sv_b2  = al.allocate<kittens::sv_bf<16*S>>();
        G::load(sv_b1, input, {});
        G::sync(0);
        G::copy(sv_fp8, sv_b1);
        G::sync(0);
        G::copy(sv_b2, sv_fp8);
        G::sync(0);
        G::store(output, sv_b2, {});
    }
};

#endif // KITTENS_BLACKWELL

void group::shared::vec::conversions::tests(test_data &results) {
    std::cout << " ----- Starting ops/group/shared/vec/conversions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  :
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;

    sweep_size_1d<vec_copy, SIZE, 2>::run(results);
    sweep_size_1d<vec_copy, SIZE, 4>::run(results);
    sweep_size_1d<vec_copy, SIZE, 12>::run(results);

#if defined(KITTENS_HOPPER) || defined(KITTENS_BLACKWELL)
    sweep_size_1d<vec_convert_fp8e4m3, SIZE, 1>::run(results);
    sweep_size_1d<vec_convert_fp8e4m3, SIZE, 2>::run(results);
    sweep_size_1d<vec_convert_fp8e4m3, SIZE, 4>::run(results);
    sweep_size_1d<vec_convert_fp8e4m3, SIZE, 12>::run(results);
    sweep_size_1d<vec_convert_fp8e5m2, SIZE, 1>::run(results);
    sweep_size_1d<vec_convert_fp8e5m2, SIZE, 2>::run(results);
    sweep_size_1d<vec_convert_fp8e5m2, SIZE, 4>::run(results);
    sweep_size_1d<vec_convert_fp8e5m2, SIZE, 12>::run(results);
#endif

#if defined(KITTENS_BLACKWELL)
    sweep_size_1d<vec_convert_fp8e8m0, SIZE, 1>::run(results);
    sweep_size_1d<vec_convert_fp8e8m0, SIZE, 2>::run(results);
    sweep_size_1d<vec_convert_fp8e8m0, SIZE, 4>::run(results);
    sweep_size_1d<vec_convert_fp8e8m0, SIZE, 12>::run(results);
#endif

    std::cout << std::endl;
}

#endif
