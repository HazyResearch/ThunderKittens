#include "reductions.cuh"

#ifdef TEST_WARP_SHARED_VEC_REDUCTIONS

template<typename T>
struct vec_norm {
    using dtype = T;
    template<int S, int NW>
    using valid = std::bool_constant<NW == 1 && S<=64 && sizeof(dtype) != 1>; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<T, kittens::bf16> ? "shared_vec_norm_gmem=bf16" :
                                                      std::is_same_v<T, kittens::half> ? "shared_vec_norm_gmem=half" :
                                                                                         "shared_vec_norm_gmem=float";
    template<int S, int NW, gl_t GL>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // turns out to get the numerics right in bf16 you have to actually simulate the reduction tree :/
        kittens::bf16 sum[32] = __float2bfloat16(0.f);
        if constexpr (S > 1) {
            for(int i = 0; i < 32; i++) sum[i] = __float2bfloat16(abs(i_ref[i]));
            for(int i = 32; i < o_ref.size(); i++) sum[i%32] += __float2bfloat16(abs(i_ref[i]));
            // now reduce first step
            for(int i = 0; i < 16; i++) sum[i] += sum[i+16];
        }
        else {
            for(int i = 0; i < 16; i++) sum[i] = __float2bfloat16(abs(i_ref[i]));
        }
        for(int i = 0; i < 8; i++) sum[i] += sum[i+8];
        for(int i = 0; i < 4; i++) sum[i] += sum[i+4];
        for(int i = 0; i < 2; i++) sum[i] += sum[i+2];
        sum[0] += sum[1];
        sum[0] += __float2bfloat16(1.f);
        for(int i = 0; i < o_ref.size(); i++) {
            kittens::bf16 o = __float2bfloat16(i_ref[i]) / sum[0];
            o_ref[i] = __bfloat162float(o);
        }
    }
    template<int S, int NW, gl_t GL>
    __device__ static void device_func(const GL &input, const GL &output) {
        extern __shared__ kittens::alignment_dummy __shm[];
        kittens::shared_allocator al((int*)&__shm[0]); 
        kittens::col_vec<kittens::st<dtype, 16*S, 16*S>> &vec    = al.allocate<kittens::col_vec<kittens::st<dtype, 16*S, 16*S>>>();
        kittens::col_vec<kittens::st<dtype, 16*S, 16*S>> &absvec = al.allocate<kittens::col_vec<kittens::st<dtype, 16*S, 16*S>>>();
        kittens::warp::load(vec, input, {});
        kittens::abs(absvec, vec);
        dtype f = kittens::base_types::constants<dtype>::one();
        kittens::sum(f, absvec, f);
        kittens::div(vec, vec, f);
        kittens::warp::store(output, vec, {});
    }
};

void group::shared::vec::reductions::tests(test_data &results) {
    std::cout << " ----- Starting ops/group/shared/vec/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_gmem_type_1d_warp<vec_norm, SIZE>::run(results);
    std::cout << std::endl;
}

#endif