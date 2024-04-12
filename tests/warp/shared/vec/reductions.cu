#include "reductions.cuh"

#ifdef TEST_WARP_SHARED_VEC_REDUCTIONS

struct vec_norm {
    template<int S, int NW>
    using valid = std::bool_constant<NW == 1 && S<=64>; // this is warp-level
    static inline const std::string test_identifier = "shared_vec_norm";
    template<int S, int NW>
    __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        // turns out to get the numerics right in bf16 you have to actually simulate the reduction tree :/
        kittens::bf16 sum[32] = __float2bfloat16(0.f);
        if(S > 1) {
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
    template<int S, int NW>
    __device__ static void device_func(const kittens::bf16 *input, kittens::bf16 *output) {
        __shared__ kittens::col_vec<kittens::st_bf<S, S>> vec;
        __shared__ kittens::col_vec<kittens::st_bf<S, S>> absvec;
        kittens::load(vec, input);
        kittens::abs(absvec, vec);
        kittens::bf16 f = __float2bfloat16(1.f);
        kittens::sum(f, absvec, f);
        kittens::div(vec, vec, f);
        kittens::store(output, vec);
    }
};

void warp::shared::vec::reductions::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/shared/vec/reductions tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    sweep_size_1d_warp<vec_norm, SIZE>::run(results);
}

#endif