#include "complex_mma_fp32_bf16.cuh"

#ifdef TEST_GROUP_COMPLEX_WGMMA_MMA_FP32_BF16

struct test_cmplx_mma_AB_fp32_bf16 {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<NW == 4 && H==4 && (2*W*H+W*K::value+H*K::value)<=256>;
    static inline const std::string test_identifier = "complex_wgmma_mma_AB_fp32_bf16";
    template<int H, int W, int NW, typename GTL_A, typename GTL_B, typename GTL_C, typename _K>
     __host__ static void host_func(
        const std::vector<float> &re_i_ref, const std::vector<float> &im_i_ref,
        std::vector<float> &re_o_ref, std::vector<float> &im_o_ref) {
        
        // ac (reals)
        constexpr int K = _K::value;
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += re_i_ref[i*16*K + k]*re_i_ref[(256*H*K) + k*16*W + j];
                }
                re_o_ref[i*16*W + j] = sum;
            }
        }

        // bd (imags)
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += im_i_ref[i*16*K + k]*im_i_ref[(256*H*K) + k*16*W + j];
                }
                re_o_ref[i*16*W + j] -= sum;
            }
        }

        // ad
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += re_i_ref[i*16*K + k]*im_i_ref[(256*H*K) + k*16*W + j];
                }
                im_o_ref[i*16*W + j] = sum;
            }
        }

        // bc
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += im_i_ref[i*16*K + k]*re_i_ref[(256*H*K) + k*16*W + j];
                }
                im_o_ref[i*16*W + j] += sum;
            }
        }

    }
    template<int H, int W, int NW, typename CGL1, typename CGL2, typename CGL3, typename _K>
    __device__ static void device_func(const CGL1 &A, const CGL2 &B, const CGL3 &C) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::cst_bf<16*H, 16*K> &a = al.allocate<kittens::cst_bf<16*H, 16*K>>();
        kittens::cst_bf<16*K, 16*W> &b = al.allocate<kittens::cst_bf<16*K, 16*W>>();
        kittens::crt_fl<16, 16*W> c;
        kittens::warpgroup::load_async(a, A, {});
        kittens::warpgroup::load_async(b, B, {});
        kittens::warpgroup::load_async_wait();
        kittens::warpgroup::mm_AB(c, a, b);
        kittens::warpgroup::mma_async_wait();
        kittens::warpgroup::store(C, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*K::value>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*K::value, 16*W>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*W>;
};
struct test_cmplx_mma_ABt_fp32_bf16 {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<NW == 4 && H==4 && (2*W*H+W*K::value+H*K::value)<=256>; // this is warp-level
    static inline const std::string test_identifier = "complex_wgmma_mma_ABt_fp32_bf16";
    template<int H, int W, int NW, typename GTL_A, typename GTL_B, typename GTL_C, typename _K>
    __host__ static void host_func(const std::vector<float> &re_i_ref, const std::vector<float> &im_i_ref,
        std::vector<float> &re_o_ref, std::vector<float> &im_o_ref) {
        
        constexpr int K = _K::value;

        // ac (reals)
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += re_i_ref[i*K*16+k]*re_i_ref[256*K*H + j*K*16+k];
                }
                re_o_ref[i*W*16+j] = sum;
            }
        }

        // bd (imags)
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += im_i_ref[i*K*16+k]*im_i_ref[256*K*H + j*K*16+k];
                }
                re_o_ref[i*W*16+j] -= sum;
            }
        }

        // ad
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += re_i_ref[i*K*16+k]*im_i_ref[256*K*H + j*K*16+k];
                }
                im_o_ref[i*W*16+j] = sum;
            }
        }

        // bc
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += im_i_ref[i*K*16+k]*re_i_ref[256*K*H + j*K*16+k];
                }
                im_o_ref[i*W*16+j] += sum;
            }
        }
    }
    template<int H, int W, int NW, typename GTL_A, typename GTL_B, typename GTL_C, typename _K>
    __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, GTL_C &c_output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::cst_bf<16*H, 16*K> &a = al.allocate<kittens::cst_bf<16*H, 16*K>>();
        kittens::cst_bf<16*W, 16*K> &b = al.allocate<kittens::cst_bf<16*W, 16*K>>();
        kittens::crt_fl<16, 16*W> c;
        kittens::warpgroup::load_async(a, a_input, {});
        kittens::warpgroup::load_async(b, b_input, {});
        kittens::warpgroup::load_async_wait();
        kittens::warpgroup::mm_ABt(c, a, b);
        kittens::warpgroup::mma_async_wait();
        kittens::warpgroup::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*K::value>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*W, 16*K::value>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*W>;
};
struct test_cmplx_mma_AtB_fp32_bf16 {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<NW == 4 && H==4 && (2*W*H+W*K::value+H*K::value)<=256>; // this is warp-level
    static inline const std::string test_identifier = "complex_wgmma_mma_AtB_fp32_bf16";
    template<int H, int W, int NW, typename GTL_A, typename GTL_B, typename GTL_C, typename _K>
     __host__ static void host_func(const std::vector<float> &re_i_ref, const std::vector<float> &im_i_ref,
        std::vector<float> &re_o_ref, std::vector<float> &im_o_ref) {
        constexpr int K = _K::value;

        // ac
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += re_i_ref[k*16*H + i]*re_i_ref[(256*H*K) + k*16*W + j];
                }
                re_o_ref[i*16*W + j] = sum;
            }
        }

        // bd (imags)
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += im_i_ref[k*16*H + i]*im_i_ref[(256*H*K) + k*16*W + j];
                }
                re_o_ref[i*16*W + j] -= sum;
            }
        }

        // ad
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += re_i_ref[k*16*H + i]*im_i_ref[(256*H*K) + k*16*W + j];
                }
                im_o_ref[i*16*W + j] = sum;
            }
        }

        // bc
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += im_i_ref[k*16*H + i]*re_i_ref[(256*H*K) + k*16*W + j];
                }
                im_o_ref[i*16*W + j] += sum;
            }
        }
    }
    template<int H, int W, int NW, typename GTL_A, typename GTL_B, typename GTL_C, typename _K>
    __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, GTL_C &c_output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::cst_bf<16*K, 16*H> &a = al.allocate<kittens::cst_bf<16*K, 16*H>>();
        kittens::cst_bf<16*K, 16*W> &b = al.allocate<kittens::cst_bf<16*K, 16*W>>();
        kittens::crt_fl<16, 16*W> c;
        kittens::warpgroup::load_async(a, a_input, {});
        kittens::warpgroup::load_async(b, b_input, {});
        kittens::warpgroup::load_async_wait();
        kittens::warpgroup::mm_AtB(c, a, b);
        kittens::warpgroup::mma_async_wait();
        kittens::warpgroup::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*K::value, 16*H>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*K::value, 16*W>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*W>;
};
struct test_cmplx_mma_AtBt_fp32_bf16 {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<NW == 4 && H==4 && (2*W*H+W*K::value+H*K::value)<=256>; // this is warp-level
    static inline const std::string test_identifier = "complex_wgmma_mma_AtBt_fp32_bf16";
    template<int H, int W, int NW, typename GTL_A, typename GTL_B, typename GTL_C, typename _K>
    __host__ static void host_func(const std::vector<float> &re_i_ref, const std::vector<float> &im_i_ref,
        std::vector<float> &re_o_ref, std::vector<float> &im_o_ref) {
        constexpr int K = _K::value;

        // ac (reals)
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += re_i_ref[k*16*H + i]*re_i_ref[256*K*H + j*K*16+k];
                }
                re_o_ref[i*W*16+j] = sum;
            }
        }

        // bd (imags)
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += im_i_ref[k*16*H + i]*im_i_ref[256*K*H + j*K*16+k];
                }
                re_o_ref[i*W*16+j] -= sum;
            }
        }

        // ad
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += re_i_ref[k*16*H + i]*im_i_ref[256*K*H + j*K*16+k];
                }
                im_o_ref[i*W*16+j] = sum;
            }
        }

        // bc
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += im_i_ref[k*16*H + i]*re_i_ref[256*K*H + j*K*16+k];
                }
                im_o_ref[i*W*16+j] += sum;
            }
        }
    }
    template<int H, int W, int NW, typename GTL_A, typename GTL_B, typename GTL_C, typename _K>
    __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, GTL_C &c_output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::cst_bf<16*K, 16*H> &a = al.allocate<kittens::cst_bf<16*K, 16*H>>();
        kittens::cst_bf<16*W, 16*K> &b = al.allocate<kittens::cst_bf<16*W, 16*K>>();
        kittens::crt_fl<16, 16*W> c;
        kittens::warpgroup::load_async(a, a_input, {});
        kittens::warpgroup::load_async(b, b_input, {});
        kittens::warpgroup::load_async_wait();
        kittens::warpgroup::mm_AtBt(c, a, b);
        kittens::warpgroup::mma_async_wait();
        kittens::warpgroup::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1,  16*K::value, 16*H>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1,  16*W, 16*K::value>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1,  16*H, 16*W>;
};

// register+shared version
struct reg_cmplx_test_mma_AB_fp32_bf16 {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<NW == 4 && H%4==0 && (2*W*H+W*K::value+H*K::value)<=256>;
    static inline const std::string test_identifier = "complex_wgmma_reg_mma_AB_fp32_bf16";
    template<int H, int W, int NW, typename GTL_A, typename GTL_B, typename GTL_C, typename _K>
     __host__ static void host_func(const std::vector<float> &re_i_ref, const std::vector<float> &im_i_ref,
        std::vector<float> &re_o_ref, std::vector<float> &im_o_ref) {
        constexpr int K = _K::value;

        // ac (reals)
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += re_i_ref[i*16*K + k]*re_i_ref[(256*H*K) + k*16*W + j];
                }
                re_o_ref[i*16*W + j] = sum;
            }
        }

        // bd (imags)
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += im_i_ref[i*16*K + k]*im_i_ref[(256*H*K) + k*16*W + j];
                }
                re_o_ref[i*16*W + j] -= sum;
            }
        }

        // ad
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += re_i_ref[i*16*K + k]*im_i_ref[(256*H*K) + k*16*W + j];
                }
                im_o_ref[i*16*W + j] = sum;
            }
        }

        // bc
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += im_i_ref[i*16*K + k]*re_i_ref[(256*H*K) + k*16*W + j];
                }
                im_o_ref[i*16*W + j] += sum;
            }
        }
    }
    template<int H, int W, int NW, typename GTL_A, typename GTL_B, typename GTL_C, typename _K>
    __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, GTL_C &c_output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::cst_bf<16*H, 16*K> &a = al.allocate<kittens::cst_bf<16*H, 16*K>>();
        kittens::cst_bf<16*K, 16*W> &b = al.allocate<kittens::cst_bf<16*K, 16*W>>();
        kittens::crt_bf<4*H, 16*K> a_reg;
        kittens::crt_fl<4*H, 16*W> c;
        kittens::warpgroup::load_async(a, a_input, {});
        kittens::warpgroup::load_async(b, b_input, {});
        kittens::warpgroup::load_async_wait();
        kittens::warpgroup::load(a_reg, a);
        kittens::warpgroup::mma_fence(c);
        kittens::warpgroup::mm_AB(c, a_reg, b);
        kittens::warpgroup::mma_commit_group();
        kittens::warpgroup::mma_async_wait();
        kittens::warpgroup::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*K::value>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*K::value, 16*W>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*W>;
};
struct reg_cmplx_test_mma_ABt_fp32_bf16 {
    template<int H, int W, int NW, typename K>
    using valid = std::bool_constant<NW == 4 && H%4==0 && (2*W*H+W*K::value+H*K::value)<=256>; // this is warp-level
    static inline const std::string test_identifier = "complex_wgmma_reg_mma_ABt_fp32_bf16";
    template<int H, int W, int NW, typename GTL_A, typename GTL_B, typename GTL_C, typename _K>
    __host__ static void host_func(const std::vector<float> &re_i_ref, const std::vector<float> &im_i_ref,
        std::vector<float> &re_o_ref, std::vector<float> &im_o_ref) {
        constexpr int K = _K::value;
        
        // ac (reals)
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += re_i_ref[i*K*16+k]*re_i_ref[256*K*H + j*K*16+k];
                }
                re_o_ref[i*W*16+j] = sum;
            }
        }

        // bd (imags)
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += im_i_ref[i*16*K + k]*im_i_ref[256*K*H + j*K*16+k];
                }
                re_o_ref[i*16*W + j] -= sum;
            }
        }

        // ad
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += re_i_ref[i*16*K + k]*im_i_ref[256*K*H + j*K*16+k];
                }
                im_o_ref[i*16*W + j] = sum;
            }
        }

        // bc
        for(int i = 0; i < H*16; i++) {
            for(int j = 0; j < W*16; j++) {
                float sum = 0;
                for(int k = 0; k < K*16; k++) {
                    sum += im_i_ref[i*16*K + k]*re_i_ref[256*K*H + j*K*16+k];
                }
                im_o_ref[i*16*W + j] += sum;
            }
        }
    }
    template<int H, int W, int NW, typename GTL_A, typename GTL_B, typename GTL_C, typename _K>
    __device__ static void device_func(const GTL_A &a_input, const GTL_B &b_input, GTL_C &c_output) {
        constexpr int K = _K::value;
        extern __shared__ kittens::alignment_dummy __shm[]; // this is the CUDA shared memory
        kittens::tma_swizzle_allocator al((int*)&__shm[0]); 
        kittens::cst_bf<16*H, 16*K> &a = al.allocate<kittens::cst_bf<16*H, 16*K>>();
        kittens::cst_bf<16*W, 16*K> &b = al.allocate<kittens::cst_bf<16*W, 16*K>>();
        kittens::crt_bf<4*H, 16*K> a_reg;
        kittens::crt_fl<4*H, 16*W> c;
        kittens::warpgroup::load_async(a, a_input, {});
        kittens::warpgroup::load_async(b, b_input, {});
        kittens::warpgroup::load_async_wait();
        kittens::warpgroup::load(a_reg, a);
        kittens::warpgroup::mma_fence(c);
        kittens::warpgroup::mm_ABt(c, a_reg, b);
        kittens::warpgroup::mma_commit_group();
        kittens::warpgroup::mma_async_wait();
        kittens::warpgroup::store(c_output, c, {});
    }
    template<int H, int W, typename K> using make_a_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*K::value>;
    template<int H, int W, typename K> using make_b_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*W, 16*K::value>;
    template<int H, int W, typename K> using make_c_layout = typename kittens::gl<kittens::bf16, 1, 1, 16*H, 16*W>;
};

// Due to the strange sizes instantiated, we need a custom base wrapper here
template<typename Ker, typename T, int H, int W, int NW, typename CGL1, typename CGL2, typename CGL3, typename... args>
static __global__ void global_cmplx_mma_wrapper_2d(const CGL1 a_input, const CGL2 b_input, CGL3 c_output) {
    Ker::template device_func<H, W, NW, CGL1, CGL2, CGL3, args...>(a_input, b_input, c_output);
}


template<typename test, int H, int W, int NUM_WORKERS, typename _K, typename... args>
struct mma_wrapper_2d {
    static void run(test_data& results) {
        using namespace kittens;
        constexpr int K = _K::value;
        test_info this_result;
        this_result.label = generate_test_name<H,W,NUM_WORKERS,_K,args...>(test::test_identifier);
        if constexpr (test::template valid<H, W, NUM_WORKERS, _K, args...>::value) {
            // initialize
            bf16 *d_re_i, *d_im_i;
            bf16 *d_re_o, *d_im_o;
            std::vector<float> re_i_ref((H+W)*K*256);
            std::vector<float> im_i_ref((H+W)*K*256);
            std::vector<float> re_o_ref(H*W*256);
            std::vector<float> im_o_ref(H*W*256);
            initialize(&d_re_i, &d_re_o, re_i_ref, re_o_ref);
            initialize<bf16, initializers::RANDOM, 24>(&d_im_i, &d_im_o, im_i_ref, im_o_ref);

            // make descriptors
            using GTL_A = test::template make_a_layout<H, W, _K>; using CGTL_A = kittens::cgl<GTL_A>; // complex-valued globals
            using GTL_B = test::template make_b_layout<H, W, _K>; using CGTL_B = kittens::cgl<GTL_B>;
            using GTL_C = test::template make_c_layout<H, W, _K>; using CGTL_C = kittens::cgl<GTL_C>;
            
            GTL_A a_input_real (d_re_i,  nullptr, nullptr, nullptr, nullptr);
            GTL_B b_input_real (d_re_i + H*K*256, nullptr, nullptr, nullptr, nullptr);
            GTL_C c_output_real(d_re_o, nullptr, nullptr, nullptr, nullptr);

            GTL_A a_input_imag (d_im_i,           nullptr, nullptr, nullptr, nullptr);
            GTL_B b_input_imag (d_im_i + H*K*256, nullptr, nullptr, nullptr, nullptr);
            GTL_C c_output_imag(d_im_o,           nullptr, nullptr, nullptr, nullptr);

            CGTL_A a_input(a_input_real, a_input_imag);
            CGTL_B b_input(b_input_real, b_input_imag);
            CGTL_C c_output(c_output_real, c_output_imag);
            
            // run kernel
            cudaFuncSetAttribute(
                global_cmplx_mma_wrapper_2d<test, kittens::bf16, H, W, NUM_WORKERS, CGTL_A, CGTL_B, CGTL_C, _K, args...>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                kittens::MAX_SHARED_MEMORY
            );
            global_cmplx_mma_wrapper_2d<test, kittens::bf16, H, W, NUM_WORKERS, CGTL_A, CGTL_B, CGTL_C, _K, args...><<<1, NUM_WORKERS*32, kittens::MAX_SHARED_MEMORY>>>(a_input, b_input, c_output);

            // fill in correct results on cpu
            test::template host_func<H, W, NUM_WORKERS, CGTL_A, CGTL_B, CGTL_C, _K, args...>(re_i_ref, im_i_ref, re_o_ref, im_o_ref);
            // check and cleanup
            test_result re_result = validate(d_re_i, d_re_o, re_i_ref, re_o_ref, this_result.label + "_real", W*16, 0.02); // mma's sometimes produce small errors. this appears to be hardware.
            test_result im_result = validate(d_im_i, d_im_o, im_i_ref, im_o_ref, this_result.label + "_imag", W*16, 0.02);
            if (re_result == test_result::PASSED && im_result == test_result::PASSED) {
                // TODO change back
                this_result.result = test_result::PASSED;
            } else {
                this_result.result = test_result::FAILED;
            }
        }
        else {
            this_result.result = test_result::INVALID;
        }
        results.push_back(this_result);
    }
};
template<typename test, int H, int MAX_W, int NUM_WORKERS=1, typename... args> using mma_sweep_width = loop_w<mma_wrapper_2d, test, H, MAX_W, NUM_WORKERS, H, MAX_W, args...>;
template<typename test, int MAX_W, typename... args> using mma_sweep_width_warpgroup = mma_sweep_width<test, 4, MAX_W, 4, args...>;
template<typename test, int MAX_W, typename... args> using mma_sweep_width_warpgroup_doubleheight = mma_sweep_width<test, 8, MAX_W, 4, args...>;

// If 1 and 3 work, the others likely will too.
using I1_t = std::integral_constant<int, 1>;
using I3_t = std::integral_constant<int, 3>;
using I5_t = std::integral_constant<int, 5>;
void group::wgmma::complex::complex_mma_fp32_bf16::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warpgroup/wgmma/complex/complex_mma_fp32_bf16 tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_1 ? 2 :
                         INTENSITY_2 ? 4 :
                         INTENSITY_3 ? 8 :
                         INTENSITY_4 ? 16 : -1;

    // shared+shared->reg

    mma_sweep_width_warpgroup<test_cmplx_mma_AB_fp32_bf16,   SIZE, I1_t>::run(results);
    mma_sweep_width_warpgroup<test_cmplx_mma_ABt_fp32_bf16,  SIZE, I1_t>::run(results);
    mma_sweep_width_warpgroup<test_cmplx_mma_AtB_fp32_bf16,  SIZE, I1_t>::run(results);
    mma_sweep_width_warpgroup<test_cmplx_mma_AtBt_fp32_bf16, SIZE, I1_t>::run(results);

    mma_sweep_width_warpgroup<test_cmplx_mma_AB_fp32_bf16,   SIZE, I3_t>::run(results);
    mma_sweep_width_warpgroup<test_cmplx_mma_ABt_fp32_bf16,  SIZE, I3_t>::run(results);
    mma_sweep_width_warpgroup<test_cmplx_mma_AtB_fp32_bf16,  SIZE, I3_t>::run(results);
    mma_sweep_width_warpgroup<test_cmplx_mma_AtBt_fp32_bf16, SIZE, I3_t>::run(results);

    mma_sweep_width_warpgroup<test_cmplx_mma_AB_fp32_bf16,   SIZE, I5_t>::run(results);
    mma_sweep_width_warpgroup<test_cmplx_mma_ABt_fp32_bf16,  SIZE, I5_t>::run(results);
    mma_sweep_width_warpgroup<test_cmplx_mma_AtB_fp32_bf16,  SIZE, I5_t>::run(results); 
    mma_sweep_width_warpgroup<test_cmplx_mma_AtBt_fp32_bf16, SIZE, I5_t>::run(results);

    // register+shared->reg
    mma_sweep_width_warpgroup<reg_cmplx_test_mma_AB_fp32_bf16,  SIZE, I1_t>::run(results);
    mma_sweep_width_warpgroup<reg_cmplx_test_mma_ABt_fp32_bf16, SIZE, I1_t>::run(results);
    mma_sweep_width_warpgroup_doubleheight<reg_cmplx_test_mma_AB_fp32_bf16,  SIZE, I1_t>::run(results);
    mma_sweep_width_warpgroup_doubleheight<reg_cmplx_test_mma_ABt_fp32_bf16, SIZE, I1_t>::run(results);

    mma_sweep_width_warpgroup<reg_cmplx_test_mma_AB_fp32_bf16,  SIZE, I3_t>::run(results);
    mma_sweep_width_warpgroup<reg_cmplx_test_mma_ABt_fp32_bf16, SIZE, I3_t>::run(results);
    mma_sweep_width_warpgroup_doubleheight<reg_cmplx_test_mma_AB_fp32_bf16,  SIZE, I3_t>::run(results);
    mma_sweep_width_warpgroup_doubleheight<reg_cmplx_test_mma_ABt_fp32_bf16, SIZE, I3_t>::run(results);

    mma_sweep_width_warpgroup<reg_cmplx_test_mma_AB_fp32_bf16,  SIZE, I5_t>::run(results);
    mma_sweep_width_warpgroup<reg_cmplx_test_mma_ABt_fp32_bf16, SIZE, I5_t>::run(results);
    mma_sweep_width_warpgroup_doubleheight<reg_cmplx_test_mma_AB_fp32_bf16,  SIZE, I5_t>::run(results);
    mma_sweep_width_warpgroup_doubleheight<reg_cmplx_test_mma_ABt_fp32_bf16, SIZE, I5_t>::run(results);
}

#endif