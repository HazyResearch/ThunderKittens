#include "mma.cuh"

#ifdef TEST_GROUP_MMA_TENSOR_MMA

namespace {

template<typename T>
using accum_t = std::conditional_t<std::is_same_v<T, kittens::int8> || std::is_same_v<T, kittens::uint8>, int, float>;

template<typename T, bool TA, bool TB>
static void host_ref(const std::vector<T> &a, const std::vector<T> &b, std::vector<accum_t<T>> &o, bool acc) {
    constexpr int M = 128;
    constexpr int N = 64;
    constexpr int K = 32;
    for(int m = 0; m < M; m++) {
        for(int n = 0; n < N; n++) {
            accum_t<T> sum = 0;
            for(int k = 0; k < K; k++) {
                const int a_idx = TA ? k*M + m : m*K + k;
                const int b_idx = TB ? n*K + k : k*N + n;
                sum += accum_t<T>(float(a[a_idx])) * accum_t<T>(float(b[b_idx]));
            }
            o[m*N+n] = acc ? 2*sum : sum;
        }
    }
}

template<typename T>
static void fill_input(std::vector<T> &v) {
    for(int i = 0; i < v.size(); i++) {
        if constexpr (std::is_same_v<T, kittens::int8>) {
            v[i] = T((i % 5) - 2);
        }
        else {
            v[i] = T((i % 5) + 1);
        }
    }
}

template<bool TA, bool TB, bool ACC, kittens::ducks::tt::all D, typename A, typename B>
__device__ static inline void run_mma(D &d, const A &a, const B &b, kittens::semaphore &sem) {
    if constexpr (TA && TB) {
        if constexpr (ACC) kittens::group<4>::mma_AtBt(d, a, b, sem);
        else               kittens::group<4>::mm_AtBt (d, a, b, sem);
    }
    else if constexpr (TA) {
        if constexpr (ACC) kittens::group<4>::mma_AtB(d, a, b, sem);
        else               kittens::group<4>::mm_AtB (d, a, b, sem);
    }
    else if constexpr (TB) {
        if constexpr (ACC) kittens::group<4>::mma_ABt(d, a, b, sem);
        else               kittens::group<4>::mm_ABt (d, a, b, sem);
    }
    else {
        if constexpr (ACC) kittens::group<4>::mma_AB(d, a, b, sem);
        else               kittens::group<4>::mm_AB (d, a, b, sem);
    }
}

template<typename T, bool TS, bool TA, bool TB, bool ACC, kittens::ducks::gl::all GL_A, kittens::ducks::gl::all GL_B, kittens::ducks::gl::all GL_O>
__global__ void tcgen05_wrapper(const __grid_constant__ GL_A a_gl, const __grid_constant__ GL_B b_gl, const __grid_constant__ GL_O o_gl) {
    constexpr int M = 128;
    constexpr int N = 64;
    constexpr int K = 32;
    using G = kittens::group<4>;
    using O = accum_t<T>;
    using D_TT = kittens::tt<O, M, N>;
    using D_RT = kittens::rt<O, M/G::GROUP_WARPS, N>;
    using B_ST = kittens::st<T, TB ? N : K, TB ? K : N>;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::tma_swizzle_allocator al((int*)&__shm[0]);
    B_ST (&b_smem) = al.allocate<B_ST>();

    kittens::tensor_allocator<1, 1> tm_alloc{};
    D_TT d_tt;
    if constexpr (kittens::ducks::tt::full<D_TT>) {
        d_tt = tm_alloc.template allocate<D_TT>(0);
    }
    else {
        d_tt = tm_alloc.template allocate<D_TT>(0, 0);
    }

    __shared__ kittens::semaphore sem;
    kittens::warp::init_semaphore(sem, 0, 1);
    __syncthreads();

    G::load(b_smem, b_gl, {});
    __syncthreads();
    if constexpr (TS) {
        static_assert(!TA, "TMEM A cannot be transposed.");
        using A_TT = kittens::tt<T, M, K>;
        using A_RT = kittens::rt<T, M/G::GROUP_WARPS, K>;
        A_TT a_tt;
        if constexpr (kittens::ducks::tt::full<A_TT>) {
            a_tt = tm_alloc.template allocate<A_TT>(128);
        }
        else {
            a_tt = tm_alloc.template allocate<A_TT>(0, 128);
        }
        A_RT a_reg;
        if constexpr (std::is_same_v<T, kittens::fp8e4m3> || std::is_same_v<T, kittens::fp8e5m2>) {
            using A_ST = kittens::st<T, M, K>;
            A_ST (&a_smem) = al.allocate<A_ST>();
            G::load(a_smem, a_gl, {});
            __syncthreads();
            G::load(a_reg, a_smem);
        }
        else {
            G::load(a_reg, a_gl, {});
        }
        G::store_async(a_tt, a_reg);
        kittens::tensor_store_wait();
        __syncthreads();
        if constexpr (ACC) {
            run_mma<TA, TB, false>(d_tt, a_tt, b_smem, sem);
            kittens::wait(sem, 0);
            run_mma<TA, TB, true>(d_tt, a_tt, b_smem, sem);
            kittens::wait(sem, 1);
        }
        else {
            run_mma<TA, TB, false>(d_tt, a_tt, b_smem, sem);
            kittens::wait(sem, 0);
        }
    }
    else {
        using A_ST = kittens::st<T, TA ? K : M, TA ? M : K>;
        A_ST (&a_smem) = al.allocate<A_ST>();
        G::load(a_smem, a_gl, {});
        __syncthreads();
        if constexpr (ACC) {
            run_mma<TA, TB, false>(d_tt, a_smem, b_smem, sem);
            kittens::wait(sem, 0);
            run_mma<TA, TB, true>(d_tt, a_smem, b_smem, sem);
            kittens::wait(sem, 1);
        }
        else {
            run_mma<TA, TB, false>(d_tt, a_smem, b_smem, sem);
            kittens::wait(sem, 0);
        }
    }

    D_RT d_reg;
    G::load_async(d_reg, d_tt);
    kittens::tensor_load_wait();
    G::store(o_gl, d_reg, {});
}

template<typename T, bool TS, bool TA, bool TB, bool ACC>
static void run_one(test_data &results, const std::string &label) {
    constexpr int M = 128;
    constexpr int N = 64;
    constexpr int K = 32;
    using O = accum_t<T>;
    constexpr int A_ROWS = TA ? K : M;
    constexpr int A_COLS = TA ? M : K;
    constexpr int B_ROWS = TB ? N : K;
    constexpr int B_COLS = TB ? K : N;

    test_info this_result;
    this_result.label = label;
    if constexpr ((TS && TA) || (TS && sizeof(T) == 1 && !std::is_same_v<T, kittens::fp8e4m3> && !std::is_same_v<T, kittens::fp8e5m2>)) {
        this_result.result = test_result::INVALID;
        results.push_back(this_result);
        return;
    }

    std::vector<T> h_a(A_ROWS*A_COLS);
    std::vector<T> h_b(B_ROWS*B_COLS);
    std::vector<O> h_o(M*N, 0);
    std::vector<O> h_ref(M*N, 0);
    fill_input(h_a);
    fill_input(h_b);
    host_ref<T, TA, TB>(h_a, h_b, h_ref, ACC);

    T *d_a, *d_b;
    O *d_o;
    cudaMalloc(&d_a, h_a.size() * sizeof(T));
    cudaMalloc(&d_b, h_b.size() * sizeof(T));
    cudaMalloc(&d_o, h_o.size() * sizeof(O));
    CudaCheckError();
    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(d_o, 0, h_o.size() * sizeof(O));
    CudaCheckError();

    using GL_A = kittens::gl<T, 1, 1, A_ROWS, A_COLS>;
    using GL_B = kittens::gl<T, 1, 1, B_ROWS, B_COLS>;
    using GL_O = kittens::gl<O, 1, 1, M, N>;
    GL_A a_gl(d_a, nullptr, nullptr, nullptr, nullptr);
    GL_B b_gl(d_b, nullptr, nullptr, nullptr, nullptr);
    GL_O o_gl(d_o, nullptr, nullptr, nullptr, nullptr);

    cudaFuncSetAttribute(
        tcgen05_wrapper<T, TS, TA, TB, ACC, GL_A, GL_B, GL_O>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kittens::MAX_SHARED_MEMORY-1024
    );
    tcgen05_wrapper<T, TS, TA, TB, ACC, GL_A, GL_B, GL_O><<<1, kittens::group<4>::GROUP_THREADS, kittens::MAX_SHARED_MEMORY-1024>>>(a_gl, b_gl, o_gl);
    CudaCheckError();
    cudaMemcpy(h_o.data(), d_o, h_o.size() * sizeof(O), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    for(int i = 0; i < h_o.size(); i++) {
        if constexpr (std::is_same_v<O, int>) {
            if(h_o[i] != h_ref[i]) {
                good = false;
                break;
            }
        }
        else {
            if(std::abs(float(h_o[i] - h_ref[i])) > 1e-3f) {
                good = false;
                break;
            }
        }
    }
    std::cout << "test `" << label << "`";
    if(good) std::cout << " -- PASSED" << std::endl;
    else     std::cout << " ----- ALERT! FAILED test `" << label << "` -----" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);
    CudaCheckError();
    this_result.result = good ? test_result::PASSED : test_result::FAILED;
    results.push_back(this_result);
}

template<typename T>
static void run_type(test_data &results, const std::string &type_name) {
    run_one<T, false, false, false, false>(results, "tcgen05_st_st_mm_AB=" + type_name);
    run_one<T, false, false, true,  false>(results, "tcgen05_st_st_mm_ABt=" + type_name);
    run_one<T, false, true,  false, false>(results, "tcgen05_st_st_mm_AtB=" + type_name);
    run_one<T, false, true,  true,  false>(results, "tcgen05_st_st_mm_AtBt=" + type_name);
    run_one<T, false, false, false, true >(results, "tcgen05_st_st_mma_AB=" + type_name);
    run_one<T, false, false, true,  true >(results, "tcgen05_st_st_mma_ABt=" + type_name);
    run_one<T, false, true,  false, true >(results, "tcgen05_st_st_mma_AtB=" + type_name);
    run_one<T, false, true,  true,  true >(results, "tcgen05_st_st_mma_AtBt=" + type_name);

    run_one<T, true,  false, false, false>(results, "tcgen05_tt_st_mm_AB=" + type_name);
    run_one<T, true,  false, true,  false>(results, "tcgen05_tt_st_mm_ABt=" + type_name);
    run_one<T, true,  false, false, true >(results, "tcgen05_tt_st_mma_AB=" + type_name);
    run_one<T, true,  false, true,  true >(results, "tcgen05_tt_st_mma_ABt=" + type_name);
}

template<typename T>
static T *copy_to_device(const std::vector<T> &host) {
    T *device;
    cudaMalloc(&device, host.size() * sizeof(T));
    cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    CudaCheckError();
    return device;
}

template<typename T>
static T *zero_on_device(size_t count) {
    T *device;
    cudaMalloc(&device, count * sizeof(T));
    cudaMemset(device, 0, count * sizeof(T));
    CudaCheckError();
    return device;
}

template<typename T>
static void check_output(
    test_data &results,
    const std::string &label,
    T *d_o,
    std::vector<T> &h_o,
    const std::vector<T> &h_ref,
    float tolerance
) {
    cudaMemcpy(h_o.data(), d_o, h_o.size() * sizeof(T), cudaMemcpyDeviceToHost);
    CudaCheckError();

    int bad_idx = -1;
    for (int i = 0; i < h_o.size(); i++) {
        if (std::abs(h_o[i] - h_ref[i]) > tolerance) {
            bad_idx = i;
            break;
        }
    }
    std::cout << "test `" << label << "`";
    if (bad_idx < 0) std::cout << " -- PASSED" << std::endl;
    else             std::cout << " ----- ALERT! FAILED test `" << label
                               << "` first mismatch got " << h_o[bad_idx]
                               << " expected " << h_ref[bad_idx] << " -----" << std::endl;
    results.push_back({label, bad_idx < 0 ? test_result::PASSED : test_result::FAILED});
}

#ifdef KITTENS_SM10X
using fp4_packed = kittens::fp4e2m1_2;

constexpr int NVFP4_K64_MMAS = 2;
constexpr int NVFP4_K64_LOGICAL = 64 * NVFP4_K64_MMAS;
constexpr int NVFP4_K64_PACKED = NVFP4_K64_LOGICAL / 2;
constexpr float NVFP4_K64_TOL = 1e-3f;

template<typename Scale>
__device__ static inline Scale k64_scale_value(int value) {
    if constexpr (std::is_same_v<Scale, kittens::fp8e4m3>) {
        const uint8_t raw = value == 1 ? 0x38 : value == 2 ? 0x40 : 0x48;
        return std::bit_cast<Scale>(raw);
    }
    else {
        return std::bit_cast<Scale>(uint8_t(0x7e + value));
    }
}

template<
    typename Scale,
    kittens::ducks::gl::all GL_A,
    kittens::ducks::gl::all GL_B,
    kittens::ducks::gl::all GL_O
>
__launch_bounds__(kittens::group<4>::GROUP_THREADS)
__global__ void tcgen05_nvfp4_k64_wrapper(
    const __grid_constant__ GL_A a_gl,
    const __grid_constant__ GL_B b_gl,
    const __grid_constant__ GL_O o_gl
) {
    constexpr int M = 128;
    constexpr int N = 256;
    constexpr bool is_e4m3 = std::is_same_v<Scale, kittens::fp8e4m3>;
    constexpr int A_SCALE_COLS = is_e4m3 ? 48 : 16;
    constexpr int B_SCALE_COLS = is_e4m3 ? 96 : 32;
    using G = kittens::group<4>;
    using A_ST = kittens::st<fp4_packed, M, NVFP4_K64_PACKED>;
    using B_ST = kittens::st<fp4_packed, N, NVFP4_K64_PACKED>;
    using D_TT = kittens::tt<float, M, N>;
    using D_RT = kittens::rt<float, M / G::GROUP_WARPS, N>;
    using S_ATOM_ST = kittens::st<Scale, 32, 16, false>;
    using SA_TT = kittens::tt<Scale, kittens::MAX_TENSOR_ROWS, A_SCALE_COLS>;
    using SB_TT = kittens::tt<Scale, kittens::MAX_TENSOR_ROWS, B_SCALE_COLS>;
    using S_ATOM_TT = kittens::tt<Scale, kittens::MAX_TENSOR_ROWS, 16>;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::tma_swizzle_allocator al((int*)&__shm[0]);
    A_ST (&a_smem) = al.allocate<A_ST>();
    B_ST (&b_smem) = al.allocate<B_ST>();
    S_ATOM_ST (&sa_smem)[A_SCALE_COLS / 16] = al.allocate<S_ATOM_ST, A_SCALE_COLS / 16>();
    S_ATOM_ST (&sb_smem)[B_SCALE_COLS / 16] = al.allocate<S_ATOM_ST, B_SCALE_COLS / 16>();

    G::load(a_smem, a_gl, kittens::coord<A_ST>{0, 0});
    G::load(b_smem, b_gl, kittens::coord<B_ST>{0, 0});
    for (int atom = 0; atom < A_SCALE_COLS / 16; ++atom) {
        for (int idx = threadIdx.x; idx < S_ATOM_ST::num_elements; idx += blockDim.x) {
            const int col = idx % S_ATOM_ST::cols;
            const int value = is_e4m3 ? (1 << atom) : ((col % 4) < 2 ? 1 : 2);
            sa_smem[atom].data[idx] = k64_scale_value<Scale>(value);
        }
    }
    for (int atom = 0; atom < B_SCALE_COLS / 16; ++atom) {
        for (int idx = threadIdx.x; idx < S_ATOM_ST::num_elements; idx += blockDim.x) {
            sb_smem[atom].data[idx] = k64_scale_value<Scale>(1);
        }
    }
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    kittens::tensor_allocator<1, 1> tm_alloc{};
    D_TT d_tt = tm_alloc.template allocate<D_TT>(0);
    SA_TT sa_tt = tm_alloc.template allocate<SA_TT>(256);
    SB_TT sb_tt = tm_alloc.template allocate<SB_TT>(256 + A_SCALE_COLS / 4);
    if (kittens::warpid() == 0) {
        #pragma unroll
        for (int i = 0; i < A_SCALE_COLS / 16; ++i) {
            auto sa_tt_atom = sa_tt.template subtile<S_ATOM_TT>(i * 16);
            load_mxnv_scale_async(sa_tt_atom, sa_smem[i]);
        }
        #pragma unroll
        for (int i = 0; i < B_SCALE_COLS / 16; ++i) {
            auto sb_tt_atom = sb_tt.template subtile<S_ATOM_TT>(i * 16);
            load_mxnv_scale_async(sb_tt_atom, sb_smem[i]);
        }
        kittens::tensor_store_wait();
    }
    __syncthreads();

    __shared__ kittens::semaphore sem;
    kittens::warp::init_semaphore(sem, 0, 1);
    __syncthreads();
    if (kittens::warpid() == 0) {
        G::mm_ABt(d_tt, a_smem, b_smem, sa_tt, sb_tt, sem);
    }
    kittens::wait(sem, 0);

    D_RT d_reg;
    G::load_async(d_reg, d_tt);
    kittens::tensor_load_wait();
    G::store(o_gl, d_reg, kittens::coord<D_RT>{0, 0});
}

template<typename Scale>
static void run_nvfp4_k64(test_data &results) {
    constexpr int M = 128;
    constexpr int N = 256;
    const std::string scale_name = std::is_same_v<Scale, kittens::fp8e4m3> ? "e4m3" : "e8m0";
    const std::string label = "tcgen05_st_st_mm_ABt_k64=nvfp4_" + scale_name;

    const fp4_packed one = std::bit_cast<fp4_packed>(uint8_t(0x22));
    std::vector<fp4_packed> h_a(M * NVFP4_K64_PACKED, one);
    std::vector<fp4_packed> h_b(N * NVFP4_K64_PACKED, one);
    std::vector<float> h_o(M * N, 0.0f);
    std::vector<float> h_ref(M * N, 64.0f * 1.0f + 64.0f * 2.0f);

    fp4_packed *d_a = copy_to_device(h_a);
    fp4_packed *d_b = copy_to_device(h_b);
    float *d_o = zero_on_device<float>(h_o.size());

    using A_ST = kittens::st<fp4_packed, M, NVFP4_K64_PACKED>;
    using B_ST = kittens::st<fp4_packed, N, NVFP4_K64_PACKED>;
    using GL_A = kittens::gl<fp4_packed, 1, 1, M, NVFP4_K64_PACKED, A_ST>;
    using GL_B = kittens::gl<fp4_packed, 1, 1, N, NVFP4_K64_PACKED, B_ST>;
    using GL_O = kittens::gl<float, 1, 1, M, N>;
    GL_A a_gl(d_a, nullptr, nullptr, nullptr, nullptr);
    GL_B b_gl(d_b, nullptr, nullptr, nullptr, nullptr);
    GL_O o_gl(d_o, nullptr, nullptr, nullptr, nullptr);

    cudaFuncSetAttribute(
        tcgen05_nvfp4_k64_wrapper<Scale, GL_A, GL_B, GL_O>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kittens::MAX_SHARED_MEMORY - 1024
    );
    tcgen05_nvfp4_k64_wrapper<Scale, GL_A, GL_B, GL_O><<<
        dim3(1), dim3(kittens::group<4>::GROUP_THREADS), kittens::MAX_SHARED_MEMORY - 1024
    >>>(a_gl, b_gl, o_gl);
    CudaCheckError();
    check_output(results, label, d_o, h_o, h_ref, NVFP4_K64_TOL);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);
    CudaCheckError();
}
#endif

#if defined(KITTENS_SM103) || defined(KITTENS_SM107)
using nvfp4_scale = kittens::fp8e8m0;
constexpr float NVFP4_E8_TOL = 1e-3f;
constexpr int NVFP4_K96_MMAS = 8;
constexpr int NVFP4_K96_PACKED = 48 * NVFP4_K96_MMAS;
constexpr int NVFP4_K96_SCALE_COLS = 16 * NVFP4_K96_MMAS;

using k96_compat_a = kittens::st<fp4_packed, 128, NVFP4_K96_PACKED>;
using k96_compat_b1 = kittens::st<fp4_packed, 256, NVFP4_K96_PACKED>;
using k96_compat_b2 = kittens::st<fp4_packed, 128, NVFP4_K96_PACKED>;
using k96_compat_d = kittens::tt<float, 128, 256>;
using k96_compat_sa = kittens::full_tt_fp8e8m0<NVFP4_K96_SCALE_COLS>;
using k96_compat_sb = kittens::full_tt_fp8e8m0<2 * NVFP4_K96_SCALE_COLS>;

[[maybe_unused]] __device__ static inline void compile_block_scaled_api_compat(
    k96_compat_d &d, const k96_compat_a &a, const k96_compat_b1 &b1, const k96_compat_b2 &b2,
    const k96_compat_sa &sa, const k96_compat_sb &sb, kittens::semaphore &sem
) {
    kittens::mm_ABt<48>(d, a, b1, sa, sb, sem);
    kittens::mma_ABt<48>(d, a, b1, sa, sb, sem);
    kittens::mm2_ABt<48>(d, a, b2, sa, sb, sem);
    kittens::mma2_ABt<48>(d, a, b2, sa, sb, sem);
    kittens::mm_ABt<48>(d, a, b1, sa, sb);
    kittens::mma_ABt<48>(d, a, b1, sa, sb);
    kittens::mm2_ABt<48>(d, a, b2, sa, sb);
    kittens::mma2_ABt<48>(d, a, b2, sa, sb);
    kittens::group<4>::mm_ABt<48>(d, a, b1, sa, sb, sem);
    kittens::group<4>::mma_ABt<48>(d, a, b1, sa, sb, sem);
    kittens::group<4>::mm2_ABt<48>(d, a, b2, sa, sb, sem);
    kittens::group<4>::mma2_ABt<48>(d, a, b2, sa, sb, sem);
    kittens::group<4>::mm_ABt<48>(d, a, b1, sa, sb);
    kittens::group<4>::mma_ABt<48>(d, a, b1, sa, sb);
    kittens::group<4>::mm2_ABt<48>(d, a, b2, sa, sb);
    kittens::group<4>::mma2_ABt<48>(d, a, b2, sa, sb);
#if defined(KITTENS_SM103) || defined(KITTENS_SM107)
    kittens::st_descriptor<k96_compat_a, 0> a_desc(a);
    (void)a_desc.chunk_descriptor_k96(0);
#endif
}

template<int MMA_K_BYTES, int MMAS, int CTA_COUNT, bool ACC, kittens::ducks::gl::all GL_A, kittens::ducks::gl::all GL_B, kittens::ducks::gl::all GL_O>
__device__ static inline void tcgen05_nvfp4_e8(
    const GL_A a_gl,
    const GL_B b_gl,
    const GL_O o_gl
) {
    static_assert(CTA_COUNT == 1 || CTA_COUNT == 2);
    constexpr int M = 128;
    constexpr int N = 256;
    constexpr int K_PACKED = MMA_K_BYTES * MMAS;
    constexpr int SCALE_COLS = 16 * MMAS;
    using G = kittens::group<4>;
    using A_ST = kittens::st<fp4_packed, M, K_PACKED>;
    using B_ST = kittens::st<fp4_packed, N / CTA_COUNT, K_PACKED>;
    using D_TT = kittens::tt<float, M, N>;
    using D_RT = kittens::rt<float, M / G::GROUP_WARPS, N>;
    using D_GROUP_RT = kittens::rt<float, G::GROUP_WARPS * D_RT::rows, D_RT::cols>;
    using S_ST = kittens::st<nvfp4_scale, 32, SCALE_COLS, false>;
    using S_ATOM_ST = kittens::st<nvfp4_scale, 32, 16, false>;
    using SA_TT = kittens::full_tt_fp8e8m0<SCALE_COLS>;
    using SB_TT = kittens::full_tt_fp8e8m0<2 * SCALE_COLS>;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::tma_swizzle_allocator al((int*)&__shm[0]);
    A_ST (&a_smem) = al.allocate<A_ST>();
    B_ST (&b_smem) = al.allocate<B_ST>();
    S_ST (&sa_smem) = al.allocate<S_ST>();
    S_ST (&sb_smem) = al.allocate<S_ST>();

    const int cta_id = CTA_COUNT == 2 ? kittens::cluster_ctarank() : 0;
    G::load(a_smem, a_gl, kittens::coord<A_ST>{cta_id, 0});
    G::load(b_smem, b_gl, kittens::coord<B_ST>{cta_id, 0});
    for (int idx = threadIdx.x; idx < S_ST::num_elements; idx += blockDim.x) {
        sa_smem.data[idx] = std::bit_cast<nvfp4_scale>(uint8_t(0x80)); // 2.0
        sb_smem.data[idx] = std::bit_cast<nvfp4_scale>(uint8_t(CTA_COUNT == 2 ? 0x7f : 0x80));
    }
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    kittens::tensor_allocator<1, CTA_COUNT> tm_alloc{};

    D_TT d_tt = tm_alloc.template allocate<D_TT>(0);
    auto sa_tt = tm_alloc.template allocate<SA_TT>(256);
    auto sb_tt = tm_alloc.template allocate<SB_TT>(256 + 4 * MMAS);
    if (cta_id == 0 && kittens::warpid() == 0) {
        #pragma unroll
        for (int i = 0; i < MMAS; ++i) {
            auto sa_tt_atom = sa_tt.template subtile<kittens::full_tt_fp8e8m0<16>>(i * 16);
            auto sb_tt_atom_0 = sb_tt.template subtile<kittens::full_tt_fp8e8m0<16>>(i * 32);
            auto sb_tt_atom_1 = sb_tt.template subtile<kittens::full_tt_fp8e8m0<16>>(i * 32 + 16);
            auto &sa_smem_atom = *reinterpret_cast<S_ATOM_ST *>(
                reinterpret_cast<uint64_t>(&sa_smem.data[0]) + i * 16 * 32);
            auto &sb_smem_atom = *reinterpret_cast<S_ATOM_ST *>(
                reinterpret_cast<uint64_t>(&sb_smem.data[0]) + i * 16 * 32);
            if constexpr (CTA_COUNT == 2) {
                load_mxnv_scale_async2(sa_tt_atom, sa_smem_atom);
                load_mxnv_scale_async2(sb_tt_atom_0, sb_smem_atom);
                load_mxnv_scale_async2(sb_tt_atom_1, sb_smem_atom);
            }
            else {
                load_mxnv_scale_async(sa_tt_atom, sa_smem_atom);
                load_mxnv_scale_async(sb_tt_atom_0, sb_smem_atom);
                load_mxnv_scale_async(sb_tt_atom_1, sb_smem_atom);
            }
        }
        kittens::tensor_store_wait();
    }
    __syncthreads();

    __shared__ kittens::semaphore sem;
    kittens::warp::init_semaphore(sem, 0, 1);
    __syncthreads();
    if constexpr (CTA_COUNT == 2) kittens::everyone::tma::cluster::sync();

    if (cta_id == 0 && kittens::warpid() == 0) {
        if constexpr (CTA_COUNT == 2) G::mm2_ABt<MMA_K_BYTES>(d_tt, a_smem, b_smem, sa_tt, sb_tt, sem);
        else                          G::mm_ABt<MMA_K_BYTES> (d_tt, a_smem, b_smem, sa_tt, sb_tt, sem);
    }
    kittens::wait(sem, 0);
    if constexpr (ACC) {
        if (cta_id == 0 && kittens::warpid() == 0) {
            if constexpr (CTA_COUNT == 2) G::mma2_ABt<MMA_K_BYTES>(d_tt, a_smem, b_smem, sa_tt, sb_tt, sem);
            else                          G::mma_ABt<MMA_K_BYTES> (d_tt, a_smem, b_smem, sa_tt, sb_tt, sem);
        }
        kittens::wait(sem, 1);
    }

    D_RT d_reg;
    G::load_async(d_reg, d_tt);
    kittens::tensor_load_wait();
    if constexpr (CTA_COUNT == 2) G::store(o_gl, d_reg, kittens::coord<D_GROUP_RT>{cta_id, 0});
    else                          G::store(o_gl, d_reg, kittens::coord<D_RT>{0, 0});
}

template<int MMA_K_BYTES, int MMAS, bool ACC, kittens::ducks::gl::all GL_A, kittens::ducks::gl::all GL_B, kittens::ducks::gl::all GL_O>
__cluster_dims__(2, 1, 1) __launch_bounds__(kittens::group<4>::GROUP_THREADS)
__global__ void tcgen05_nvfp4_e8_2cta_wrapper(
    const __grid_constant__ GL_A a_gl,
    const __grid_constant__ GL_B b_gl,
    const __grid_constant__ GL_O o_gl
) {
    tcgen05_nvfp4_e8<MMA_K_BYTES, MMAS, 2, ACC>(a_gl, b_gl, o_gl);
}

template<int MMA_K_BYTES, int MMAS, bool ACC, kittens::ducks::gl::all GL_A, kittens::ducks::gl::all GL_B, kittens::ducks::gl::all GL_O>
__launch_bounds__(kittens::group<4>::GROUP_THREADS)
__global__ void tcgen05_nvfp4_e8_1cta_wrapper(
    const __grid_constant__ GL_A a_gl,
    const __grid_constant__ GL_B b_gl,
    const __grid_constant__ GL_O o_gl
) {
    tcgen05_nvfp4_e8<MMA_K_BYTES, MMAS, 1, ACC>(a_gl, b_gl, o_gl);
}

template<int MMA_K_BYTES, int MMAS, int CTA_COUNT, bool ACC>
static void run_nvfp4_e8_case(test_data &results) {
    static_assert(CTA_COUNT == 1 || CTA_COUNT == 2);
    constexpr int N = 256;
    constexpr int M = 128 * CTA_COUNT;
    constexpr int K_LOGICAL = 2 * MMA_K_BYTES * MMAS;
    constexpr int K_PACKED = MMA_K_BYTES * MMAS;
    const std::string op = ACC ? "mma" : "mm";
    const std::string cta = CTA_COUNT == 2 ? "2" : "";
    const std::string label = "tcgen05_st_st_" + op + cta + "_ABt_k" + std::to_string(2 * MMA_K_BYTES) + "=nvfp4_e8m0";

    const fp4_packed one = std::bit_cast<fp4_packed>(uint8_t(0x22));
    std::vector<fp4_packed> h_a(M * K_PACKED, one);
    std::vector<fp4_packed> h_b(N * K_PACKED, one);
    std::vector<float> h_o(M * N, 0.0f);
    constexpr int scale_product = CTA_COUNT == 2 ? 2 : 4;
    std::vector<float> h_ref(M * N, float(K_LOGICAL * scale_product * (ACC ? 2 : 1)));

    fp4_packed *d_a = copy_to_device(h_a);
    fp4_packed *d_b = copy_to_device(h_b);
    float *d_o = zero_on_device<float>(h_o.size());

    using A_ST = kittens::st<fp4_packed, 128, K_PACKED>;
    using B_ST = kittens::st<fp4_packed, N / CTA_COUNT, K_PACKED>;
    using GL_A = kittens::gl<fp4_packed, 1, 1, M, K_PACKED, A_ST>;
    using GL_B = kittens::gl<fp4_packed, 1, 1, N, K_PACKED, B_ST>;
    using GL_O = kittens::gl<float, 1, 1, M, N>;
    GL_A a_gl(d_a, nullptr, nullptr, nullptr, nullptr);
    GL_B b_gl(d_b, nullptr, nullptr, nullptr, nullptr);
    GL_O o_gl(d_o, nullptr, nullptr, nullptr, nullptr);

    if constexpr (CTA_COUNT == 2) {
        cudaFuncSetAttribute(
            tcgen05_nvfp4_e8_2cta_wrapper<MMA_K_BYTES, MMAS, ACC, GL_A, GL_B, GL_O>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kittens::MAX_SHARED_MEMORY - 1024
        );
        kittens::LaunchConfig<true> launch_config(
            dim3(2), dim3(kittens::group<4>::GROUP_THREADS), kittens::MAX_SHARED_MEMORY - 1024, nullptr, dim3(2)
        );
        cudaLaunchKernelEx(
            launch_config, tcgen05_nvfp4_e8_2cta_wrapper<MMA_K_BYTES, MMAS, ACC, GL_A, GL_B, GL_O>, a_gl, b_gl, o_gl
        );
    }
    else {
        cudaFuncSetAttribute(
            tcgen05_nvfp4_e8_1cta_wrapper<MMA_K_BYTES, MMAS, ACC, GL_A, GL_B, GL_O>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kittens::MAX_SHARED_MEMORY - 1024
        );
        tcgen05_nvfp4_e8_1cta_wrapper<MMA_K_BYTES, MMAS, ACC, GL_A, GL_B, GL_O><<<
            dim3(1), dim3(kittens::group<4>::GROUP_THREADS), kittens::MAX_SHARED_MEMORY - 1024
        >>>(a_gl, b_gl, o_gl);
    }
    CudaCheckError();
    check_output(results, label, d_o, h_o, h_ref, NVFP4_E8_TOL);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);
    CudaCheckError();
}

static void run_nvfp4_k96(test_data &results) {
    run_nvfp4_e8_case<48, NVFP4_K96_MMAS, 2, false>(results);
    run_nvfp4_e8_case<48, NVFP4_K96_MMAS, 2, true>(results);
    run_nvfp4_e8_case<48, NVFP4_K96_MMAS, 1, false>(results);
    run_nvfp4_e8_case<48, NVFP4_K96_MMAS, 1, true>(results);
}
#endif

#ifdef KITTENS_SM107
constexpr float NVFP4_K128_TOL = 1e-3f;
constexpr int NVFP4_K128_SF128_MMAS = 2;
constexpr int NVFP4_K128_SF128_LOGICAL = 128 * NVFP4_K128_SF128_MMAS;
constexpr int NVFP4_K128_SF128_PACKED = NVFP4_K128_SF128_LOGICAL / 2;

// Layout-discriminating scale values (exact powers of two).
__host__ __device__ static inline constexpr int k128_sa_log2(int m, int kb) { return (m/32 + kb) % 3; }
__host__ __device__ static inline constexpr int k128_sb_log2(int n, int kb) { return (n/32 + kb) % 2; }
__device__ static inline kittens::fp8e4m3 k128_scale_value(int log2v) {
    return std::bit_cast<kittens::fp8e4m3>(uint8_t(0x38 + (log2v << 3)));
}

// Fill either the SM107 128-lane A-scale layout or consecutive 32x16 B-scale atoms.
template<int DIM, bool IS_A, kittens::ducks::st::all S_ST>
__device__ static inline void k128_fill_scales(S_ST &s_smem) {
    for (int idx = threadIdx.x; idx < S_ST::num_elements; idx += blockDim.x) {
        if constexpr (IS_A) {
            s_smem.data[idx] = k128_scale_value(k128_sa_log2(idx / 16, idx % 16));
        }
        else {
            const int i = (idx >> 4) % 32, c = 4*(idx >> 9) + ((idx >> 2) & 3), b = idx & 3;
            const int region = 2 * (DIM/32);                                       // word-columns per MMA
            const int j = c / region, lc = c % region;
            const int mn = 32*(lc % (DIM/32)) + i, kb = 4*(2*j + lc/(DIM/32)) + b; // matrix row/col, 16-elem block index
            s_smem.data[idx] = k128_scale_value(k128_sb_log2(mn, kb));
        }
    }
}

template<
    kittens::ducks::gl::all GL_A,
    kittens::ducks::gl::all GL_B,
    kittens::ducks::gl::all GL_O
>
__launch_bounds__(kittens::group<4>::GROUP_THREADS)
__global__ void tcgen05_nvfp4_k128_1cta_wrapper(
    const __grid_constant__ GL_A a_gl,
    const __grid_constant__ GL_B b_gl,
    const __grid_constant__ GL_O o_gl
) {
    constexpr int M = 128;
    constexpr int N = 256;
    constexpr int A_SCALE_COLS = 16;
    constexpr int B_SCALE_COLS = 32 * 2 * NVFP4_K128_SF128_MMAS;
    using Scale = kittens::fp8e4m3;
    using G = kittens::group<4>;
    using A_ST = kittens::st<fp4_packed, M, NVFP4_K128_SF128_PACKED>;
    using B_ST = kittens::st<fp4_packed, N, NVFP4_K128_SF128_PACKED>;
    using D_TT = kittens::tt<float, M, N>;
    using D_RT = kittens::rt<float, M / G::GROUP_WARPS, N>;
    using D_GROUP_RT = kittens::rt<float, G::GROUP_WARPS * D_RT::rows, D_RT::cols>;
    using S_ATOM_ST = kittens::st<Scale, 32, 16, false>;
    using SA_ST = kittens::st<Scale, kittens::MAX_TENSOR_ROWS, 16, false>;
    using SA_TT = kittens::full_tt_fp8e4m3<16>;
    using SB_TT = kittens::tt<Scale, kittens::MAX_TENSOR_ROWS, B_SCALE_COLS>;
    using S_ATOM_TT = kittens::tt<Scale, kittens::MAX_TENSOR_ROWS, 16>;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::tma_swizzle_allocator al((int*)&__shm[0]);
    A_ST (&a_smem) = al.allocate<A_ST>();
    B_ST (&b_smem) = al.allocate<B_ST>();
    SA_ST (&sa_smem) = al.allocate<SA_ST>();
    S_ATOM_ST (&sb_smem)[B_SCALE_COLS / 16] = al.allocate<S_ATOM_ST, B_SCALE_COLS / 16>();

    G::load(a_smem, a_gl, kittens::coord<A_ST>{0, 0});
    G::load(b_smem, b_gl, kittens::coord<B_ST>{0, 0});
    k128_fill_scales<M, true>(sa_smem);
    k128_fill_scales<N, false>(*reinterpret_cast<kittens::st<Scale, 32, B_SCALE_COLS, false> *>(&sb_smem[0]));
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    kittens::tensor_allocator<1, 1> tm_alloc{};
    D_TT d_tt = tm_alloc.template allocate<D_TT>(0);
    SA_TT sa_tt = tm_alloc.template allocate<SA_TT>(256);
    SB_TT sb_tt = tm_alloc.template allocate<SB_TT>(256 + A_SCALE_COLS / 4);
    if (kittens::warpid() == 0) {
        load_mxnv_scale_async(sa_tt, sa_smem);
        #pragma unroll
        for (int i = 0; i < B_SCALE_COLS / 16; ++i) {
            auto sb_tt_atom = sb_tt.template subtile<S_ATOM_TT>(i * 16);
            load_mxnv_scale_async(sb_tt_atom, sb_smem[i]);
        }
        kittens::tensor_store_wait();
    }
    __syncthreads();

    __shared__ kittens::semaphore sem;
    kittens::warp::init_semaphore(sem, 0, 1);
    __syncthreads();

    if (kittens::warpid() == 0) {
        G::mm_ABt<64, true>(d_tt, a_smem, b_smem, sa_tt, sb_tt, sem);
    }
    kittens::wait(sem, 0);
    D_RT d_reg;
    G::load_async(d_reg, d_tt);
    kittens::tensor_load_wait();
    G::store(o_gl, d_reg, kittens::coord<D_GROUP_RT>{0, 0});
}

static void run_nvfp4_k128_sf128(test_data &results) {
    constexpr int M = 128;
    constexpr int N = 256;
    const std::string label = "tcgen05_st_st_mm_ABt_k128=nvfp4_e4m3";

    const fp4_packed one = std::bit_cast<fp4_packed>(uint8_t(0x22));
    std::vector<fp4_packed> h_a(M * NVFP4_K128_SF128_PACKED, one);
    std::vector<fp4_packed> h_b(N * NVFP4_K128_SF128_PACKED, one);
    std::vector<float> h_o(M * N, 0.0f);
    std::vector<float> h_ref(M * N);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int kb = 0; kb < NVFP4_K128_SF128_LOGICAL / 16; kb++)
                sum += 16 * std::ldexp(1.0f, k128_sa_log2(m, kb)) * std::ldexp(1.0f, k128_sb_log2(n, kb));
            h_ref[m*N + n] = sum;
        }
    }

    fp4_packed *d_a = copy_to_device(h_a);
    fp4_packed *d_b = copy_to_device(h_b);
    float *d_o = zero_on_device<float>(h_o.size());

    using A_ST = kittens::st<fp4_packed, M, NVFP4_K128_SF128_PACKED>;
    using B_ST = kittens::st<fp4_packed, N, NVFP4_K128_SF128_PACKED>;
    using GL_A = kittens::gl<fp4_packed, 1, 1, M, NVFP4_K128_SF128_PACKED, A_ST>;
    using GL_B = kittens::gl<fp4_packed, 1, 1, N, NVFP4_K128_SF128_PACKED, B_ST>;
    using GL_O = kittens::gl<float, 1, 1, M, N>;
    GL_A a_gl(d_a, nullptr, nullptr, nullptr, nullptr);
    GL_B b_gl(d_b, nullptr, nullptr, nullptr, nullptr);
    GL_O o_gl(d_o, nullptr, nullptr, nullptr, nullptr);

    cudaFuncSetAttribute(
        tcgen05_nvfp4_k128_1cta_wrapper<GL_A, GL_B, GL_O>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kittens::MAX_SHARED_MEMORY - 1024
    );
    tcgen05_nvfp4_k128_1cta_wrapper<GL_A, GL_B, GL_O><<<
        dim3(1), dim3(kittens::group<4>::GROUP_THREADS), kittens::MAX_SHARED_MEMORY - 1024
    >>>(a_gl, b_gl, o_gl);
    CudaCheckError();
    check_output(results, label, d_o, h_o, h_ref, NVFP4_K128_TOL);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);
    CudaCheckError();
}

static void run_nvfp4_k128(test_data &results) {
    // Each call initializes D on its first K chunk and accumulates the remaining chunks.
    run_nvfp4_k128_sf128(results);
    run_nvfp4_e8_case<64, 4, 2, false>(results);
}

constexpr float F8_K64_TOL = 1e-3f;
constexpr int F8_K64_MMAS = 4;
constexpr int F8_K64_K = 64 * F8_K64_MMAS;

// Chunk-index-discriminating exact values (powers of two) so descriptor-stepping bugs are visible.
__host__ __device__ static inline constexpr int f8_k64_a_log2(int m, int kc) { return (m/32 + kc) % 3 - 1; }
__host__ __device__ static inline constexpr int f8_k64_b_log2(int n, int kc) { return (n/32 + kc) % 2; }

template<kittens::ducks::gl::all GL_A, kittens::ducks::gl::all GL_B, kittens::ducks::gl::all GL_O>
__cluster_dims__(2, 1, 1) __launch_bounds__(kittens::group<4>::GROUP_THREADS)
__global__ void tcgen05_f8_k64_wrapper(
    const __grid_constant__ GL_A a_gl,
    const __grid_constant__ GL_B b_gl,
    const __grid_constant__ GL_O o_gl
) {
    constexpr int M = 128;
    constexpr int N = 256;
    using G = kittens::group<4>;
    using A_ST = kittens::st<kittens::fp8e4m3, M, F8_K64_K>;
    using B_ST = kittens::st<kittens::fp8e4m3, N / 2, F8_K64_K>;
    using D_TT = kittens::tt<float, M, N>;
    using D_RT = kittens::rt<float, M / G::GROUP_WARPS, N>;
    using D_GROUP_RT = kittens::rt<float, G::GROUP_WARPS * D_RT::rows, D_RT::cols>;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::tma_swizzle_allocator al((int*)&__shm[0]);
    A_ST (&a_smem) = al.allocate<A_ST>();
    B_ST (&b_smem) = al.allocate<B_ST>();

    const int cta_id = kittens::cluster_ctarank();
    G::load(a_smem, a_gl, kittens::coord<A_ST>{cta_id, 0});
    G::load(b_smem, b_gl, kittens::coord<B_ST>{cta_id, 0});
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    kittens::tensor_allocator<1, 2> tm_alloc{};
    D_TT d_tt = tm_alloc.template allocate<D_TT>(0);
    __shared__ kittens::semaphore sem;
    kittens::warp::init_semaphore(sem, 0, 1);
    __syncthreads();

    kittens::everyone::tma::cluster::sync();

    if (cta_id == 0 && kittens::warpid() == 0 && kittens::laneid() == 0) {
        kittens::st_descriptor<A_ST, 0> a_desc(a_smem);
        kittens::st_descriptor<B_ST, 0> b_desc(b_smem);
        #pragma unroll
        for (int i = 0; i < F8_K64_MMAS; ++i) {
            kittens::mma2_ABt_chunk(d_tt, a_desc, b_desc, i, i == 0);
        }
        kittens::tensor_commit<2>(sem);
    }
    kittens::wait(sem, 0);

    D_RT d_reg;
    G::load_async(d_reg, d_tt);
    kittens::tensor_load_wait();
    G::store(o_gl, d_reg, kittens::coord<D_GROUP_RT>{cta_id, 0});
}

static void run_f8_k64(test_data &results) {
    constexpr int M = 256;
    constexpr int N = 256;
    const std::string label = "tcgen05_st_st_mm2_ABt_k64=f8";

    std::vector<kittens::fp8e4m3> h_a(M * F8_K64_K), h_b(N * F8_K64_K);
    for (int m = 0; m < M; m++) for (int k = 0; k < F8_K64_K; k++)
        h_a[m*F8_K64_K + k] = kittens::fp8e4m3(std::ldexp(1.0f, f8_k64_a_log2(m, k/64)));
    for (int n = 0; n < N; n++) for (int k = 0; k < F8_K64_K; k++)
        h_b[n*F8_K64_K + k] = kittens::fp8e4m3(std::ldexp(1.0f, f8_k64_b_log2(n, k/64)));
    std::vector<float> h_o(M * N, 0.0f), h_ref(M * N);
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        float sum = 0.0f;
        for (int kc = 0; kc < F8_K64_MMAS; kc++)
            sum += 64.0f * std::ldexp(1.0f, f8_k64_a_log2(m, kc) + f8_k64_b_log2(n, kc));
        h_ref[m*N + n] = sum;
    }

    kittens::fp8e4m3 *d_a = copy_to_device(h_a);
    kittens::fp8e4m3 *d_b = copy_to_device(h_b);
    float *d_o = zero_on_device<float>(h_o.size());

    using A_ST = kittens::st<kittens::fp8e4m3, M / 2, F8_K64_K>;
    using B_ST = kittens::st<kittens::fp8e4m3, N / 2, F8_K64_K>;
    using GL_A = kittens::gl<kittens::fp8e4m3, 1, 1, M, F8_K64_K, A_ST>;
    using GL_B = kittens::gl<kittens::fp8e4m3, 1, 1, N, F8_K64_K, B_ST>;
    using GL_O = kittens::gl<float, 1, 1, M, N>;
    GL_A a_gl(d_a, nullptr, nullptr, nullptr, nullptr);
    GL_B b_gl(d_b, nullptr, nullptr, nullptr, nullptr);
    GL_O o_gl(d_o, nullptr, nullptr, nullptr, nullptr);

    kittens::LaunchConfig<true> launch_config(
        dim3(2), dim3(kittens::group<4>::GROUP_THREADS), kittens::MAX_SHARED_MEMORY - 1024, nullptr, dim3(2)
    );
    cudaFuncSetAttribute(
        tcgen05_f8_k64_wrapper<GL_A, GL_B, GL_O>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kittens::MAX_SHARED_MEMORY - 1024
    );
    cudaLaunchKernelEx(
        launch_config, tcgen05_f8_k64_wrapper<GL_A, GL_B, GL_O>, a_gl, b_gl, o_gl
    );
    CudaCheckError();
    check_output(results, label, d_o, h_o, h_ref, F8_K64_TOL);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);
    CudaCheckError();
}

// tensor_aread_commit fires after prior MMAs finish reading A, so refilling A then must leave D intact.
// An early arrive corrupts D, a missing one hangs, and the refill must be a real global load the compiler keeps.
template<kittens::ducks::gl::all GL_A, kittens::ducks::gl::all GL_B, kittens::ducks::gl::all GL_O>
__cluster_dims__(2, 1, 1) __launch_bounds__(kittens::group<4>::GROUP_THREADS)
__global__ void tcgen05_aread_commit_2cta_wrapper(
    const __grid_constant__ GL_A a_gl,
    const __grid_constant__ GL_A az_gl,
    const __grid_constant__ GL_B b_gl,
    const __grid_constant__ GL_O o_gl
) {
    constexpr int M = 128;
    constexpr int N = 256;
    using G = kittens::group<4>;
    using A_ST = kittens::st<kittens::fp8e4m3, M, F8_K64_K>;
    using B_ST = kittens::st<kittens::fp8e4m3, N / 2, F8_K64_K>;
    using D_TT = kittens::tt<float, M, N>;
    using D_RT = kittens::rt<float, M / G::GROUP_WARPS, N>;
    using D_GROUP_RT = kittens::rt<float, G::GROUP_WARPS * D_RT::rows, D_RT::cols>;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::tma_swizzle_allocator al((int*)&__shm[0]);
    A_ST (&a_smem) = al.allocate<A_ST>();
    B_ST (&b_smem) = al.allocate<B_ST>();

    const int cta_id = kittens::cluster_ctarank();
    G::load(a_smem, a_gl, kittens::coord<A_ST>{cta_id, 0});
    G::load(b_smem, b_gl, kittens::coord<B_ST>{cta_id, 0});
    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();

    kittens::tensor_allocator<1, 2> tm_alloc{};
    D_TT d_tt = tm_alloc.template allocate<D_TT>(0);
    __shared__ kittens::semaphore a_sem, d_sem;
    kittens::warp::init_semaphore(a_sem, 0, 1);
    kittens::warp::init_semaphore(d_sem, 0, 1);
    __syncthreads();

    kittens::everyone::tma::cluster::sync();

    if (cta_id == 0 && kittens::warpid() == 0 && kittens::laneid() == 0) {
        kittens::st_descriptor<A_ST, 0> a_desc(a_smem);
        kittens::st_descriptor<B_ST, 0> b_desc(b_smem);
        #pragma unroll
        for (int i = 0; i < F8_K64_MMAS; ++i) {
            kittens::mma2_ABt_chunk(d_tt, a_desc, b_desc, i, i == 0);
        }
        kittens::tensor_aread_commit<2>(a_sem);
        kittens::tensor_commit<2>(d_sem);
    }
    kittens::wait(a_sem, 0);
    G::load(a_smem, az_gl, kittens::coord<A_ST>{cta_id, 0});
    __syncthreads();
    kittens::wait(d_sem, 0);

    D_RT d_reg;
    G::load_async(d_reg, d_tt);
    kittens::tensor_load_wait();
    G::store(o_gl, d_reg, kittens::coord<D_GROUP_RT>{cta_id, 0});
}

static void run_aread_commit(test_data &results) {
    constexpr int M = 256;
    constexpr int N = 256;
    const std::string label = "tcgen05_aread_commit2_k64=f8";

    std::vector<kittens::fp8e4m3> h_a(M * F8_K64_K, kittens::fp8e4m3(1.0f));
    std::vector<kittens::fp8e4m3> h_b(N * F8_K64_K, kittens::fp8e4m3(1.0f));
    std::vector<float> h_o(M * N, 0.0f), h_ref(M * N, float(F8_K64_K));

    kittens::fp8e4m3 *d_a = copy_to_device(h_a);
    kittens::fp8e4m3 *d_az = zero_on_device<kittens::fp8e4m3>(h_a.size());
    kittens::fp8e4m3 *d_b = copy_to_device(h_b);
    float *d_o = zero_on_device<float>(h_o.size());

    using A_ST = kittens::st<kittens::fp8e4m3, M / 2, F8_K64_K>;
    using B_ST = kittens::st<kittens::fp8e4m3, N / 2, F8_K64_K>;
    using GL_A = kittens::gl<kittens::fp8e4m3, 1, 1, M, F8_K64_K, A_ST>;
    using GL_B = kittens::gl<kittens::fp8e4m3, 1, 1, N, F8_K64_K, B_ST>;
    using GL_O = kittens::gl<float, 1, 1, M, N>;
    GL_A a_gl(d_a, nullptr, nullptr, nullptr, nullptr);
    GL_A az_gl(d_az, nullptr, nullptr, nullptr, nullptr);
    GL_B b_gl(d_b, nullptr, nullptr, nullptr, nullptr);
    GL_O o_gl(d_o, nullptr, nullptr, nullptr, nullptr);

    kittens::LaunchConfig<true> launch_config(
        dim3(2), dim3(kittens::group<4>::GROUP_THREADS), kittens::MAX_SHARED_MEMORY - 1024, nullptr, dim3(2)
    );
    cudaFuncSetAttribute(
        tcgen05_aread_commit_2cta_wrapper<GL_A, GL_B, GL_O>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kittens::MAX_SHARED_MEMORY - 1024
    );
    cudaLaunchKernelEx(
        launch_config, tcgen05_aread_commit_2cta_wrapper<GL_A, GL_B, GL_O>, a_gl, az_gl, b_gl, o_gl
    );
    CudaCheckError();
    check_output(results, label, d_o, h_o, h_ref, F8_K64_TOL);

    cudaFree(d_a);
    cudaFree(d_az);
    cudaFree(d_b);
    cudaFree(d_o);
    CudaCheckError();
}

#endif

}

void group::mma::tensor::mma::tests(test_data &results) {
    std::cout << " ----- Starting ops/group/mma/tensor/mma tests! -----\n" << std::endl;
    run_type<kittens::bf16>(results, "bf16");
    run_type<kittens::half>(results, "half");
    run_type<kittens::fp8e4m3>(results, "fp8e4m3");
    run_type<kittens::fp8e5m2>(results, "fp8e5m2");
#if !defined(KITTENS_SM103) && !defined(KITTENS_SM107)
    run_type<kittens::int8>(results, "int8");
    run_type<kittens::uint8>(results, "uint8");
#endif
#ifdef KITTENS_SM10X
    run_nvfp4_k64<kittens::fp8e4m3>(results);
    run_nvfp4_k64<kittens::fp8e8m0>(results);
#endif
#if defined(KITTENS_SM103) || defined(KITTENS_SM107)
    run_nvfp4_k96(results);
#endif
#ifdef KITTENS_SM107
    run_nvfp4_k128(results);
    run_f8_k64(results);
    run_aread_commit(results);
#endif
    std::cout << std::endl;
}

#endif
