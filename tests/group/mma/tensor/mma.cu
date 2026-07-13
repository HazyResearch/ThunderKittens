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
    test_info this_result;
    const std::string scale_name = std::is_same_v<Scale, kittens::fp8e4m3> ? "e4m3" : "e8m0";
    this_result.label = "tcgen05_st_st_mm_ABt_k64=nvfp4_" + scale_name;

    const fp4_packed one = std::bit_cast<fp4_packed>(uint8_t(0x22));
    std::vector<fp4_packed> h_a(M * NVFP4_K64_PACKED, one);
    std::vector<fp4_packed> h_b(N * NVFP4_K64_PACKED, one);
    std::vector<float> h_o(M * N, 0.0f);
    std::vector<float> h_ref(M * N, 64.0f * 1.0f + 64.0f * 2.0f);

    fp4_packed *d_a, *d_b;
    float *d_o;
    cudaMalloc(&d_a, h_a.size() * sizeof(fp4_packed));
    cudaMalloc(&d_b, h_b.size() * sizeof(fp4_packed));
    cudaMalloc(&d_o, h_o.size() * sizeof(float));
    CudaCheckError();
    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(fp4_packed), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(fp4_packed), cudaMemcpyHostToDevice);
    cudaMemset(d_o, 0, h_o.size() * sizeof(float));
    CudaCheckError();

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
    cudaMemcpy(h_o.data(), d_o, h_o.size() * sizeof(float), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    int bad_idx = -1;
    for (int i = 0; i < h_o.size(); i++) {
        if (std::abs(h_o[i] - h_ref[i]) > NVFP4_K64_TOL) {
            good = false;
            bad_idx = i;
            break;
        }
    }
    std::cout << "test `" << this_result.label << "`";
    if(good) std::cout << " -- PASSED" << std::endl;
    else     std::cout << " ----- ALERT! FAILED test `" << this_result.label
                       << "` first mismatch got " << h_o[bad_idx]
                       << " expected " << h_ref[bad_idx] << " -----" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);
    CudaCheckError();
    this_result.result = good ? test_result::PASSED : test_result::FAILED;
    results.push_back(this_result);
}
#endif

#ifdef KITTENS_SM103
using nvfp4_scale = kittens::fp8e8m0;
constexpr float NVFP4_K96_TOL = 1e-3f;
constexpr int NVFP4_K96_MMAS = 8;
constexpr int NVFP4_K96_LOGICAL = 96 * NVFP4_K96_MMAS;
constexpr int NVFP4_K96_PACKED = NVFP4_K96_LOGICAL / 2;
constexpr int NVFP4_K96_SCALE_COLS = 16 * NVFP4_K96_MMAS;

template<bool ACC, kittens::ducks::gl::all GL_A, kittens::ducks::gl::all GL_B, kittens::ducks::gl::all GL_O>
__cluster_dims__(2, 1, 1) __launch_bounds__(kittens::group<4>::GROUP_THREADS)
__global__ void tcgen05_nvfp4_k96_wrapper(
    const __grid_constant__ GL_A a_gl,
    const __grid_constant__ GL_B b_gl,
    const __grid_constant__ GL_O o_gl
) {
    constexpr int M = 128;
    constexpr int N = 256;
    using G = kittens::group<4>;
    using A_ST = kittens::st<fp4_packed, M, NVFP4_K96_PACKED>;
    using B_ST = kittens::st<fp4_packed, N / 2, NVFP4_K96_PACKED>;
    using D_TT = kittens::tt<float, M, N>;
    using D_RT = kittens::rt<float, M / G::GROUP_WARPS, N>;
    using S_ST = kittens::st<nvfp4_scale, 32, NVFP4_K96_SCALE_COLS, false>;
    using S_ATOM_ST = kittens::st<nvfp4_scale, 32, 16, false>;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::tma_swizzle_allocator al((int*)&__shm[0]);
    A_ST (&a_smem) = al.allocate<A_ST>();
    B_ST (&b_smem) = al.allocate<B_ST>();
    S_ST (&sa_smem) = al.allocate<S_ST>();
    S_ST (&sb_smem) = al.allocate<S_ST>();

    const int cta_id = kittens::cluster_ctarank();
    G::load(a_smem, a_gl, kittens::coord<A_ST>{cta_id, 0});
    G::load(b_smem, b_gl, kittens::coord<B_ST>{cta_id, 0});
    for (int idx = threadIdx.x; idx < S_ST::num_elements; idx += blockDim.x) {
        sa_smem.data[idx] = std::bit_cast<nvfp4_scale>(uint8_t(0x80));
        sb_smem.data[idx] = std::bit_cast<nvfp4_scale>(uint8_t(0x00));
    }
    __syncthreads();

    kittens::tensor_allocator<1, 2> tm_alloc{};

    D_TT d_tt = tm_alloc.template allocate<D_TT>(0);
    auto sa_tt = tm_alloc.template allocate<kittens::full_tt_fp8e8m0<NVFP4_K96_SCALE_COLS>>(256);
    auto sb_tt = tm_alloc.template allocate<kittens::full_tt_fp8e8m0<2 * NVFP4_K96_SCALE_COLS>>(256 + 4 * NVFP4_K96_MMAS);
    if (cta_id == 0 && kittens::warpid() == 0) {
        #pragma unroll
        for (int i = 0; i < NVFP4_K96_MMAS; ++i) {
            auto sa_tt_atom = sa_tt.template subtile<kittens::full_tt_fp8e8m0<16>>(i * 16);
            auto sb_tt_atom_0 = sb_tt.template subtile<kittens::full_tt_fp8e8m0<16>>(i * 32);
            auto sb_tt_atom_1 = sb_tt.template subtile<kittens::full_tt_fp8e8m0<16>>(i * 32 + 16);
            auto &sa_smem_atom = *reinterpret_cast<S_ATOM_ST *>(
                reinterpret_cast<uint64_t>(&sa_smem.data[0]) + i * 16 * 32);
            auto &sb_smem_atom = *reinterpret_cast<S_ATOM_ST *>(
                reinterpret_cast<uint64_t>(&sb_smem.data[0]) + i * 16 * 32);
            load_mxnv_scale_async2(sa_tt_atom, sa_smem_atom);
            load_mxnv_scale_async2(sb_tt_atom_0, sb_smem_atom);
            load_mxnv_scale_async2(sb_tt_atom_1, sb_smem_atom);
        }
        kittens::tensor_store_wait();
    }
    __syncthreads();

    __shared__ kittens::semaphore sem;
    kittens::warp::init_semaphore(sem, 0, 1);
    __syncthreads();

    if constexpr (ACC) {
        if (cta_id == 0 && kittens::warpid() == 0) {
            G::mm2_ABt<48>(d_tt, a_smem, b_smem, sa_tt, sb_tt, sem);
        }
        kittens::wait(sem, 0);
        if (cta_id == 0 && kittens::warpid() == 0) {
            G::mma2_ABt<48>(d_tt, a_smem, b_smem, sa_tt, sb_tt, sem);
        }
        kittens::wait(sem, 1);
    }
    else {
        if (cta_id == 0 && kittens::warpid() == 0) {
            G::mm2_ABt<48>(d_tt, a_smem, b_smem, sa_tt, sb_tt, sem);
        }
        kittens::wait(sem, 0);
    }

    D_RT d_reg;
    G::load_async(d_reg, d_tt);
    kittens::tensor_load_wait();
    G::store(o_gl, d_reg, kittens::coord<D_RT>{cta_id, 0});
}

template<bool ACC, kittens::ducks::gl::all GL_A, kittens::ducks::gl::all GL_B, kittens::ducks::gl::all GL_O>
__launch_bounds__(kittens::group<4>::GROUP_THREADS)
__global__ void tcgen05_nvfp4_k96_1cta_wrapper(
    const __grid_constant__ GL_A a_gl,
    const __grid_constant__ GL_B b_gl,
    const __grid_constant__ GL_O o_gl
) {
    constexpr int M = 128;
    constexpr int N = 256;
    using G = kittens::group<4>;
    using A_ST = kittens::st<fp4_packed, M, NVFP4_K96_PACKED>;
    using B_ST = kittens::st<fp4_packed, N, NVFP4_K96_PACKED>;
    using D_TT = kittens::tt<float, M, N>;
    using D_RT = kittens::rt<float, M / G::GROUP_WARPS, N>;
    using S_ST = kittens::st<nvfp4_scale, 32, NVFP4_K96_SCALE_COLS, false>;
    using S_ATOM_ST = kittens::st<nvfp4_scale, 32, 16, false>;

    extern __shared__ kittens::alignment_dummy __shm[];
    kittens::tma_swizzle_allocator al((int*)&__shm[0]);
    A_ST (&a_smem) = al.allocate<A_ST>();
    B_ST (&b_smem) = al.allocate<B_ST>();
    S_ST (&sa_smem) = al.allocate<S_ST>();
    S_ST (&sb_smem) = al.allocate<S_ST>();

    G::load(a_smem, a_gl, kittens::coord<A_ST>{0, 0});
    G::load(b_smem, b_gl, kittens::coord<B_ST>{0, 0});
    for (int idx = threadIdx.x; idx < S_ST::num_elements; idx += blockDim.x) {
        sa_smem.data[idx] = std::bit_cast<nvfp4_scale>(uint8_t(0x80));
        sb_smem.data[idx] = std::bit_cast<nvfp4_scale>(uint8_t(0x80));
    }
    __syncthreads();

    kittens::tensor_allocator<1, 1> tm_alloc{};

    D_TT d_tt = tm_alloc.template allocate<D_TT>(0);
    auto sa_tt = tm_alloc.template allocate<kittens::full_tt_fp8e8m0<NVFP4_K96_SCALE_COLS>>(256);
    auto sb_tt = tm_alloc.template allocate<kittens::full_tt_fp8e8m0<2 * NVFP4_K96_SCALE_COLS>>(256 + 4 * NVFP4_K96_MMAS);
    if (kittens::warpid() == 0) {
        #pragma unroll
        for (int i = 0; i < NVFP4_K96_MMAS; ++i) {
            auto sa_tt_atom = sa_tt.template subtile<kittens::full_tt_fp8e8m0<16>>(i * 16);
            auto sb_tt_atom_0 = sb_tt.template subtile<kittens::full_tt_fp8e8m0<16>>(i * 32);
            auto sb_tt_atom_1 = sb_tt.template subtile<kittens::full_tt_fp8e8m0<16>>(i * 32 + 16);
            auto &sa_smem_atom = *reinterpret_cast<S_ATOM_ST *>(
                reinterpret_cast<uint64_t>(&sa_smem.data[0]) + i * 16 * 32);
            auto &sb_smem_atom = *reinterpret_cast<S_ATOM_ST *>(
                reinterpret_cast<uint64_t>(&sb_smem.data[0]) + i * 16 * 32);
            load_mxnv_scale_async(sa_tt_atom, sa_smem_atom);
            load_mxnv_scale_async(sb_tt_atom_0, sb_smem_atom);
            load_mxnv_scale_async(sb_tt_atom_1, sb_smem_atom);
        }
        kittens::tensor_store_wait();
    }
    __syncthreads();

    __shared__ kittens::semaphore sem;
    kittens::warp::init_semaphore(sem, 0, 1);
    __syncthreads();

    if constexpr (ACC) {
        if (kittens::warpid() == 0) {
            G::mm_ABt<48>(d_tt, a_smem, b_smem, sa_tt, sb_tt, sem);
        }
        kittens::wait(sem, 0);
        if (kittens::warpid() == 0) {
            G::mma_ABt<48>(d_tt, a_smem, b_smem, sa_tt, sb_tt, sem);
        }
        kittens::wait(sem, 1);
    }
    else {
        if (kittens::warpid() == 0) {
            G::mm_ABt<48>(d_tt, a_smem, b_smem, sa_tt, sb_tt, sem);
        }
        kittens::wait(sem, 0);
    }

    D_RT d_reg;
    G::load_async(d_reg, d_tt);
    kittens::tensor_load_wait();
    G::store(o_gl, d_reg, kittens::coord<D_RT>{0, 0});
}

template<bool ACC>
static void run_nvfp4_k96_2cta(test_data &results) {
    constexpr int M = 256;
    constexpr int N = 256;
    test_info this_result;
    this_result.label = ACC ? "tcgen05_st_st_mma2_ABt_k96=nvfp4_e8m0"
                            : "tcgen05_st_st_mm2_ABt_k96=nvfp4_e8m0";

    const fp4_packed one = std::bit_cast<fp4_packed>(uint8_t(0x22));
    std::vector<fp4_packed> h_a(M * NVFP4_K96_PACKED, one);
    std::vector<fp4_packed> h_b(N * NVFP4_K96_PACKED, one);
    std::vector<float> h_o(M * N, 0.0f);
    const float expected = std::ldexp(float(NVFP4_K96_LOGICAL * 2 * (ACC ? 2 : 1)), -127);
    std::vector<float> h_ref(M * N, expected);

    fp4_packed *d_a, *d_b;
    float *d_o;
    cudaMalloc(&d_a, h_a.size() * sizeof(fp4_packed));
    cudaMalloc(&d_b, h_b.size() * sizeof(fp4_packed));
    cudaMalloc(&d_o, h_o.size() * sizeof(float));
    CudaCheckError();
    cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(fp4_packed), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(fp4_packed), cudaMemcpyHostToDevice);
    cudaMemset(d_o, 0, h_o.size() * sizeof(float));
    CudaCheckError();

    using A_ST = kittens::st<fp4_packed, 128, NVFP4_K96_PACKED>;
    using B_ST = kittens::st<fp4_packed, N / 2, NVFP4_K96_PACKED>;
    using GL_A = kittens::gl<fp4_packed, 1, 1, M, NVFP4_K96_PACKED, A_ST>;
    using GL_B = kittens::gl<fp4_packed, 1, 1, N, NVFP4_K96_PACKED, B_ST>;
    using GL_O = kittens::gl<float, 1, 1, M, N>;
    GL_A a_gl(d_a, nullptr, nullptr, nullptr, nullptr);
    GL_B b_gl(d_b, nullptr, nullptr, nullptr, nullptr);
    GL_O o_gl(d_o, nullptr, nullptr, nullptr, nullptr);

    cudaFuncSetAttribute(
        tcgen05_nvfp4_k96_wrapper<ACC, GL_A, GL_B, GL_O>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kittens::MAX_SHARED_MEMORY - 1024
    );
    kittens::LaunchConfig<true> launch_config(
        dim3(2), dim3(kittens::group<4>::GROUP_THREADS), kittens::MAX_SHARED_MEMORY - 1024, nullptr, dim3(2)
    );
    cudaLaunchKernelEx(
        launch_config, tcgen05_nvfp4_k96_wrapper<ACC, GL_A, GL_B, GL_O>, a_gl, b_gl, o_gl
    );
    CudaCheckError();
    cudaMemcpy(h_o.data(), d_o, h_o.size() * sizeof(float), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    int bad_idx = -1;
    for (int i = 0; i < h_o.size(); i++) {
        if (std::abs(h_o[i] - h_ref[i]) > 1e-3f) {
            good = false;
            bad_idx = i;
            break;
        }
    }
    std::cout << "test `" << this_result.label << "`";
    if(good) std::cout << " -- PASSED" << std::endl;
    else     std::cout << " ----- ALERT! FAILED test `" << this_result.label
                       << "` first mismatch got " << h_o[bad_idx]
                       << " expected " << h_ref[bad_idx] << " -----" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);
    CudaCheckError();
    this_result.result = good ? test_result::PASSED : test_result::FAILED;
    results.push_back(this_result);
}

template<bool ACC>
static void run_nvfp4_k96_1cta(test_data &results) {
    constexpr int N = 256;
    constexpr int M_1CTA = 128;
    test_info one_cta_result;
    one_cta_result.label = ACC ? "tcgen05_st_st_mma_ABt_k96=nvfp4_e8m0"
                               : "tcgen05_st_st_mm_ABt_k96=nvfp4_e8m0";

    const fp4_packed one = std::bit_cast<fp4_packed>(uint8_t(0x22));
    std::vector<fp4_packed> h_a_1cta(M_1CTA * NVFP4_K96_PACKED, one);
    std::vector<fp4_packed> h_b_1cta(N * NVFP4_K96_PACKED, one);
    std::vector<float> h_o_1cta(M_1CTA * N, 0.0f);
    std::vector<float> h_ref_1cta(M_1CTA * N, float(NVFP4_K96_LOGICAL * 4 * (ACC ? 2 : 1)));

    fp4_packed *d_a_1cta, *d_b_1cta;
    float *d_o_1cta;
    cudaMalloc(&d_a_1cta, h_a_1cta.size() * sizeof(fp4_packed));
    cudaMalloc(&d_b_1cta, h_b_1cta.size() * sizeof(fp4_packed));
    cudaMalloc(&d_o_1cta, h_o_1cta.size() * sizeof(float));
    CudaCheckError();
    cudaMemcpy(d_a_1cta, h_a_1cta.data(), h_a_1cta.size() * sizeof(fp4_packed), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_1cta, h_b_1cta.data(), h_b_1cta.size() * sizeof(fp4_packed), cudaMemcpyHostToDevice);
    cudaMemset(d_o_1cta, 0, h_o_1cta.size() * sizeof(float));
    CudaCheckError();

    using A_ST_1CTA = kittens::st<fp4_packed, M_1CTA, NVFP4_K96_PACKED>;
    using B_ST_1CTA = kittens::st<fp4_packed, N, NVFP4_K96_PACKED>;
    using GL_A_1CTA = kittens::gl<fp4_packed, 1, 1, M_1CTA, NVFP4_K96_PACKED, A_ST_1CTA>;
    using GL_B_1CTA = kittens::gl<fp4_packed, 1, 1, N, NVFP4_K96_PACKED, B_ST_1CTA>;
    using GL_O_1CTA = kittens::gl<float, 1, 1, M_1CTA, N>;
    GL_A_1CTA a_gl_1cta(d_a_1cta, nullptr, nullptr, nullptr, nullptr);
    GL_B_1CTA b_gl_1cta(d_b_1cta, nullptr, nullptr, nullptr, nullptr);
    GL_O_1CTA o_gl_1cta(d_o_1cta, nullptr, nullptr, nullptr, nullptr);

    cudaFuncSetAttribute(
        tcgen05_nvfp4_k96_1cta_wrapper<ACC, GL_A_1CTA, GL_B_1CTA, GL_O_1CTA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kittens::MAX_SHARED_MEMORY - 1024
    );
    tcgen05_nvfp4_k96_1cta_wrapper<ACC, GL_A_1CTA, GL_B_1CTA, GL_O_1CTA><<<
        dim3(1), dim3(kittens::group<4>::GROUP_THREADS), kittens::MAX_SHARED_MEMORY - 1024
    >>>(a_gl_1cta, b_gl_1cta, o_gl_1cta);
    CudaCheckError();
    cudaMemcpy(h_o_1cta.data(), d_o_1cta, h_o_1cta.size() * sizeof(float), cudaMemcpyDeviceToHost);
    CudaCheckError();

    bool good = true;
    int bad_idx = -1;
    for (int i = 0; i < h_o_1cta.size(); i++) {
        if (std::abs(h_o_1cta[i] - h_ref_1cta[i]) > NVFP4_K96_TOL) {
            good = false;
            bad_idx = i;
            break;
        }
    }
    std::cout << "test `" << one_cta_result.label << "`";
    if(good) std::cout << " -- PASSED" << std::endl;
    else     std::cout << " ----- ALERT! FAILED test `" << one_cta_result.label
                       << "` first mismatch got " << h_o_1cta[bad_idx]
                       << " expected " << h_ref_1cta[bad_idx] << " -----" << std::endl;

    cudaFree(d_a_1cta);
    cudaFree(d_b_1cta);
    cudaFree(d_o_1cta);
    CudaCheckError();
    one_cta_result.result = good ? test_result::PASSED : test_result::FAILED;
    results.push_back(one_cta_result);
}

static void run_nvfp4_k96(test_data &results) {
    run_nvfp4_k96_2cta<false>(results);
    run_nvfp4_k96_2cta<true>(results);
    run_nvfp4_k96_1cta<false>(results);
    run_nvfp4_k96_1cta<true>(results);
}
#endif

}

void group::mma::tensor::mma::tests(test_data &results) {
    std::cout << " ----- Starting ops/group/mma/tensor/mma tests! -----\n" << std::endl;
    run_type<kittens::bf16>(results, "bf16");
    run_type<kittens::half>(results, "half");
    run_type<kittens::fp8e4m3>(results, "fp8e4m3");
    run_type<kittens::fp8e5m2>(results, "fp8e5m2");
#ifndef KITTENS_SM103
    run_type<kittens::int8>(results, "int8");
    run_type<kittens::uint8>(results, "uint8");
#endif
#ifdef KITTENS_SM10X
    run_nvfp4_k64<kittens::fp8e4m3>(results);
    run_nvfp4_k64<kittens::fp8e8m0>(results);
#endif
#ifdef KITTENS_SM103
    run_nvfp4_k96(results);
#endif
    std::cout << std::endl;
}

#endif
