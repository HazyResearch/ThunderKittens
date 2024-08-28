//#define TORCH_COMPILE

#include "include/kittens.cuh"

using namespace kittens;

// Number of batches per SM
#define B_TILE 8 //16
// Number of heads per SM
#define H_TILE 8 //32

#define NUM_WORKERS 2
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

// We need to create a real and imag tile since we're passing thru TMA
#define tile_smem_bf_2x2 kittens::st_bf<2, 2>


// launch_bounds(Max threads per block, min blocks per SM)
__global__ __launch_bounds__(NUM_THREADS, 1) 
void fftconv_tk(const bf16 *u_real, const bf16 *u_imag, const CUtensorMap* tma_kf_real, const CUtensorMap* tma_kf_imag,
            const bf16 *f_real, const bf16 *f_imag, const bf16 *finv_real, const bf16 *finv_imag, 
            const bf16 *tw_real, const bf16 *tw_imag, const bf16 *twinv_real, const bf16 *twinv_imag, 
            bf16 *o, 
            int h, int n, int n1){
    
    auto warpid = kittens::warpid();

    // Every block loads same seq tile
    int h_stride = n;
    int b_stride = h * n;
    // number of batches per warp
    int batches = (B_TILE + NUM_WORKERS - 1) / NUM_WORKERS;
    // One warp needs to handle all heads of certain batch? - we can't tile over both batches/heads unless
    // multiple warps can take care of same batches but diff heads
    int h_start = (blockIdx.y * H_TILE);
    int b_start = (blockIdx.x * B_TILE) + (warpid * batches);
    // Doing 32x32 tiles
    // Bc we're using HMMA instructions, we need an fl accum reg for MMA
    kittens::rt_cmplx_fl<2, 2> mma_reg;
    // Separate reg to hold accumulated X values (in bf)
    kittens::rt_cmplx_bf<2, 2> accum/*, a_tr*/; // TODO eliminate use of a_tr and perform in-place transpose?

    // Row layout for element-wise mul
    kittens::rt_cmplx_bf<2, 2, ducks::rt_layout::row> k_reg, tw_reg, twinv_reg;
    // Col layout for mma b matrix
    kittens::rt_cmplx_bf<2, 2, ducks::rt_layout::col> b_reg, f_reg, finv_reg;
    
    kittens::rt_cmplx_bf<2, 2, ducks::rt_layout::row> ft_reg;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al((int*)&__shm[0]);

    // Non-TMA, loaded once at beginning
    kittens::st_cmplx_bf<2, 2> (&f_smem) = al.allocate<st_cmplx_bf<2, 2>>();
    kittens::st_cmplx_bf<2, 2> (&finv_smem) = al.allocate<st_cmplx_bf<2, 2>>();
    kittens::st_cmplx_bf<2, 2> (&tw_smem) = al.allocate<st_cmplx_bf<2, 2>>();
    kittens::st_cmplx_bf<2, 2> (&twinv_smem) = al.allocate<st_cmplx_bf<2, 2>>();

    // TMA - don't have it implemented for complex tiles yet so split into real and imag for now
    tile_smem_bf_2x2 (&kf_real_s)[2] = al.allocate<tile_smem_bf_2x2, 2>();
    tile_smem_bf_2x2 (&kf_imag_s)[2] = al.allocate<tile_smem_bf_2x2, 2>();
    
    // TODO TMA x too?
    //kittens::st_cmplx_bf<2, 2, ducks::st_layout::swizzle> (&x_smem)[2][NUM_WORKERS] = al.allocate<st_cmplx_bf<2, 2, ducks::st_layout::swizzle>, 2, NUM_WORKERS>();

    //if(threadIdx.x ==0 && blockIdx.x == 0) printf("%llu\n", (uint64_t)(&x_smem) - (uint64_t)(&__shm[0]));

    // pipelining
    int k_tic = 0, k_toc = 1;
    __shared__ kittens::barrier bar;
    int k_phasebit = 0;
    // 2 dims for each ST
    if (threadIdx.x == 0) {
        kittens::init_barrier(bar, 0, 1);
    }
    //__syncthreads();

    // Global loads
    if (warpid == 0) {
        kittens::load(f_smem, f_real, f_imag, n1, n1);
        kittens::load(finv_smem, finv_real, finv_imag, n1, n1);
        kittens::load(tw_smem, tw_real, tw_imag, n1, n1);
        kittens::load(twinv_smem, twinv_real, twinv_imag, n1, n1);
    }

    if (warpid == 0) {
        // TODO eventually add x smem
        tma::expect_bytes(bar,
            size_bytes<typeof(kf_real_s[k_tic])> +
            size_bytes<typeof(kf_imag_s[k_tic])>
        );
        // The tile index is the head index
        tma::load_async(kf_real_s[k_tic], tma_kf_real, bar, h_start);
        tma::load_async(kf_imag_s[k_tic], tma_kf_imag, bar, h_start);
    }
    __syncthreads();

    kittens::load(f_reg, f_smem);
    kittens::load(finv_reg, finv_smem);
    kittens::load(tw_reg, tw_smem);
    kittens::load(twinv_reg, twinv_smem);

    for (int i = 0; i < H_TILE; i++, k_tic ^=1, k_toc ^=1) {
        kittens::wait(bar, k_phasebit);
        k_phasebit ^= 1;
        kittens::load(k_reg.real, kf_real_s[k_tic]);
        kittens::load(k_reg.imag, kf_imag_s[k_tic]);

        int k_start = (h_start + i) * h_stride;

        if (warpid == 0 && i < H_TILE - 1) {
            tma::expect_bytes(bar,
                size_bytes<typeof(kf_real_s[k_toc])>+
                size_bytes<typeof(kf_imag_s[k_toc])>
            );
            int next_tile = h_start + i+1;
            // Kick off load for next k, while we do computation on current set
            tma::load_async(kf_real_s[k_toc], tma_kf_real, bar, next_tile);
            tma::load_async(kf_imag_s[k_toc], tma_kf_imag, bar, next_tile);
        }

        for (int j = 0; j < batches; j++) {
            int u_start = ((b_start + j) * b_stride) + k_start;
            // X = F^T X
            swap_layout(ft_reg, f_reg);
            ft_reg = transpose_inplace(ft_reg);
            kittens::load(b_reg, u_real + u_start, u_imag + u_start, n1, n1);
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, ft_reg, b_reg, mma_reg);
            kittens::copy(accum, mma_reg);
            // X = X * tw
            kittens::mul(accum, accum, tw_reg);
            // X = XF
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, accum, f_reg, mma_reg);
            kittens::copy(accum, mma_reg);
            // X = X * K_f^T
            kittens::mul(accum, accum, k_reg);
            // X = XFinv
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, accum, finv_reg, mma_reg);
            kittens::copy(accum, mma_reg);
            // X = X^T * twinv
            accum = kittens::transpose_inplace(accum);
            kittens::mul(accum, accum, twinv_reg);
            // Y = XFinv
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, accum, finv_reg, mma_reg);
            kittens::copy(accum, mma_reg);
            // Write Y^T to HBM
            accum = kittens::transpose_inplace(accum);
            kittens::store(o + ((b_start + j) * b_stride) + k_start, accum.real, n1);
        }
    }
}

void launch_fftconv_tk(const bf16 *u_real, const bf16 *u_imag, const CUtensorMap* tma_kf_real, const CUtensorMap* tma_kf_imag, 
                        const bf16 *f_real, const bf16 *f_imag, const bf16 *finv_real, const bf16 *finv_imag, 
                        const bf16 *tw_real, const bf16 *tw_imag, const bf16 *twinv_real, const bf16 *twinv_imag, 
                        bf16 *o, 
                        int b, int h, int n, int n1) {
    // Multiple warps
    const dim3 block_dim{
        (unsigned int)(NUM_THREADS)
    };
    const dim3 grid_dim{
        (unsigned int)(b + B_TILE - 1) / B_TILE,
        (unsigned int)(h + H_TILE - 1) / H_TILE
    };

    
    // Complex shared tile size (bytes)
    int st_size = (2 * (2 * 2 * 256 * 4));
    // fft and twiddles (4), 2 kf smem, 2*NUM_WORKERS x_smem
    unsigned long mem_size = (4 * st_size) + (2 * st_size);

    //printf("%lu\n", mem_size);

    cudaFuncSetAttribute(
        fftconv_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    fftconv_tk<<<grid_dim, block_dim, mem_size>>>(u_real, u_imag, tma_kf_real, tma_kf_imag, 
                                                f_real, f_imag, finv_real, finv_imag, 
                                                tw_real, tw_imag, twinv_real, twinv_imag, 
                                                o, 
                                                h, n, n1);
}

// For launching from PyTorch
#ifdef TORCH_COMPILE
#include "include/common/pyutils/torch_helpers.cuh"
torch::Tensor fftconv_tk(
    const torch::Tensor u_real,
    const torch::Tensor u_imag,
    const torch::Tensor kf_real,
    const torch::Tensor kf_imag,
    const torch::Tensor f_real,
    const torch::Tensor f_imag,
    const torch::Tensor finv_real,
    const torch::Tensor finv_imag,
    const torch::Tensor tw_real,
    const torch::Tensor tw_imag,
    const torch::Tensor twinv_real,
    const torch::Tensor twinv_imag,
    int B,
    int H,
    int N,
    int N1
) {
    CHECK_INPUT(u_real);
    CHECK_INPUT(u_imag);
    CHECK_INPUT(kf_real);
    CHECK_INPUT(kf_imag);
    CHECK_INPUT(f_real);
    CHECK_INPUT(f_imag);
    CHECK_INPUT(finv_real);
    CHECK_INPUT(finv_imag);
    CHECK_INPUT(tw_real);
    CHECK_INPUT(tw_imag);
    CHECK_INPUT(twinv_real);
    CHECK_INPUT(twinv_imag);

    TORCH_CHECK(u_real.size(0) == B && u_real.size(1) == H 
        && u_real.size(2) == N1 && u_real.size(3) == N1, "u_real correct shape?");
    TORCH_CHECK(u_imag.size(0) == B && u_imag.size(1) == H 
        && u_imag.size(2) == N1 && u_imag.size(3) == N1, "u_imag correct shape?");
    TORCH_CHECK(kf_real.size(0) == H && kf_real.size(1) == N1 && kf_real.size(2) == N1, "kf_real correct shape?");
    TORCH_CHECK(kf_imag.size(0) == H && kf_imag.size(1) == N1 && kf_imag.size(2) == N1, "kf_imag correct shape?");
    TORCH_CHECK(f_real.size(0) == N1 && f_real.size(1) == N1, "f_real correct shape?");
    TORCH_CHECK(f_imag.size(0) == N1 && f_imag.size(1) == N1, "f_imag correct shape?");
    TORCH_CHECK(finv_real.size(0) == N1 && finv_real.size(1) == N1, "finv_real correct shape?");
    TORCH_CHECK(finv_imag.size(0) == N1 && finv_imag.size(1) == N1, "finv_imag correct shape?");
    TORCH_CHECK(tw_real.size(0) == N1 && tw_real.size(1) == N1, "tw_real correct shape?");
    TORCH_CHECK(tw_imag.size(0) == N1 && tw_imag.size(1) == N1, "tw_imag correct shape?");
    TORCH_CHECK(twinv_real.size(0) == N1 && twinv_real.size(1) == N1, "twinv_real correct shape?");
    TORCH_CHECK(twinv_imag.size(0) == N1 && twinv_imag.size(1) == N1, "twinv_imag correct shape?");

    torch::Tensor out = torch::empty({B, H, N1, N1}, u_real.options());

    c10::BFloat16 *u_real_ptr               = u_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *u_imag_ptr               = u_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *kf_real_ptr              = kf_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *kf_imag_ptr              = kf_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *f_real_ptr               = f_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *f_imag_ptr               = f_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *finv_real_ptr            = finv_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *finv_imag_ptr            = finv_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *tw_real_ptr              = tw_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *tw_imag_ptr              = tw_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *twinv_real_ptr           = twinv_real.data_ptr<c10::BFloat16>();
    c10::BFloat16 *twinv_imag_ptr           = twinv_imag.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr                    = out.data_ptr<c10::BFloat16>();

    const bf16* u_real_bf                   = reinterpret_cast<const bf16*>(u_real_ptr);
    const bf16* u_imag_bf                   = reinterpret_cast<const bf16*>(u_imag_ptr);
    const bf16* kf_real_bf                  = reinterpret_cast<const bf16*>(kf_real_ptr);
    const bf16* kf_imag_bf                  = reinterpret_cast<const bf16*>(kf_imag_ptr);
    const bf16* f_real_bf                   = reinterpret_cast<const bf16*>(f_real_ptr);
    const bf16* f_imag_bf                   = reinterpret_cast<const bf16*>(f_imag_ptr);
    const bf16* finv_real_bf                = reinterpret_cast<const bf16*>(finv_real_ptr);
    const bf16* finv_imag_bf                = reinterpret_cast<const bf16*>(finv_imag_ptr);
    const bf16* tw_real_bf                  = reinterpret_cast<const bf16*>(tw_real_ptr);
    const bf16* tw_imag_bf                  = reinterpret_cast<const bf16*>(tw_imag_ptr);
    const bf16* twinv_real_bf               = reinterpret_cast<const bf16*>(twinv_real_ptr);
    const bf16* twinv_imag_bf               = reinterpret_cast<const bf16*>(twinv_imag_ptr);
          bf16* o_bf                        = reinterpret_cast<bf16*>(o_ptr);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    launch_fftconv_tk(
        u_real_bf, u_imag_bf, kf_real_bf, kf_imag_bf,
        f_real_bf, f_imag_bf, finv_real_bf, finv_imag_bf,
        tw_real_bf, tw_imag_bf, twinv_real_bf, twinv_imag_bf,
        o_bf,
        B, H, N, N1
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaStreamDestroy(stream);

    return out;
}
#else
#include "harness.impl"
#endif