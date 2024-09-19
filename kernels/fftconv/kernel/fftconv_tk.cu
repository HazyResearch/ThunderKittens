#define TORCH_COMPILE

#define TORCH_COMPILE

#include "include/kittens.cuh"
//#include <cooperative_groups.h>

using namespace kittens;

#define NUM_WORKERS 2
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

// We need to create a real and imag tile since we're passing thru TMA
#define tile_smem_bf_2x2 kittens::st_bf_2x2


// launch_bounds(Max threads per block, min blocks per SM)
__global__ __launch_bounds__(NUM_THREADS, 1)
void fftconv_tk(const CUtensorMap* tma_u_real, const CUtensorMap* tma_u_imag, const CUtensorMap* tma_kf_real, const CUtensorMap* tma_kf_imag,
            const bf16 *f_real, const bf16 *f_imag, const bf16 *finv_real, const bf16 *finv_imag, 
            const bf16 *tw_real, const bf16 *tw_imag, const bf16 *twinv_real, const bf16 *twinv_imag, 
            CUtensorMap* tma_o, 
            int h, int n, int n1, int B_TILE, int H_TILE){
    
    auto warpid = kittens::warpid();

    // Every block loads same seq tile
    int h_stride = n;
    int b_stride = h * n;
    // number of batches per warp
    int batches = (B_TILE + NUM_WORKERS - 1) / NUM_WORKERS;
    // One warp needs to handle all heads of certain batch? - we can't tile over both batches/heads unless
    // multiple warps can take care of same batches but diff heads
    int h_start = (blockIdx.y * H_TILE);
    int b_start = (blockIdx.x * B_TILE);
    // Doing 32x32 tiles
    // Bc we're using HMMA instructions, we need an fl accum reg for MMA
    kittens::rt_cmplx_fl<2, 2> mma_reg;
    // Separate reg to hold accumulated X values (in bf)
    kittens::rt_cmplx_bf<2, 2> accum;

    // Row layout for element-wise mul
    kittens::rt_cmplx_bf<2, 2, ducks::rt_layout::row> k_reg, tw_reg, twinv_reg;
    // Col layout for mma b matrix
    kittens::rt_cmplx_bf<2, 2, ducks::rt_layout::col> b_reg, f_reg, finv_reg;
    
    kittens::rt_cmplx_bf<2, 2, ducks::rt_layout::row> ft_reg;

    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    tma_swizzle_allocator al((int*)&__shm[0]);

    kittens::st_cmplx_bf<2, 2> (&f_smem) = al.allocate<st_cmplx_bf<2, 2>>();
    kittens::st_cmplx_bf<2, 2> (&finv_smem) = al.allocate<st_cmplx_bf<2, 2>>();
    kittens::st_cmplx_bf<2, 2> (&tw_smem) = al.allocate<st_cmplx_bf<2, 2>>();
    kittens::st_cmplx_bf<2, 2> (&twinv_smem) = al.allocate<st_cmplx_bf<2, 2>>();

    // TMA - don't have it implemented for complex tiles yet so split into real and imag for now
    tile_smem_bf_2x2 (&kf_real_s)[2] = al.allocate<tile_smem_bf_2x2, 2>();
    tile_smem_bf_2x2 (&kf_imag_s)[2] = al.allocate<tile_smem_bf_2x2, 2>();
    
    tile_smem_bf_2x2 (&u_real_s)[2][NUM_WORKERS] = al.allocate<tile_smem_bf_2x2, 2, NUM_WORKERS>();
    tile_smem_bf_2x2 (&u_imag_s)[2][NUM_WORKERS] = al.allocate<tile_smem_bf_2x2, 2, NUM_WORKERS>();

    tile_smem_bf_2x2 (&o_s)[2][NUM_WORKERS] = al.allocate<tile_smem_bf_2x2, 2, NUM_WORKERS>();

    //if(threadIdx.x ==0 && blockIdx.x == 0) printf("%llu\n", (uint64_t)(&u_imag_s[1][1]) - (uint64_t)(&__shm[0]));


    // Global loads - reuse kf and u smem
    if (warpid == 0) {
        kittens::load(f_smem, f_real, f_imag, n1, n1);
        kittens::load(finv_smem, finv_real, finv_imag, n1, n1);
        kittens::load(tw_smem, tw_real, tw_imag, n1, n1);
        kittens::load(twinv_smem, twinv_real, twinv_imag, n1, n1);
        // kittens::load(u_real_s[0][0], f_real, n1);
        // kittens::load(u_real_s[0][1], f_imag, n1);
        // kittens::load(u_real_s[1][0], finv_real, n1);
        // kittens::load(u_real_s[1][1], finv_imag, n1);
        // kittens::load(u_imag_s[0][0], tw_real, n1);
        // kittens::load(u_imag_s[0][1], tw_imag, n1);
        // kittens::load(u_imag_s[1][0], twinv_real, n1);
        // kittens::load(u_imag_s[1][1], twinv_imag, n1);
    }
    __syncthreads();

    // pipelining
    int k_tic = 0, k_toc = 1;
    int u_tic = 0, u_toc = 1;
    __shared__ kittens::barrier k_arrived, u_arrived;
    // 2 dims for each ST
    if (threadIdx.x == 0) {
        kittens::init_barrier(k_arrived, 0, 1);
        kittens::init_barrier(u_arrived, 0, 1);

        tma::expect_bytes(k_arrived,
            size_bytes<tile_smem_bf_2x2> +
            size_bytes<tile_smem_bf_2x2>
        );
        tma::load_async(kf_real_s[0], tma_kf_real, k_arrived, h_start);
        tma::load_async(kf_imag_s[0], tma_kf_imag, k_arrived, h_start);
    }
    __syncthreads();

    kittens::load(f_reg, f_smem);
    kittens::load(finv_reg, finv_smem);
    kittens::load(tw_reg, tw_smem);
    kittens::load(twinv_reg, twinv_smem);
    // kittens::load(f_reg, f_real, f_imag, n1, n1);
    // kittens::load(finv_reg, finv_real, finv_imag, n1, n1);
    // kittens::load(tw_reg, tw_real, tw_imag, n1, n1);
    // kittens::load(twinv_reg, twinv_real, twinv_imag, n1, n1);
    // kittens::load(f_reg.real, u_real_s[0][0]);
    // kittens::load(f_reg.imag, u_real_s[0][1]);
    // kittens::load(finv_reg.real, u_real_s[1][0]);
    // kittens::load(finv_reg.imag, u_real_s[1][1]);
    // kittens::load(tw_reg.real, u_imag_s[0][0]);
    // kittens::load(tw_reg.imag, u_imag_s[0][1]);
    // kittens::load(twinv_reg.real, u_imag_s[1][0]);
    // kittens::load(twinv_reg.imag, u_imag_s[1][1]);
    __syncthreads();
    

    for (int i = 0; i < H_TILE; i++, k_tic ^=1, k_toc ^=1) {
        kittens::wait(k_arrived, k_tic);
        __syncthreads();

        kittens::load(k_reg.real, kf_real_s[k_tic]);
        kittens::load(k_reg.imag, kf_imag_s[k_tic]);

        if (warpid == 0){
            if (i+1 < H_TILE) {
                tma::expect_bytes(k_arrived,
                size_bytes<tile_smem_bf_2x2> +
                size_bytes<tile_smem_bf_2x2>
                );
                int next_tile = h_start + i+1;
                tma::load_async(kf_real_s[k_toc], tma_kf_real, k_arrived, next_tile);
                tma::load_async(kf_imag_s[k_toc], tma_kf_imag, k_arrived, next_tile);
            }
            
            tma::expect_bytes(u_arrived,
                size_bytes<tile_smem_bf_2x2, NUM_WORKERS> +
                size_bytes<tile_smem_bf_2x2, NUM_WORKERS>
            );
            for (int w = 0; w < NUM_WORKERS; w++) {
                tma::load_async(u_real_s[w][0], tma_u_real, u_arrived, b_start + w*batches, h_start + i);
                tma::load_async(u_imag_s[w][0], tma_u_imag, u_arrived, b_start + w*batches, h_start + i);
            }
        }
        u_tic = 0, u_toc = 1;
        for (int j = 0; j < batches; j++, u_tic ^= 1, u_toc ^= 1) {
            // We track arrivals for both tic and toc, so the phase bit is half the number of iterations
            kittens::wait(u_arrived, u_tic); //1, 0
            // Not the problem
            __syncthreads();

            if (warpid == 0 && j+1 < batches) {
                tma::expect_bytes(u_arrived,
                    size_bytes<tile_smem_bf_2x2, NUM_WORKERS> +
                    size_bytes<tile_smem_bf_2x2, NUM_WORKERS>
                );
                for (int w = 0; w < NUM_WORKERS; w++) {
                    tma::load_async(u_real_s[w][u_toc], tma_u_real, u_arrived, (b_start + w*batches) + j+1, h_start + i);
                    tma::load_async(u_imag_s[w][u_toc], tma_u_imag, u_arrived, (b_start + w*batches) + j+1, h_start + i);
                }
            }

            // X = F^T X
            swap_layout(ft_reg, f_reg);
            ft_reg = transpose_inplace(ft_reg);
            kittens::load(b_reg.real, u_real_s[warpid][u_tic]);
            kittens::load(b_reg.imag, u_imag_s[warpid][u_tic]);
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
            kittens::store(o_s[warpid][u_tic], accum.real);
            
            if (j>0) tma::store_async_wait();
            if (warpid == 0) {
                for (int w = 0; w < NUM_WORKERS; w++) {
                    tma::store_async(tma_o, o_s[w][u_tic], (b_start + w*batches) + j, h_start+i);
                }
                tma::store_commit_group();
            } tma::store_async_wait();
            //__syncthreads();
        }
        //__syncthreads();
    }
}

void launch_fftconv_tk(const CUtensorMap* tma_u_real, const CUtensorMap* tma_u_imag, const CUtensorMap* tma_kf_real, const CUtensorMap* tma_kf_imag, 
                        const bf16 *f_real, const bf16 *f_imag, const bf16 *finv_real, const bf16 *finv_imag, 
                        const bf16 *tw_real, const bf16 *tw_imag, const bf16 *twinv_real, const bf16 *twinv_imag, 
                        CUtensorMap* tma_o, 
                        int B, int H, int N, int N1) {
    // Multiple warps
    const dim3 blockDim{
        (unsigned int)(NUM_THREADS)
    };
    dim3 gridDim;
    int B_TILE;
    int H_TILE;

    if (B >= 8 && (B % 8) == 0 && H >= 8 && (H % 8) == 0) {
        gridDim.x = B / 8;
        gridDim.y = H / 8;
        B_TILE = 8, H_TILE = 8;
    } else if (B >= 4 && (B % 4) == 0 && H >= 8 && (H % 8) == 0) {
        gridDim.x = B / 4;
        gridDim.y = H / 8;
        B_TILE = 4, H_TILE = 8;
    } else if ((H % 8) == 0) {
        gridDim.x = B;
        gridDim.y = H / 8;
        B_TILE = 1, H_TILE = 8;
    }
    
    // Complex shared tile size (bytes)
    unsigned long st_size = 2 * (2 * 2 * 256 * 2);
    //unsigned long st_size = (2 * (2 * 2 * 256 * 4));
    // fft and twiddles (4), 2 kf smem, 2*NUM_WORKERS x_smem, 2*NUM_WORKERS o_smem
    // We need to calculate SMEM better
    unsigned long mem_size = /*(4 * st_size) + */(2 * st_size) + (2*NUM_WORKERS * st_size) + (2*NUM_WORKERS * st_size) + 6000;
    //unsigned long mem_size = 108000;

    //printf("%lu\n", mem_size);

    cudaFuncSetAttribute(
        fftconv_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    fftconv_tk<<<gridDim, blockDim, mem_size>>>(tma_u_real, tma_u_imag, tma_kf_real, tma_kf_imag, 
                                                f_real, f_imag, finv_real, finv_imag, 
                                                tw_real, tw_imag, twinv_real, twinv_imag, 
                                                tma_o,
                                                H, N, N1, B_TILE, H_TILE);
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


    const CUtensorMap* tma_u_real = tma::allocate_and_create_tensor_map<tile_smem_bf_2x2>(u_real_bf, B, H);
    const CUtensorMap* tma_u_imag = tma::allocate_and_create_tensor_map<tile_smem_bf_2x2>(u_imag_bf, B, H);
    // H tiles worth of mem, its 1 2x2 ST wide
    const CUtensorMap* tma_kf_real = tma::allocate_and_create_tensor_map<tile_smem_bf_2x2>(kf_real_bf, H);
    const CUtensorMap* tma_kf_imag = tma::allocate_and_create_tensor_map<tile_smem_bf_2x2>(kf_imag_bf, H);
    CUtensorMap* tma_o = tma::allocate_and_create_tensor_map<tile_smem_bf_2x2>(o_bf, B, H);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    launch_fftconv_tk(
        tma_u_real, tma_u_imag, tma_kf_real, tma_kf_imag,
        f_real_bf, f_imag_bf, finv_real_bf, finv_imag_bf,
        tw_real_bf, tw_imag_bf, twinv_real_bf, twinv_imag_bf,
        tma_o,
        B, H, N, N1
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    cudaStreamDestroy(stream);

    return out;
}
#else
#include "harness.impl"
#endif