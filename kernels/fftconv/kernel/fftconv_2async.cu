//#define TORCH_COMPILE

#include "include/kittens.cuh"
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda/barrier>

using namespace kittens;

// Number of batches per SM
#define B_TILE 8 //16
// Number of heads per SM
#define H_TILE 8 //32

#define NUM_WORKERS 2
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)


// launch_bounds(Max threads per block, min blocks per SM)
__global__ __launch_bounds__(NUM_THREADS, 1) 
void fftconv_tk(const bf16 *u_real, const bf16 *u_imag, const bf16 *kf_real, const bf16 *kf_imag, 
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

    kittens::st_cmplx_bf<2, 2> (&f_smem) = al.allocate<st_cmplx_bf<2, 2>>();
    kittens::st_cmplx_bf<2, 2> (&finv_smem) = al.allocate<st_cmplx_bf<2, 2>>();
    kittens::st_cmplx_bf<2, 2> (&tw_smem) = al.allocate<st_cmplx_bf<2, 2>>();
    kittens::st_cmplx_bf<2, 2> (&twinv_smem) = al.allocate<st_cmplx_bf<2, 2>>();

    kittens::st_cmplx_bf<2, 2> (&kf_smem)[2] = al.allocate<st_cmplx_bf<2, 2>, 2>();
    kittens::st_cmplx_bf<2, 2> (&x_smem)[2][NUM_WORKERS] = al.allocate<st_cmplx_bf<2, 2>, 2, NUM_WORKERS>();

    //if(threadIdx.x ==0 && blockIdx.x == 0) printf("%llu\n", (uint64_t)(&x_smem) - (uint64_t)(&__shm[0]));

    // pipelining
    int k_tic = 0, k_toc = 1;
    int x_tic = 0, x_toc = 1;
    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> k_barrier;
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> x_barrier;
    if (threadIdx.x == 0) {
        init(&k_barrier, block.size());
        init(&x_barrier, block.size());
    }
    block.sync(); // Need to make sure none calls before setup.

    // Global loads
    if (warpid == 0) {
        kittens::load(f_smem, f_real, f_imag, n1, n1);
        kittens::load(finv_smem, finv_real, finv_imag, n1, n1);
        kittens::load(tw_smem, tw_real, tw_imag, n1, n1);
        kittens::load(twinv_smem, twinv_real, twinv_imag, n1, n1);
        load_async(kf_smem[k_tic], kf_real + (h_start*h_stride), kf_imag + (h_start*h_stride), n1, n1, k_barrier);
    }
    __syncthreads();

    kittens::load(f_reg, f_smem);
    kittens::load(finv_reg, finv_smem);
    kittens::load(tw_reg, tw_smem);
    kittens::load(twinv_reg, twinv_smem);

    for (int i = 0; i < H_TILE; i++, k_tic ^=1, k_toc ^=1) {
        k_barrier.arrive_and_wait();
        kittens::load(k_reg, kf_smem[k_tic]);

        int k_start = (h_start + i) * h_stride;
        int next_head = (h_start + i+1) * h_stride;

        if (i < H_TILE - 1) {
            if (warpid == 0) {
                // Kick off load for next k, while we do computation on current set
                kittens::load_async(kf_smem[k_toc], kf_real + next_head, kf_imag + next_head, n1, n1, k_barrier);
            }  
        }

        x_tic = 0;
        x_toc = 1;
        int u_initial = (b_start*b_stride) + k_start;
        load_async(x_smem[warpid][x_tic], u_real + u_initial, u_imag + u_initial, n1, n1, x_barrier);
        for (int j = 0; j < batches; j++, x_tic ^=1, x_toc ^=1) {
            // TODO remove barrier in inner loop
            x_barrier.arrive_and_wait();
            // Next batch but w/ same head
            int next_batch = ((b_start + j+1) * b_stride) + k_start;
            if (j < batches - 1) {
                // Kick off load for next x, while we do computation on current set
                kittens::load_async(x_smem[warpid][x_toc], u_real + next_batch, u_imag + next_batch, n1, n1, x_barrier);
            }
            // X = F^T X
            
            swap_layout(ft_reg, f_reg);
            ft_reg = transpose_inplace(ft_reg);

            kittens::load(b_reg, x_smem[warpid][x_tic]);
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, ft_reg, b_reg, mma_reg);
            kittens::copy(accum, mma_reg);
            // X = X * tw
            // kittens::load(a_reg, tw_smem[0]);
            kittens::mul(accum, accum, tw_reg);
            // X = XF
            //kittens::load(b_reg, f_smem[0]);
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, accum, f_reg, mma_reg);
            kittens::copy(accum, mma_reg);
            // X = X * K_f^T
            kittens::mul(accum, accum, k_reg);
            //kittens::load(a_reg, kf_real + k_start, kf_imag + k_start, n1, n1);
            //kittens::mul(accum, accum, a_reg);
            // X = XFinv
            //kittens::load(b_reg, finv_smem[0]);
            kittens::zero(mma_reg);
            kittens::mma_AB(mma_reg, accum, finv_reg, mma_reg);
            kittens::copy(accum, mma_reg);
            // X = X^T * twinv
            //kittens::transpose_sep(a_tr, accum);
            accum = kittens::transpose_inplace(accum);
            //kittens::load(a_reg, twinv_smem[0]);
            //kittens::mul(a_tr, a_tr, twinv_reg);
            kittens::mul(accum, accum, twinv_reg);
            // Y = XFinv
            //kittens::load(b_reg, finv_smem[0]);
            kittens::zero(mma_reg);
            //kittens::mma_AB(mma_reg, a_tr, finv_reg, mma_reg);
            kittens::mma_AB(mma_reg, accum, finv_reg, mma_reg);
            kittens::copy(accum, mma_reg);
            // Write Y^T to HBM
            //kittens::transpose_sep(a_tr, accum);
            accum = kittens::transpose_inplace(accum);
            //kittens::store(o + ((b_start + j) * b_stride) + k_start, a_tr.real, n1);
            kittens::store(o + ((b_start + j) * b_stride) + k_start, accum.real, n1);
        }
    }
}

void launch_fftconv_tk(const bf16 *u_real, const bf16 *u_imag, const bf16 *kf_real, const bf16 *kf_imag, 
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
    unsigned long mem_size = (4 * st_size) + (2 * st_size) + (2 * NUM_WORKERS * st_size);

    //printf("%lu\n", mem_size);

    cudaFuncSetAttribute(
        fftconv_tk,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    fftconv_tk<<<grid_dim, block_dim, mem_size>>>(u_real, u_imag, kf_real, kf_imag, 
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
#include "harness_async.impl"
#endif