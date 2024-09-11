// #define TORCH_COMPILE 

#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <cuda/barrier>

#define NUM_WORKERS_NORM (2) 
#define NUM_THREADS_NORM (NUM_WORKERS_NORM*kittens::WARP_THREADS)
     
const int TXT_D = 512;
const int IMG_D = 4080;  
const int HEAD_D = 128;
const int d_head_tile = HEAD_D / kittens::TILE_DIM;

using namespace kittens;
#define vec_smem_1xHEAD_D sv_bf<d_head_tile>


// rms norm 
__global__ __launch_bounds__(NUM_THREADS_NORM, 1)
void flux_rmsnorm(
    const bf16* __img_q,
    const bf16* __img_k,
    const bf16* __txt_q,
    const bf16* __txt_k,
    const bf16* __rms_norm_scale_img_q,
    const bf16* __rms_norm_scale_img_k,
    const bf16* __rms_norm_scale_txt_q,
    const bf16* __rms_norm_scale_txt_k,
    bf16* __o_q,
    bf16* __o_k
) {
    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();

    auto txt_offset = TXT_D*HEAD_D;

    // shared memory setup to load from hbm
    const bf16 *img_q_g          = reinterpret_cast<const bf16*>(__img_q)+blockIdx.x*(IMG_D*HEAD_D);
    const bf16 *txt_q_g          = reinterpret_cast<const bf16*>(__txt_q)+blockIdx.x*(TXT_D*HEAD_D);
    const bf16 *img_k_g          = reinterpret_cast<const bf16*>(__img_k)+blockIdx.x*(IMG_D*HEAD_D);
    const bf16 *txt_k_g          = reinterpret_cast<const bf16*>(__txt_k)+blockIdx.x*(TXT_D*HEAD_D);
          
    const bf16 *rms_norm_scale_g_img_q = reinterpret_cast<const bf16*>(__rms_norm_scale_img_q);
    const bf16 *rms_norm_scale_g_img_k = reinterpret_cast<const bf16*>(__rms_norm_scale_img_k);
    const bf16 *rms_norm_scale_g_txt_q = reinterpret_cast<const bf16*>(__rms_norm_scale_txt_q);
    const bf16 *rms_norm_scale_g_txt_k = reinterpret_cast<const bf16*>(__rms_norm_scale_txt_k);

          bf16 *o_q_g            = reinterpret_cast<bf16*>(__o_q)+blockIdx.x*((IMG_D+TXT_D)*HEAD_D);
          bf16 *o_k_g            = reinterpret_cast<bf16*>(__o_k)+blockIdx.x*((IMG_D+TXT_D)*HEAD_D);

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    vec_smem_1xHEAD_D (&img_q_s)   [2][NUM_WORKERS_NORM] = al.allocate<vec_smem_1xHEAD_D,2,NUM_WORKERS_NORM>();
    vec_smem_1xHEAD_D (&img_k_s)   [2][NUM_WORKERS_NORM] = al.allocate<vec_smem_1xHEAD_D,2,NUM_WORKERS_NORM>();
    vec_smem_1xHEAD_D (&txt_q_s)   [2][NUM_WORKERS_NORM] = al.allocate<vec_smem_1xHEAD_D,2,NUM_WORKERS_NORM>();
    vec_smem_1xHEAD_D (&txt_k_s)   [2][NUM_WORKERS_NORM] = al.allocate<vec_smem_1xHEAD_D,2,NUM_WORKERS_NORM>();
    
    vec_smem_1xHEAD_D (&scratch_s) [2][NUM_WORKERS_NORM] = al.allocate<vec_smem_1xHEAD_D,2,NUM_WORKERS_NORM>();
    
    vec_smem_1xHEAD_D (&rms_norm_scale_s_img_q) = al.allocate<vec_smem_1xHEAD_D>(); 
    vec_smem_1xHEAD_D (&rms_norm_scale_s_img_k) = al.allocate<vec_smem_1xHEAD_D>(); 
    vec_smem_1xHEAD_D (&rms_norm_scale_s_txt_q) = al.allocate<vec_smem_1xHEAD_D>(); 
    vec_smem_1xHEAD_D (&rms_norm_scale_s_txt_k) = al.allocate<vec_smem_1xHEAD_D>(); 

    // global loads
    if (warpid == 0) { 
        load(rms_norm_scale_s_img_q, rms_norm_scale_g_img_q); 
        load(rms_norm_scale_s_img_k, rms_norm_scale_g_img_k); 
        load(rms_norm_scale_s_txt_q, rms_norm_scale_g_txt_q); 
        load(rms_norm_scale_s_txt_k, rms_norm_scale_g_txt_k); 
    }

    // pipelining
    int tic = 0, toc = 1;
    auto block = cooperative_groups::this_thread_block();
     __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier_cheat;
    if (threadIdx.x == 0) {init(&barrier_cheat, block.size());}
    block.sync(); // Need to make sure none calls before setup.
 
    bf16 mean_img_q = __float2bfloat16(0.0f);
    bf16 rrms_img_q = __float2bfloat16(0.0f);
    bf16 mean_img_k = __float2bfloat16(0.0f);
    bf16 rrms_img_k = __float2bfloat16(0.0f);

    bf16 mean_txt_q = __float2bfloat16(0.0f);
    bf16 rrms_txt_q = __float2bfloat16(0.0f);
    bf16 mean_txt_k = __float2bfloat16(0.0f);
    bf16 rrms_txt_k = __float2bfloat16(0.0f);

    load_async(img_q_s[warpid][tic], img_q_g + warpid*HEAD_D, HEAD_D, barrier_cheat);
    load_async(img_k_s[warpid][tic], img_k_g + warpid*HEAD_D, HEAD_D, barrier_cheat);
    load_async(txt_q_s[warpid][tic], txt_q_g + warpid*HEAD_D, HEAD_D, barrier_cheat);
    load_async(txt_k_s[warpid][tic], txt_k_g + warpid*HEAD_D, HEAD_D, barrier_cheat);
    __syncthreads();
    
    int n_blocks = IMG_D / NUM_WORKERS_NORM;
    int n_blocks_txt = TXT_D / NUM_WORKERS_NORM;
    for (int block = 0; block < n_blocks; block ++, tic ^=1, toc ^=1) {
        barrier_cheat.arrive_and_wait();  

        // kick off load for the next block
        if( block < n_blocks - 1 ) {
            auto next_idx = (block + 1)*NUM_WORKERS_NORM + warpid; 
            load_async(img_q_s[warpid][toc], img_q_g + next_idx*HEAD_D, HEAD_D, barrier_cheat);
            load_async(img_k_s[warpid][toc], img_k_g + next_idx*HEAD_D, HEAD_D, barrier_cheat);
        }

        if (block < n_blocks_txt - 1) {
            auto next_idx = (block + 1)*NUM_WORKERS_NORM + warpid; 
            load_async(txt_q_s[warpid][toc], txt_q_g + next_idx*HEAD_D, HEAD_D, barrier_cheat);
            load_async(txt_k_s[warpid][toc], txt_k_g + next_idx*HEAD_D, HEAD_D, barrier_cheat);
        }

        // rrms = torch.rsqrt(torch.mean(img_q**2, dim=-1, keepdim=True) + 1e-6)
        copy(scratch_s[warpid][tic], img_q_s[warpid][tic]);
        mul(scratch_s[warpid][tic], scratch_s[warpid][tic], scratch_s[warpid][tic]); // img_q**2
        sum(mean_img_q, scratch_s[warpid][tic]);
        mean_img_q = mean_img_q / __float2bfloat16(HEAD_D);
        rrms_img_q = __float2bfloat16(1 / sqrt(__bfloat162float(mean_img_q + __float2bfloat16(1e-06f))));

        copy(scratch_s[warpid][tic], img_k_s[warpid][tic]);
        mul(scratch_s[warpid][tic], scratch_s[warpid][tic], scratch_s[warpid][tic]); // img_q**2
        sum(mean_img_k, scratch_s[warpid][tic]);
        mean_img_k = mean_img_k / __float2bfloat16(HEAD_D);
        rrms_img_k = __float2bfloat16(1 / sqrt(__bfloat162float(mean_img_k + __float2bfloat16(1e-06f))));

        // img_q = (img_q * rrms) * q_img_rms_norm_scale;
        mul(img_q_s[warpid][tic], img_q_s[warpid][tic], rrms_img_q);
        mul(img_q_s[warpid][tic], img_q_s[warpid][tic], rms_norm_scale_s_img_q);
        mul(img_k_s[warpid][tic], img_k_s[warpid][tic], rrms_img_k);
        mul(img_k_s[warpid][tic], img_k_s[warpid][tic], rms_norm_scale_s_img_k);

        if ( block < n_blocks_txt ) {
            copy(scratch_s[warpid][tic], txt_q_s[warpid][tic]);
            mul(scratch_s[warpid][tic], scratch_s[warpid][tic], scratch_s[warpid][tic]); // img_q**2
            sum(mean_txt_q, scratch_s[warpid][tic]);
            mean_txt_q = mean_txt_q / __float2bfloat16(HEAD_D);
            rrms_txt_q = __float2bfloat16(1 / sqrt(__bfloat162float(mean_txt_q + __float2bfloat16(1e-06f))));

            copy(scratch_s[warpid][tic], txt_k_s[warpid][tic]);
            mul(scratch_s[warpid][tic], scratch_s[warpid][tic], scratch_s[warpid][tic]); // img_q**2
            sum(mean_txt_k, scratch_s[warpid][tic]);
            mean_txt_k = mean_txt_k / __float2bfloat16(HEAD_D);
            rrms_txt_k = __float2bfloat16(1 / sqrt(__bfloat162float(mean_txt_k + __float2bfloat16(1e-06f))));

            // img_q = (img_q * rrms) * q_img_rms_norm_scale;
            mul(txt_q_s[warpid][tic], txt_q_s[warpid][tic], rrms_txt_q);
            mul(txt_q_s[warpid][tic], txt_q_s[warpid][tic], rms_norm_scale_s_txt_q);
            mul(txt_k_s[warpid][tic], txt_k_s[warpid][tic], rrms_txt_k);
            mul(txt_k_s[warpid][tic], txt_k_s[warpid][tic], rms_norm_scale_s_txt_k);

            store(o_q_g + (block*NUM_WORKERS_NORM +warpid)*HEAD_D, txt_q_s[warpid][tic]); 
            store(o_k_g + (block*NUM_WORKERS_NORM +warpid)*HEAD_D, txt_k_s[warpid][tic]); 
        }

        // save output
        store(o_q_g + txt_offset + (block*NUM_WORKERS_NORM +warpid)*HEAD_D, img_q_s[warpid][tic]); 
        store(o_k_g + txt_offset + (block*NUM_WORKERS_NORM +warpid)*HEAD_D, img_k_s[warpid][tic]); 
    }
}

#ifdef TORCH_COMPILE
#include "common/pyutils/torch_helpers.cuh"
#include <iostream>
void fused_flux_rmsnorm(
    const torch::Tensor img_q, 
    const torch::Tensor img_k, 
    const torch::Tensor txt_q, 
    const torch::Tensor txt_k, 
    const torch::Tensor rms_scale_q_img, 
    const torch::Tensor rms_scale_k_img, 
    const torch::Tensor rms_scale_q_txt, 
    const torch::Tensor rms_scale_k_txt, 
    torch::Tensor out_q,
    torch::Tensor out_k
) {
    CHECK_INPUT(img_q);
    CHECK_INPUT(img_k);
    CHECK_INPUT(txt_q);
    CHECK_INPUT(txt_k);
    CHECK_INPUT(rms_scale_q_img);
    CHECK_INPUT(rms_scale_k_img);
    CHECK_INPUT(rms_scale_q_txt);
    CHECK_INPUT(rms_scale_k_txt);
    CHECK_INPUT(out_q);
    CHECK_INPUT(out_k);

    int batch = img_q.size(0);
    int heads = img_q.size(1);
    int head_dim = img_q.size(3);
    int img_seq_dim = img_q.size(2);
    int txt_seq_dim = txt_q.size(2);
    int seq_dim = img_seq_dim + txt_seq_dim;

    TORCH_CHECK(batch == out_q.size(0),    "Differing batch sizes for input and output?");
    TORCH_CHECK(seq_dim == out_q.size(2),  "Differing seq len for input and output?");
    TORCH_CHECK(head_dim == out_q.size(3),  "Differing head dim for input and output?");
    TORCH_CHECK(heads == out_q.size(1),    "Differing heads for input and output?");

    TORCH_CHECK(batch == out_k.size(0),    "Differing batch sizes for input and output?");
    TORCH_CHECK(seq_dim == out_k.size(2),  "Differing seq len for input and output?");
    TORCH_CHECK(head_dim == out_k.size(3),  "Differing head dim for input and output?");
    TORCH_CHECK(heads == out_k.size(1),    "Differing heads for input and output?");

    TORCH_CHECK(head_dim == rms_scale_q_img.size(0),  "Differing head_dim for input and rms_scale?");
    TORCH_CHECK(head_dim == rms_scale_k_img.size(0),  "Differing head_dim for input and rms_scale?");
    TORCH_CHECK(head_dim == rms_scale_q_txt.size(0),  "Differing head_dim for input and rms_scale?");
    TORCH_CHECK(head_dim == rms_scale_k_txt.size(0),  "Differing head_dim for input and rms_scale?");

    TORCH_CHECK(img_seq_dim % kittens::TILE_DIM == 0,  "sequence length is divisible by 16?");
    TORCH_CHECK(txt_seq_dim % kittens::TILE_DIM == 0,  "sequence length is divisible by 16?");
    TORCH_CHECK(seq_dim % kittens::TILE_DIM == 0,  "sequence length is divisible by 16?");
    TORCH_CHECK(head_dim % kittens::TILE_DIM == 0, "head dimension is divisible by 16?");

    // convert to bf16
    c10::BFloat16 *img_q_ptr  = img_q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *img_k_ptr  = img_k.data_ptr<c10::BFloat16>();
    c10::BFloat16 *txt_q_ptr  = txt_q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *txt_k_ptr  = txt_k.data_ptr<c10::BFloat16>();

    c10::BFloat16 *rms_scale_q_img_ptr   = rms_scale_q_img.data_ptr<c10::BFloat16>();
    c10::BFloat16 *rms_scale_k_img_ptr   = rms_scale_k_img.data_ptr<c10::BFloat16>();
    c10::BFloat16 *rms_scale_q_txt_ptr   = rms_scale_q_txt.data_ptr<c10::BFloat16>();
    c10::BFloat16 *rms_scale_k_txt_ptr   = rms_scale_k_txt.data_ptr<c10::BFloat16>();

    c10::BFloat16 *o_ptr_q            = out_q.data_ptr<c10::BFloat16>();
    c10::BFloat16 *o_ptr_k            = out_k.data_ptr<c10::BFloat16>();

    const bf16* d_img_q = reinterpret_cast<const bf16*>(img_q_ptr);
    const bf16* d_img_k = reinterpret_cast<const bf16*>(img_k_ptr);
    const bf16* d_txt_q = reinterpret_cast<const bf16*>(txt_q_ptr);
    const bf16* d_txt_k = reinterpret_cast<const bf16*>(txt_k_ptr);

    const bf16* d_rms_scale_q_img  = reinterpret_cast<const bf16*>(rms_scale_q_img_ptr);
    const bf16* d_rms_scale_k_img  = reinterpret_cast<const bf16*>(rms_scale_k_img_ptr);
    const bf16* d_rms_scale_q_txt  = reinterpret_cast<const bf16*>(rms_scale_q_txt_ptr);
    const bf16* d_rms_scale_k_txt  = reinterpret_cast<const bf16*>(rms_scale_k_txt_ptr);

          bf16* d_o_q            = reinterpret_cast<bf16*>(o_ptr_q);
          bf16* d_o_k            = reinterpret_cast<bf16*>(o_ptr_k);

    // launch variables
    unsigned long mem_size = 30000;

    cudaFuncSetAttribute(
        flux_rmsnorm,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    flux_rmsnorm<<<batch*heads,NUM_THREADS_NORM,mem_size>>>(
        d_img_q, d_img_k, d_txt_q, d_txt_k, 
        d_rms_scale_q_img, d_rms_scale_k_img, d_rms_scale_q_txt, d_rms_scale_k_txt, 
        d_o_q, d_o_k
    );  
    CHECK_CUDA_ERROR(cudaGetLastError());
}
#else
#include "harness.impl"
#endif
