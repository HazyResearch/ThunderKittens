// #define TORCH_COMPILE 

#include "kittens.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <cuda/barrier>

#define NUM_WORKERS 2
// #define NUM_WARPS   (NUM_WORKERS*4)
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)
     
const int TXT_D = 512;
const int IMG_D = 3072;  
const int HEAD_D = 128;
const int N_CHUNK = 16;

const int d_head_tile = HEAD_D / kittens::TILE_DIM;
const int seq_tiles = N_CHUNK / kittens::TILE_DIM;

using namespace kittens;

// shared memory
#define tile_1xHEAD_D st<bf16, seq_tiles, d_head_tile>
#define vec_smem_1xHEAD_D sv_bf<d_head_tile>
#define vec_smem_1xSEQ_TILE sv_bf<seq_tiles>

// register
#define reg_tile_1xHEAD_D rt<bf16, seq_tiles, d_head_tile>
#define col_reg_vec_1xSEQ_TILE col_vec<reg_tile_1xHEAD_D>

// rms norm 
__global__ __launch_bounds__(NUM_THREADS, 1)
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
    const bf16 *img_k_g          = reinterpret_cast<const bf16*>(__img_k)+blockIdx.x*(IMG_D*HEAD_D);
    const bf16 *scale_g_img_q = reinterpret_cast<const bf16*>(__rms_norm_scale_img_q);
    const bf16 *scale_g_img_k = reinterpret_cast<const bf16*>(__rms_norm_scale_img_k);

    const bf16 *txt_q_g          = reinterpret_cast<const bf16*>(__txt_q)+blockIdx.x*(TXT_D*HEAD_D);
    const bf16 *txt_k_g          = reinterpret_cast<const bf16*>(__txt_k)+blockIdx.x*(TXT_D*HEAD_D);
    const bf16 *scale_g_txt_q = reinterpret_cast<const bf16*>(__rms_norm_scale_txt_q);
    const bf16 *scale_g_txt_k = reinterpret_cast<const bf16*>(__rms_norm_scale_txt_k);
          bf16 *o_q_g            = reinterpret_cast<bf16*>(__o_q)+blockIdx.x*((IMG_D+TXT_D)*HEAD_D);
          bf16 *o_k_g            = reinterpret_cast<bf16*>(__o_k)+blockIdx.x*((IMG_D+TXT_D)*HEAD_D);

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    tile_1xHEAD_D (&img_q_s)   [2][NUM_WORKERS] = al.allocate<tile_1xHEAD_D,2,NUM_WORKERS>();
    tile_1xHEAD_D (&img_k_s)   [2][NUM_WORKERS] = al.allocate<tile_1xHEAD_D,2,NUM_WORKERS>();
    tile_1xHEAD_D (&txt_q_s)   [2][NUM_WORKERS] = al.allocate<tile_1xHEAD_D,2,NUM_WORKERS>();
    tile_1xHEAD_D (&txt_k_s)   [2][NUM_WORKERS] = al.allocate<tile_1xHEAD_D,2,NUM_WORKERS>();

    // all workers bring rms norm scales to register
    col_reg_vec_1xSEQ_TILE scale_reg_img_q;
    load(scale_reg_img_q, scale_g_img_q); 
    col_reg_vec_1xSEQ_TILE scale_reg_img_k;
    load(scale_reg_img_k, scale_g_img_k); 

    col_reg_vec_1xSEQ_TILE scale_reg_txt_q;
    load(scale_reg_txt_q, scale_g_txt_q); 
    col_reg_vec_1xSEQ_TILE scale_reg_txt_k;
    load(scale_reg_txt_k, scale_g_txt_k); 

    // pipelining
    int tic = 0, toc = 1;
    const int total_elements = N_CHUNK * HEAD_D;
    auto block = cooperative_groups::this_thread_block();
     __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> barrier_cheat;
    if (threadIdx.x == 0) {init(&barrier_cheat, block.size());}
    block.sync(); // Need to make sure none calls before setup.

    // initial load
    load_async(img_q_s[warpid][tic], img_q_g + warpid*total_elements, HEAD_D, barrier_cheat);
    load_async(img_k_s[warpid][tic], img_k_g + warpid*total_elements, HEAD_D, barrier_cheat);
    load_async(txt_q_s[warpid][tic], txt_q_g + warpid*total_elements, HEAD_D, barrier_cheat);
    load_async(txt_k_s[warpid][tic], txt_k_g + warpid*total_elements, HEAD_D, barrier_cheat);
    __syncthreads();
    
    int n_blocks = IMG_D / (NUM_WORKERS*kittens::TILE_DIM);
    int n_blocks_txt = TXT_D / (NUM_WORKERS*kittens::TILE_DIM);
    for (int block = 0; block < n_blocks; block ++, tic ^=1, toc ^=1) {
        barrier_cheat.arrive_and_wait();  

        // inits for the strips
        reg_tile_1xHEAD_D data;
        reg_tile_1xHEAD_D scratch;
        col_reg_vec_1xSEQ_TILE mean;
        col_reg_vec_1xSEQ_TILE rrms;
        zero(mean);
        zero(rrms);
        zero(data);
        zero(scratch);

        // kick off load for the next block
        auto next_idx = (block + 1)*NUM_WORKERS + warpid; 
        if( block < n_blocks - 1 ) {
            load_async(img_q_s[warpid][toc], img_q_g + next_idx*total_elements, HEAD_D, barrier_cheat);
            load_async(img_k_s[warpid][toc], img_k_g + next_idx*total_elements, HEAD_D, barrier_cheat);
        }
        if (block < n_blocks_txt - 1) {
            load_async(txt_q_s[warpid][toc], txt_q_g + next_idx*total_elements, HEAD_D, barrier_cheat);
            load_async(txt_k_s[warpid][toc], txt_k_g + next_idx*total_elements, HEAD_D, barrier_cheat);
        }

        // img q
        load(data, img_q_s[warpid][tic]);
        // rrms = torch.rsqrt(torch.mean(img_q**2, dim=-1, keepdim=True) + 1e-6)
        copy(scratch, data);
        mul(scratch, scratch, scratch); // img_q**2
        row_sum(mean, scratch);
        div(mean, mean, __float2bfloat16(HEAD_D));
        add(mean, mean, __float2bfloat16(1e-06f));
        rsqrt(rrms, mean);
        // rmsnorm
        mul_row(data, data, rrms);
        mul_row(data, data, scale_reg_img_q);
        store(img_q_s[warpid][tic], data);

        /// img k
        zero(mean);
        zero(rrms);
        zero(data);
        load(data, img_k_s[warpid][tic]);
        // rrms = torch.rsqrt(torch.mean(img_q**2, dim=-1, keepdim=True) + 1e-6)
        copy(scratch, data);
        mul(scratch, scratch, scratch); // img_q**2
        row_sum(mean, scratch);
        div(mean, mean, __float2bfloat16(HEAD_D));
        add(mean, mean, __float2bfloat16(1e-06f));
        rsqrt(rrms, mean);
        // rmsnorm
        mul_row(data, data, rrms);
        mul_row(data, data, scale_reg_img_k);
        store(img_k_s[warpid][tic], data);
        

        if ( block < n_blocks_txt ) {
            // txt q
            zero(mean);
            zero(rrms);
            zero(data);
            load(data, txt_q_s[warpid][tic]);
            // rrms = torch.rsqrt(torch.mean(img_q**2, dim=-1, keepdim=True) + 1e-6)
            copy(scratch, data);
            mul(scratch, scratch, scratch); // img_q**2
            row_sum(mean, scratch);
            div(mean, mean, __float2bfloat16(HEAD_D));
            add(mean, mean, __float2bfloat16(1e-06f));
            rsqrt(rrms, mean);
            // rmsnorm
            mul_row(data, data, rrms);
            mul_row(data, data, scale_reg_img_q);
            store(txt_q_s[warpid][tic], data);

            // txt k
            zero(mean);
            zero(rrms);
            zero(data);
            load(data, txt_k_s[warpid][tic]);
            // rrms = torch.rsqrt(torch.mean(img_q**2, dim=-1, keepdim=True) + 1e-6)
            copy(scratch, data);
            mul(scratch, scratch, scratch); // img_q**2
            row_sum(mean, scratch);
            div(mean, mean, __float2bfloat16(HEAD_D));
            add(mean, mean, __float2bfloat16(1e-06f));
            rsqrt(rrms, mean);
            // rmsnorm
            mul_row(data, data, rrms);
            mul_row(data, data, scale_reg_img_q);
            store(txt_k_s[warpid][tic], data);

            store(o_q_g + (block*NUM_WORKERS +warpid)*total_elements, txt_q_s[warpid][tic], HEAD_D); 
            store(o_k_g + (block*NUM_WORKERS +warpid)*total_elements, txt_k_s[warpid][tic], HEAD_D); 
        }

        store_async(o_q_g + txt_offset + (block*NUM_WORKERS +warpid)*total_elements, img_q_s[warpid][tic], HEAD_D, barrier_cheat);
        store_async(o_k_g + txt_offset + (block*NUM_WORKERS +warpid)*total_elements, img_k_s[warpid][tic], HEAD_D, barrier_cheat);
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
    unsigned long mem_size = 72000;

    cudaFuncSetAttribute(
        flux_rmsnorm,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mem_size
    );

    flux_rmsnorm<<<batch*heads,NUM_THREADS,mem_size>>>(
        d_img_q, d_img_k, d_txt_q, d_txt_k, 
        d_rms_scale_q_img, d_rms_scale_k_img, d_rms_scale_q_txt, d_rms_scale_k_txt, 
        d_o_q, d_o_k
    );  
    CHECK_CUDA_ERROR(cudaGetLastError());
}
#else
#include "harness.impl"
#endif
