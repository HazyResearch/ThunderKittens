#include <iostream>
#include <math.h>
#include <assert.h>
#include <mma.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
using namespace nvcuda;

# include "src/kittens.cuh"
# include "src/common/pyutils/torch_helpers.cuh"

using namespace kittens;

// Types
typedef rt_bf<1, 1> _rtd_qk;
typedef rt_bf<1, 4> _rtd_v;
typedef rt_fl<1, 1> _rtd_qk_accum;
typedef rt_fl<1, 4> _rtd_v_accum;
typedef rt_bf<1, 1, ducks::rt_layout::col> _rtd_qk_col;
typedef rt_bf<1, 4, ducks::rt_layout::col> _rtd_v_col;

// Compute A0.
// We are computing V.cumsum(dim=0) in this example (Across the sequence)
// We first compute the local cumulative sum.
// Each has their local copy of V, we have to add in two elements
// 1. the preceding a0 from the last iteration (Stored in total_a0)
// 2. We need to compute a cumulative sum across these tiles.
// To handle 1, we add in total_a0 to a0
__device__
void tb_cumsum(
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&dst)[kittens::N_WARPS], 
    st_bf_1x4<ducks::st_layout::xor_swizzle>::row_vec &total, // tile should have same width as the tile and same type.
    const st_bf_1x4<ducks::st_layout::xor_swizzle> (&src)[kittens::N_WARPS],  
    const int nThreads = 256
) {
    using T = st_bf_1x4<ducks::st_layout::xor_swizzle>;
    using H = T::dtype;

    const int width = T::width;
    const int height = T::height;
    const int rows = T::rows;
    const int row_stride = T::cols;
    auto col = threadIdx.x % (kittens::TILE_DIM*width);
    
    // Threads are assigned to cols, and then go sequentially through the all rows in the warps
    __syncthreads();
    for(auto col = threadIdx.x; col < kittens::TILE_DIM*width; col+= nThreads) {
        // this is resonsible for this column value.
        H v = total.data[col];
        for(auto w = 0; w < kittens::N_WARPS; w++) {
            H *_dst = dst[w].data;
            const H *_src = src[w].data;
            auto idx = col;
            for(auto r = 0; r < rows; r++, idx += row_stride) {
                v += _src[idx];
                _dst[idx] = v;
            }
        }
        total.data[col] = v;
    } 
}

// // We write the local copy, and we want to compute a cumulative sum:
// // 1. we need to add in the A0 that we computed in the last loop (handled by warp adding to its copy)
// // 2. we need the A1 fragments computed from the preceding warp.
// // Then a1 has the "preceding" a1 for each warp; total_a1 is the next stage of what we need to build.
__device__
void tb_cumsum_delay_tiles_inplace(
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&x)[kittens::N_WARPS], 
    st_bf_1x4<ducks::st_layout::xor_swizzle> &total, 
    const int nThreads = 256
) {
    using T = st_bf_1x4<ducks::st_layout::xor_swizzle>;
    using TT = T::dtype;

    auto col = threadIdx.x % (kittens::TILE_DIM*T::width);
    auto row = threadIdx.x / (kittens::TILE_DIM*T::width); 
    __syncthreads(); 
    const int _row_stride = T::cols; // SA TODO: double check
    auto idx = row*_row_stride+col;
    assert(nThreads % (kittens::TILE_DIM * T::width) == 0);
    auto rows_per_block = nThreads / (kittens::TILE_DIM*T::width);
    auto row_skip       = rows_per_block * _row_stride;

    for(auto h = 0; h < T::height; h++) {
        for(auto rows = 0; rows < rows_per_block; rows ++, idx += row_skip) {
            const int _idx = h*_row_stride*kittens::TILE_DIM + 0*kittens::TILE_DIM;
            TT *src0 = x[0].data + _idx; // TODO: SA confirm this rewrite // typename T::T t = x[0].tile_start(h,0)[idx];
            TT v = src0[idx];
        
            // The "delay" happens here. We store the history in the first warp, total.
            // Then the cumulative sum is delayed by 1 warp (which corresponds to the history)
            TT *src1 = total.data + _idx; // Getting total.tile_start(h,0)[idx];
            TT v1 = src1[idx];
            TT *dst0 = x[0].data + _idx;  // Getting x[0].raw_tile_start(h,0)[idx];
            dst0[idx] = v1;

            v += v1;

            for(int wrp = 1; wrp < kittens::N_WARPS; wrp++) {
                TT *src1 = x[wrp].data + _idx;
                TT _v1 = src1[idx];       // Getting x[wrp].tile_start(h,0)[idx];

                TT *dst1 = x[wrp].data + _idx;
                dst1[idx] = v;            // Setting x[wrp].raw_tile_start(h,0)[idx] = v; 
                v += _v1;
            }                             // Updating t += t1;

            TT *out = total.data + _idx;  // Getting total.tile_start(h,0)[idx];
            out[idx] = v;                 // Store the full count in Y
        } 
    }
}

__device__
void reduce_tile_tiles(
    st_bf_1x4<ducks::st_layout::xor_swizzle> &dst, 
    const st_bf_1x4<ducks::st_layout::xor_swizzle> (&src)[kittens::N_WARPS], 
    const int nThreads = 256
) {
    using T = st_bf_1x4<ducks::st_layout::xor_swizzle>;
    using TT = T::dtype;
    auto col = threadIdx.x % (kittens::TILE_DIM*T::width);
    auto row = threadIdx.x / (kittens::TILE_DIM*T::width); 
    __syncthreads(); 
    const int _row_stride = T::cols; // SA TODO: double check
    auto idx = row*_row_stride+col;
    assert(nThreads % (kittens::TILE_DIM * T::width) == 0);
    auto rows_per_block = nThreads / (kittens::TILE_DIM*T::width);
    auto row_skip       = rows_per_block * _row_stride;
    for(auto h = 0; h < T::height; h++) {
        for(auto rows = 0; rows < rows_per_block; rows ++, idx += row_skip) {
            int _idx = h*_row_stride*kittens::TILE_DIM + 0*kittens::TILE_DIM;

            T t = src[0];   // TODO: SA confirm this rewrite
            TT *src0 = t.data + _idx;
            TT v = src0[idx];
            for(int wrp = 1; wrp < kittens::N_WARPS; wrp++) {
                T t1 = src[wrp];
                TT *src1 = t1.data + _idx;
                v += src1[idx];
            }
            TT *dst0 = dst.data + _idx;
            dst0[idx] += v;
        } 
   }
}


__device__
static void
make_causal(_rtd_qk_accum &accum) {
    using T = _rtd_qk_accum::dtype;
    using T2 = rt_base<T, _rtd_qk_accum::layout>;
    
    // Structure of rt_tiles
    // tiles = [ [0, 1], [2, 3] ]
    // accum_top_row    = src.tiles[i][0].data[0], src.tiles[i][0].data[2];
    // accum_bottom_row = src.tiles[i][0].data[1], src.tiles[i][0].data[3];
                
    const int tile_height = _rtd_qk_accum::height;
    const int tile_width  = _rtd_qk_accum::width;
    auto lane  = kittens::laneid();
    auto row      = (lane / 4);
    auto next_row = row + 8;
    auto col      = 2*(lane % 4);
    float2 _zero{0.,0.};
    __syncwarp();
     for(auto h = 0; h < tile_height; h++){
        for(auto i = 0; i < 2; i++) {
            if(row       < col +     8*i) {
                // Activates on data[0][0].x and data[1][0].x --> Corresponds to: data[0], data[1]
                T2 t = accum.tiles[h][h];
                T _t = t.data[i];
                _t.x = 0.;
            }
            if (row       < col + 1 + 8*i) {
                // Activates on data[0][0].y and data[1][0].y --> Corresponds to: data[0], data[1]
                T2 t = accum.tiles[h][h];
                T _t = t.data[i];
                _t.y = 0.;
            }
            if (next_row  < col +     8*i) {
                // Activates on data[0][1].x and data[1][1].x --> Corresponds to: data[2], data[3]
                T2 t = accum.tiles[h][h];
                T _t = t.data[i+2];
                _t.x = 0.;
            }
            if (next_row  < col + 1 + 8*i) {
                // Activates on data[0][1].y and data[1][1].y --> Corresponds to: data[2], data[3]
                T2 t = accum.tiles[h][h];
                T _t = t.data[i+2];
                _t.y = 0.;
            }

            // Old version: SA confirm above replacement
            // if(row       < col +     8*i) {accum[h][h].data[i][0].x = 0.;} // i; 0-1 / 0-1 
            // if(row       < col + 1 + 8*i) {accum[h][h].data[i][0].y = 0.;}
            // if(next_row  < col +     8*i) {accum[h][h].data[i][1].x = 0.;}
            // if(next_row  < col + 1 + 8*i) {accum[h][h].data[i][1].y = 0.;} // occurs on 
        }
        for(auto w = h+1; w < tile_width; w++) {
            for(auto i = 0; i < 2; i++) {
                accum.tiles[h][w].data[i] = _zero;
            }
        }
    }
}

// Note chris left a comment: this is a wasteful way to do this.
__device__
static void mul_row_slice(_rtd_qk &reg, const int index) {
    using T = _rtd_qk::dtype;
    using T2 = rt_base<T, _rtd_qk::layout>;

    auto lane       = kittens::laneid();
    auto row        = lane / 4;
    auto col        = lane % 4; // * 2
    
    __syncwarp();
    for(auto col_offset = 0; col_offset < 2; col_offset++) {
        T2 v = (index < 8) ? reg.tiles[col_offset][0] : reg.tiles[col_offset][1]; // SA: likely need to update indexing
        T2 vs[4];
        #pragma unroll
        for(auto j=0; j < 4; j++) {
        //     vs[j] = __shfl_sync(0xFFFFFFFF, v, index*4 + j);
        }
        auto my_v = vs[col];
        __syncwarp();
        #pragma unroll
        for(auto i=0; i < 2; i++) {
            // reg.tiles[col_offset][i] = base_ops::mul(reg.tiles[col_offset][i],my_v);
        }
    }
}

template <typename H, typename T, bool _debug_build>
__global__
void a012_compute_ker(int n, int d, int dv, const T* __q, const T* __k, 
                                 const T* __v, T* __y, T* __a0, T* __a1, T* __a1y) { 

    auto warpid = kittens::warpid();
    auto lane   = kittens::laneid();
    constexpr int NUM_WORKERS = kittens::N_WARPS;

    const H *_q   = reinterpret_cast<const H*>(__q)+blockIdx.x*(n*d);
    const H *_k   = reinterpret_cast<const H*>(__k)+blockIdx.x*(n*d);
    const H *_v   = reinterpret_cast<const H*>(__v)+blockIdx.x*(n*dv);
          H *_y   = reinterpret_cast<H*>(__y)+blockIdx.x*(n*dv);
    
    // Debugging Data structures
    H *_a0  = _debug_build ? reinterpret_cast<H*>(__a0)+blockIdx.x*(n*dv) : NULL;
    H *_a1  = _debug_build ? reinterpret_cast<H*>(__a1)+blockIdx.x*(n*dv) : NULL;
    H *_a1y = _debug_build ? reinterpret_cast<H*>(__a1y)+blockIdx.x*(n*dv) : NULL;
    
    // this is the CUDA shared memory
    extern __shared__ alignment_dummy __shm[]; // this is the CUDA shared memory
    shared_allocator al = shared_allocator::create_allocator((int*)&__shm[0]);
    st_bf_1x1<ducks::st_layout::xor_swizzle> (&q)[2][NUM_WORKERS] = al.allocate<st_bf_1x1<ducks::st_layout::xor_swizzle>, 2, NUM_WORKERS>();
    st_bf_1x1<ducks::st_layout::xor_swizzle> (&k)[2][NUM_WORKERS] = al.allocate<st_bf_1x1<ducks::st_layout::xor_swizzle>, 2, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&v)[2][NUM_WORKERS] = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, 2, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&y)[NUM_WORKERS]    = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&ty)[NUM_WORKERS]   = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&a0)[NUM_WORKERS]   = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, NUM_WORKERS>();
    st_bf_1x4<ducks::st_layout::xor_swizzle> (&a1)[NUM_WORKERS]   = al.allocate<st_bf_1x4<ducks::st_layout::xor_swizzle>, NUM_WORKERS>();

    // A0, A1, A2 (a2 is stored in register throughout)
    __shared__ st_bf_1x4<ducks::st_layout::xor_swizzle>::row_vec total_a0;
    __shared__ st_bf_1x4<ducks::st_layout::xor_swizzle> total_a1;

    // Registers per thread for fragments
    _rtd_qk qj0, qj1, kj0, kj1;
    _rtd_qk qfrag, qkfrag;
    _rtd_qk_col kfrag; 
    _rtd_v_col A2j0, A2j1, vfrag;
    _rtd_v_accum A2j0_accum, A2j1_accum, o_accum, qA2_accum;
    
    // Registers for a1
    _rtd_qk qk_a1_f;
    _rtd_qk_accum qk_a1, temp_accum; 
    _rtd_v_accum a1_accum, a1_out;
    _rtd_v a1_frag;
    _rtd_v_col a1_col_frag; 
     
    // Pipeline handlers and barriers
    int tic = 0, toc = 1;
    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> qkv_barrier;
    if (threadIdx.x == 0) {init(&qkv_barrier, block.size());}
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> store_barrier;
    if (threadIdx.x == 0) {init(&store_barrier, block.size());}
    // Make sure no gets to the barrier before its initialized.
    block.sync(); 

    // Contstants
    const int qk_tile_elements = _rtd_qk::num_elements;
    const int  v_tile_elements = _rtd_v::num_elements; 
    auto n_tiles  = n/kittens::TILE_DIM;
    auto n_blocks = n_tiles/NUM_WORKERS;
    assert(n_tiles % NUM_WORKERS == 0);

    // Load in initial batches of QKV along the sequence dimension
    kittens::load_async(q[tic][warpid], _q + warpid*qk_tile_elements, d,  qkv_barrier);
    kittens::load_async(k[tic][warpid], _k + warpid*qk_tile_elements, d,  qkv_barrier);
    kittens::load_async(v[tic][warpid], _v + warpid*v_tile_elements , dv, qkv_barrier);
                
    // Set the tiles and accumulators to 0.
    zero(A2j0);
    zero(A2j1);
    zero(A2j0_accum);
    zero(A2j1_accum);
    zero(a1_accum);
    if(warpid == 0) {
        // zero(total_a1); 
        zero(total_a0);
    }
    __syncthreads();

    for(auto cur_block = 0; cur_block < n_blocks; cur_block++, tic ^= 1, toc ^= 1) {
        qkv_barrier.arrive_and_wait(); 
        if(cur_block < n_blocks - 1) { // Kick off the next block load.
            auto next_idx = (cur_block + 1)*NUM_WORKERS + warpid; 
            kittens::load_async(q[toc][warpid], _q + next_idx * qk_tile_elements, d, qkv_barrier);
            kittens::load_async(k[toc][warpid], _k + next_idx * qk_tile_elements, d, qkv_barrier);
            kittens::load_async(v[toc][warpid], _v + next_idx * v_tile_elements, dv, qkv_barrier);
        } 
        
        // We first handle the causal portion, the diagonal elements.
        // 1. We multiply (QK.T) on the diagonal tiles.
        // 2. Entry-wise square this
        // 3. Multiply by V.
        // Do the multiplication (qk)^2@V and store the result in y[warpid]
        load(qfrag, q[tic][warpid]);
        // Note we want an outer product here of Q and K, so we load K transposed from ocl.
        load(kfrag, k[tic][warpid]);
        transpose_inplace(kfrag); 
        
        // zero(temp_accum);
        // mma(temp_accum, qfrag, kfrag, temp_accum);
        // make_causal(temp_accum);
        // copy(qk_a1, temp_accum); 
        // mul(temp_accum, temp_accum, temp_accum); // square it, since this is the A2 term.
        
        // load(vfrag, v[tic][warpid]);
        // zero(o_accum);

        // // Compute the a0 portion: V.cumsum(dim=0) in this example (Across the sequence)
        // tb_cumsum(a0, total_a0, v[tic]);
        // __syncthreads();
        
        // // Compute the a1 output portion: Qc@A1 + make_causal(Qc@Ktc)@Vc
        // zero(a1_out);
        // copy(qk_a1_f, qk_a1);
        // mma(o_accum, qk_a1_f, vfrag, o_accum);
        
        // // This is updating our local slice.
        // // This is the update for A1 "after" our slice, i.e., containing everything it has seen.
        // // A1 += Kt[:,whole_slice]@V[whole_slice]
        // zero(a1_accum);
        // _rtd_qk rkfrag;
        // swap_layout(rkfrag, kfrag);
        // mma(a1_accum, rkfrag, vfrag, a1_accum);

        // // We write the local copy, and we want to compute a cumulative sum:
        // // 1. we need to add in the A0 that we computed in the last loop
        // // 2. we need the A1 fragments computed from the preceding war.
        // // To handle both, we do a cumulative sum. 1 is handled by warp  adding
        // // to its copy and letting the cumulative sum take care of it.
        // store(a1[warpid], a1_accum);

        // // Now, a1 has the "preceding" a1 for each warp; Total_a1 is the next stage of what we need to build.
        // tb_cumsum_delay_tiles_inplace(a1, total_a1); 
        // __syncthreads(); // need the writes to a1 to finish.

        // // Now, each warp loads a1[warpid] into a1_col_frag for the multiplication 
        // // This captures all the history add it to accum, Then accum contains the whole part of a1y
        // load(a1_col_frag, a1[warpid]);
        // mma(o_accum, qfrag, a1_col_frag, o_accum);
        
        // // From above o_accum holds + causal(QK)@V + Q@A1.
        // // Computes += causal(QK)**2@V/2 
        // mul(temp_accum, temp_accum, 0.5f);
        // copy(qkfrag, temp_accum);
        // mma(o_accum, qkfrag, vfrag, o_accum);

        // Copy in the the a0 portion; the a1 and a2 portions are in o_accum
        copy(y[warpid], a0[warpid]); 
        store(y[warpid], o_accum);

        // ** This is the A2 non-diag case and handles the update **
        // This is the in-shared-mem portion We keep A2 in register spread across the warps. 
        // Each warp has a 2 fragments of q and k and 1 fragment of v in memory.
        // The indexing is below, but these are the outer products. 
        // At this point, y[0].. y[read_block-1] contains the "diagonal" blocks of all the outputs.
        // * We keep A2[2*warp], A2[2*warp+1] in register.
        // * Each computes their local portion of Q[j,:]*Q*A2 and Stores it back in ty[warpid]
        // This is hard-coded to A2 having dimension 16.
        // __syncthreads();
        // for(auto blk = 0; blk < NUM_WORKERS; blk++) { 
            
        //     // This computes the "history": Q[j]@A2[j] for j=0,dots,15.
        //     load(qj0, q[tic][warpid]);
        //     copy(qj1, qj0); // faster than reloading?

        //     // We store Q_j <- Q[:,j]*Q
        //     mul_col_slice(qj0[0][0], 2*warpid);
        //     mul_col_slice(qj1[0][0], 2*warpid+1);

        //     // Compute qj, a2j portion
        //     zero(qA2_accum);
        //     mma(qA2_accum, qj0, A2j0, qA2_accum); // false means clear registers
        //     mma(qA2_accum, qj1, A2j1, qA2_accum); // false means clear registers
        //     mul(qA2_accum,  qA2_accum, 0.5f);
        //     store(ty[warpid], qA2_accum);
            
        //     // reduce_tile_tiles(y[blk], ty);   # SA: WARNING -- TAKING SO LONG TO COMPILE
        //     __syncthreads();

        //     // Update state for next round only needed if there is more work.
        //     load(kj0, k[tic][blk]);
        //     transpose_inplace(kj0); 
        //     copy(kj1, kj0); 
        //     mul_row_slice(kj0[0][0], 2*warpid); 
        //     mul_row_slice(kj1[0][0], 2*warpid+1);

        //     // Compute the A2[j] update and put it back in the register
        //     load(vfrag, v[tic][blk]);
        //     mma(A2j0_accum, kj0, vfrag, A2j0_accum);

        //     _rtd_v copy_bf_A2j0;
        //     copy(copy_bf_A2j0, A2j0_accum);
        //     swap_layout(A2j0, copy_bf_A2j0);            

        //     mma(A2j1_accum, kj1, vfrag, A2j1_accum); 
        //     _rtd_v copy_bf_A2j1;
        //     copy(copy_bf_A2j1, A2j1_accum);
        //     swap_layout(A2j1, copy_bf_A2j1); 
        // }
        __syncthreads();
        store(_y + (cur_block * NUM_WORKERS + warpid)*v_tile_elements, y[warpid], dv);
    }
}

void
a012_compute(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(o);
    
    auto batch = q.size(0);
    auto head  = q.size(1);
    auto n     = q.size(2);
    auto d     = q.size(3);
    auto dv    = v.size(3);
    bool k_same = true, o_same = true;
    for(auto i = 0; i < 4; i++) { 
        k_same &= q.size(i) == k.size(i);
        o_same &= v.size(i) == o.size(i);
    }
    // This is just a restriction of what we're doing now...
    TORCH_CHECK(k_same, "Q and K should be same size");
    TORCH_CHECK(o_same, "V and O should be same size");

    TORCH_CHECK(q.scalar_type() == c10::ScalarType::BFloat16, "Q is a Bfloat");
    TORCH_CHECK(k.scalar_type() == c10::ScalarType::BFloat16, "K is a Bfloat");
    TORCH_CHECK(v.scalar_type() == c10::ScalarType::BFloat16, "V is a Bfloat");
    TORCH_CHECK(o.scalar_type() == c10::ScalarType::BFloat16, "O is a Bfloat");

    using H = __nv_bfloat16;
    using T = c10::BFloat16;
    constexpr bool _debug_build = false;
    const int NUM_WORKERS = 8;

    // q,k,v, and o are all double buffered
    unsigned long mem_size  =  2*2*NUM_WORKERS*sizeof(st_bf_1x1<ducks::st_layout::xor_swizzle>); // q, k and v are double buffered.
                  mem_size +=    2*NUM_WORKERS*sizeof(st_bf_1x4<ducks::st_layout::xor_swizzle>);
                  mem_size += (NUM_WORKERS+NUM_WORKERS)*sizeof(st_bf_1x4<ducks::st_layout::xor_swizzle>);
                  mem_size += 2*NUM_WORKERS*sizeof(st_bf_1x4<ducks::st_layout::xor_swizzle>); // a0 and a1y

    TORCH_CHECK(n % (NUM_WORKERS*kittens::TILE_DIM) == 0, "The number of elements should be divisible the number of NUM_WORKERS times stored fragments");
    auto threads = NUM_WORKERS * kittens::WARP_SIZE;
    CHECK_CUDA_ERROR(cudaFuncSetAttribute(
             a012_compute_ker<H, T, _debug_build>,
             cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size));
    
    a012_compute_ker<H,T,false><<<batch*head,threads,mem_size>>>(n, d, dv, q.data_ptr<T>(), k.data_ptr<T>(), v.data_ptr<T>(),
          o.data_ptr<T>(), NULL, NULL, NULL);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}
