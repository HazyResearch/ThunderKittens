#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kittens.dp.hpp"
#include "prototype.dp.hpp"
#include <sycl/ext/intel/math.hpp>

using namespace kittens;

// ----- HELPER FUNCTIONS NEEDED BY BASED LINEAR PREFILL -----

// cumulative sum of v onto a0_total
template<kittens::ducks::st::all ST>
void accumulate_a0(sv_bf<ST::cols> &a0_total, const ST &v) {
    int col =
        sycl::ext::oneapi::this_work_item::get_nd_item<3>().get_local_id(2) * 2;
    if(col < ST::cols) {
        sycl::float2 acc = sycl::float2(sycl::ext::intel::math::bfloat162float(
                                            (*(bf16_2 *)&a0_total[col]).x()),
                                        sycl::ext::intel::math::bfloat162float(
                                            (*(bf16_2 *)&a0_total[col]).y()));
#pragma unroll
        for(int row = 0; row < ST::rows; row++) {
            sycl::float2 v_data =
                sycl::float2(sycl::ext::intel::math::bfloat162float(
                                 (*(bf16_2 *)&v[sycl::int2{row, col}]).x()),
                             sycl::ext::intel::math::bfloat162float(
                                 (*(bf16_2 *)&v[sycl::int2{row, col}]).y()));
            acc.x() += v_data.x();
            acc.y() += v_data.y();
        }
        *(bf16_2 *)&a0_total[col] =
            sycl::vec<sycl::ext::oneapi::bfloat16, 2>(acc.x(), acc.y());
    }
}

// in pytorch, this computes, for a 16x64 tensor dst and 16x16 tensor src:
// dst = torch.cat([src * src[:,starting_col+i].unsqueeze(0) for i in range(4)], dim=-1)
static void mul_slice_row(rt_bf<16,64> &dst, const rt_bf<16,16> &src, const int starting_col) {

    const int lane = kittens::laneid(); // 0...31    
    // each thread is responsible for two rows
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        copy(reinterpret_cast<rt_bf<16,16>&>(dst.tiles[0][i]), src);
        const int target_col = starting_col + i;
        #pragma unroll
        for(int row_offset = 0; row_offset < 2; row_offset++) {
            const int src_thread = (lane / 4)*4 + (target_col%8)/2;
            const int col_offset = target_col >= 8;
            bf16_2 src_val = dst.tiles[0][i].data[2*col_offset + row_offset];
            /*
            DPCT1108:414: '__shfl_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            bf16 val = dpct::experimental::select_from_sub_group(
                kittens::MASK_ALL,
                sycl::ext::oneapi::this_work_item::get_sub_group(),
                (target_col % 2 == 0) ? src_val.x() : src_val.y(),
                src_thread); // correct value obtained and passed around

            dst.tiles[0][i].data[row_offset] *= bf16_2{val, val};
            dst.tiles[0][i].data[row_offset+2] *= bf16_2{val, val};
        }
    }
}


// in pytorch, this computes, for a 16x64 tensor dst and 16x16 tensor src:
// dst = torch.cat([src * src[:,starting_col].unsqueeze(-1) for _ in range(4)], dim=-1)
static void mul_slice_col(rt_bf<16,64> &dst, const rt_bf<16,64> &src, const int target_row) {

    const int lane = kittens::laneid(); // 0...31    
    // each thread is responsible for two cols
    copy(dst, src);
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        #pragma unroll
        for(int col_offset = 0; col_offset < 2; col_offset++) {
            const int src_thread = (target_row%8)*4 + (lane%4);
            const int row_offset = target_row >= 8;
            bf16_2 src_val = dst.tiles[0][i].data[2*col_offset + row_offset];
            /*
            DPCT1108:415: '__shfl_sync' was migrated with the experimental
            feature masked sub_group function which may not be supported by all
            compilers or runtimes. You may need to adjust the code.
            */
            bf16_2 val = dpct::experimental::select_from_sub_group(
                kittens::MASK_ALL,
                sycl::ext::oneapi::this_work_item::get_sub_group(), src_val,
                src_thread); // correct value obtained and passed around

            dst.tiles[0][i].data[col_offset*2+0] *= val;
            dst.tiles[0][i].data[col_offset*2+1] *= val;
        }
    }
}


// ----- BASED LINEAR PREFILL KERNEL -----


using qk_tile = st_bf<64,16>;
using vo_tile = st_bf<64,64>;
using a2_tile = st_bf<64,64>;
using a1_tile = st_bf<64,16>;
using a0_vec  = sv_bf<64>;
using namespace kittens::prototype;
struct based_prefill_layout {
    struct globals { // global layout (here with TMA descriptors)
        gl<bf16, -1, -1, -1, qk_tile::cols, qk_tile> q;
        gl<bf16, -1, -1, -1, qk_tile::cols, qk_tile> k;
        gl<bf16, -1, -1, -1, vo_tile::cols, vo_tile> v;
        gl<bf16, -1, -1, -1, vo_tile::cols, vo_tile> o;
    };
    struct input_block {
        qk_tile q;
        qk_tile k;
        vo_tile v, v2;
    };
    struct output_block { vo_tile o; };
    struct scratch_block {
        a2_tile a2;
        a1_tile a1_trans;
        a0_vec a0;
    };
    struct consumer_state {
        rt_fl<16,16> a1_trans;
        rt_fl<16,64> a2[4];
    };
};
struct based_prefill_template {
    using layout = based_prefill_layout;
    static constexpr int NUM_CONSUMER_WARPS = 4, NUM_BLOCKS = 2, OUTPUT_PIPE_STAGES = 2, INPUT_PIPE_STAGES = 2;
    static int iters(layout::globals &g) { return g.q.rows / qk_tile::rows; }
    struct producer {
        static void setup(producer_setup_args<layout> args) { // setup and load the first iteration
            warpgroup::producer_registers(); // decrease registers for the producer warpgroup
        }
        static void load(producer_load_args<layout> args) { // semaphore for the producer to load into
            auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
            if (warpgroup::warpid() != args.iter % 2) return;
            sycl::int4 index = {(int)item_ct1.get_group(1),
                                (int)item_ct1.get_group(2), args.iter, 0};
            tma::expect(args.inputs_arrived, args.input);
            tma::load_async(args.input.q,  args.globals.q, index, args.inputs_arrived);
            tma::load_async(args.input.k,  args.globals.k, index, args.inputs_arrived);
            tma::load_async(args.input.v,  args.globals.v, index, args.inputs_arrived);
            tma::load_async(args.input.v2, args.globals.v, index, args.inputs_arrived);
            arrive(args.inputs_arrived, 3); // arrive on behalf of other warps
        }
        static void store(producer_store_args<layout> args) {
            auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
            if (warpgroup::warpid() != 2 + args.iter % 2) return;
            sycl::int4 index = {(int)item_ct1.get_group(1),
                                (int)item_ct1.get_group(2), args.iter, 0};
            tma::store_async(args.globals.o, args.output.o, index);
            tma::store_async_read_wait();
            arrive(args.outputs_finished, 4);
        }
    };
    struct consumer {
        static void setup(consumer_setup_args<layout> args) { // setup locals for before the first iteration
            warpgroup::increase_registers<232>();
            warpgroup::zero(args.scratch.a0);
            warpgroup::zero(args.scratch.a1_trans);
            warpgroup::zero(args.scratch.a2);
            zero(args.state.a1_trans);
            for(int i = 0; i < 4; i++) zero(args.state.a2[i]);
            warpgroup::sync();
        }
        /*
        DPCT1110:416: The total declared local variable size in device function
        work exceeds 128 bytes and may cause high register pressure. Consult
        with your hardware vendor to find the total register size available and
        adjust the code, or use smaller sub-group size to avoid high register
        pressure.
        */
        static void work(consumer_work_args<layout> args) {
            int warp = warpgroup::warpid();
            rt_bf<16,64> local_attn_bf; // 4 registers
            rt_fl<16,64> local_attn, temp_attn_accum; // 32 registers
            rt_fl<16,64> o; // 32 registers
            warpgroup::mm_ABt(local_attn, args.input.q, args.input.k);
            warpgroup::mma_async_wait();
            copy(temp_attn_accum, local_attn);
            mul(temp_attn_accum, temp_attn_accum, temp_attn_accum); // square it; note this converts sqrt(d) to d
            mul(temp_attn_accum, temp_attn_accum, 0.5f);            // divide by 2
            add(temp_attn_accum, temp_attn_accum, local_attn);      // add back in 1x for the linear term
            add(temp_attn_accum, temp_attn_accum, 1.f);             // cumulative sum for a0
            copy(local_attn_bf, temp_attn_accum); // now stored.
            // now make causal
            #pragma unroll
            for(int j = 0; j < 4; j++) {
                auto &attn_subtile = reinterpret_cast<rt_bf<16,16>&>(local_attn_bf.tiles[0][j]);
                if (j>warp) zero(attn_subtile);
                else if (j==warp) make_causal(attn_subtile, attn_subtile, kittens::base_types::constants<bf16>::zero());
            }
            warpgroup::mm_AB(o, local_attn_bf, args.input.v);       // reset o here, and do local chunk.
            warpgroup::mma_ABt(o, args.input.q, args.scratch.a1_trans);
            warpgroup::mma_async_wait();
            warpgroup::mma_AtB(args.state.a1_trans, args.input.v2, args.input.k);
            warpgroup::mma_async_wait();
            warpgroup::store(args.scratch.a1_trans, args.state.a1_trans); // store up to shared memory
            warpgroup::sync();
            rt_bf<16,16> q_src; // the source 16x16 tiles -- we'll draw on these for future mul_slice's.
            warpgroup::load(q_src, args.input.q);
            mul(q_src, q_src,
                sycl::ext::intel::math::float2bfloat16(
                    0.70710678118)); // divide by 2 for A2 here.
            rt_bf<64,16> k_src_tmp;
            rt_bf<16,64> k_src;
            load(k_src_tmp, args.input.k);
            transpose_sep(k_src, k_src_tmp); // transpose K into Kt
            // 2nd order taylor, about 75% of execution time is in this loop
            #pragma unroll
            for(int t = 0; t < 4; t++) {
                rt_bf<16,64> q, k;
                mul_slice_row(q, q_src, t*4);
                mul_slice_col(k, k_src, t*4+warp);
                warpgroup::store(args.scratch.a2, args.state.a2[t]); // take previous one and move up to smem for wgmma.
                sycl::group_barrier(
                    sycl::ext::oneapi::this_work_item::get_sub_group());
                warpgroup::mma_AB(o, q, args.scratch.a2); // incorporate a1 onto o
                warpgroup::mma_AB(args.state.a2[t], k, args.input.v); // incorporate KtV onto a2
                warpgroup::mma_async_wait(); // ding dong! o matmuls have now arrived, too.
            }
            // now we do the sum of the previous a0 onto o
            #pragma unroll
            for(int i = 0; i < 4; i++) {
                #pragma unroll
                for(int j = 0; j < 2; j++) {
                    int col = i*16 + j*8 + (kittens::laneid()%4)*2;
                    sycl::float2 data = sycl::float2(
                        sycl::ext::intel::math::bfloat162float(
                            (*(bf16_2 *)&args.scratch.a0[col]).x()),
                        sycl::ext::intel::math::bfloat162float(
                            (*(bf16_2 *)&args.scratch.a0[col]).y()));
                    o.tiles[0][i].data[2 * j].x() += data.x();
                    o.tiles[0][i].data[2 * j].y() += data.y();
                    o.tiles[0][i].data[2 * j + 1].x() += data.x();
                    o.tiles[0][i].data[2 * j + 1].y() += data.y();
                }
            }
            warpgroup::store(args.output.o, o);
            warpgroup::sync();
            arrive(args.outputs_arrived);
            accumulate_a0(args.scratch.a0, args.input.v2);
            warpgroup::sync();
            arrive(args.inputs_finished);
        }
        static void finish(consumer_finish_args<layout> args) {

        }
    };
};

#include "harness.impl"


