#include "kittens.cuh"
#include "prototype.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

// New causal_mask function for LCF backward pass
template <int tile_h_qo, int tile_h>
__device__ static inline void causal_mask(auto& reg_tile, int q_block_idx,
                                          int k_block_idx) {
  // q_block_idx corresponds to args.common.seq (current Q block index in the
  // sequence) k_block_idx corresponds to args.iter (current K/V block index in
  // the sequence)

  // Iterate over rows of the Q tile within the current attention block
  for (int j = 0; j < (tile_h_qo / kittens::TILE_ROW_DIM<bf16>); j++) {
    // Absolute row index for the current Q element being processed
    // This takes into account the block's overall position in the sequence and
    // the row within the current tile
    int q_abs_idx_in_seq =
        q_block_idx * (tile_h_qo / kittens::TILE_ROW_DIM<bf16>)+j;

    auto& attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(reg_tile.tiles[0][j]);

    // Absolute starting index of the current K block in the sequence
    int k_abs_start_idx_in_seq =
        k_block_idx *
        (tile_h /
         kittens::TILE_ROW_DIM<bf16>);  // Assuming K tiles have same height as
                                        // Q tiles for row-wise processing

    // If the current Q row's absolute index is less than the current K block's
    // absolute starting index, it means the Q block is entirely before the K
    // block in sequence. For causal attention, we must mask out such elements.
    if (q_abs_idx_in_seq < k_abs_start_idx_in_seq) {
      neg_infty(attn_subtile);
    } else if (q_abs_idx_in_seq == k_abs_start_idx_in_seq) {
      // If the Q block and K block are at the same sequence position,
      // apply an intra-tile causal mask (upper triangle elements set to
      // negative infinity).
      make_causal_t(attn_subtile, attn_subtile,
                    kittens::base_types::constants<float>::neg_infty());
    }
  }
}

template <int D>
struct bwd_attend_ker_tile_dims {};
template <>
struct bwd_attend_ker_tile_dims<64> {
  constexpr static int tile_width = (64);
  constexpr static int tile_h = (4 * 16);
  constexpr static int tile_h_qo = (4 * 16);
  constexpr static int blocks_sm = 1;
};
template <>
struct bwd_attend_ker_tile_dims<128> {
  constexpr static int tile_width = (128);
  constexpr static int tile_h = (4 * 16);
  constexpr static int tile_h_qo = (4 * 16);
  constexpr static int blocks_sm = 1;
};

template <int DIM, int BWD_CONSUMER_WARPGROUPS>
struct attn_bwd_layout {
  static constexpr int head_dim = DIM;
  using G = bwd_attend_ker_tile_dims<DIM>;
  using qo_tile = st_bf<G::tile_h_qo, G::tile_width>;
  using kv_tile = st_bf<G::tile_h, G::tile_width>;
  using dq_tile = st_fl<G::tile_h_qo, G::tile_width>;
  using dk_tile = st_fl<G::tile_h, G::tile_width>;
  using dv_tile = st_fl<G::tile_h, G::tile_width>;
  using attn_tile = st_bf<G::tile_h_qo, G::tile_h>;
  using logits_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
  using denom_tile = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

  using qo_global = gl<bf16, -1, -1, -1, -1, qo_tile>;
  using kv_global = gl<bf16, -1, -1, -1, -1, kv_tile>;
  using og_global = gl<bf16, -1, -1, -1, -1, qo_tile>;
  using qg_global = gl<float, -1, -1, -1, -1, dq_tile>;
  using kg_global = gl<float, -1, -1, -1, -1, dk_tile>;
  using vg_global = gl<float, -1, -1, -1, -1, dv_tile>;
  using logits_global = gl<float, -1, -1, -1, -1, logits_tile>;
  using denom_global = gl<float, -1, -1, -1, -1, denom_tile>;

  struct globals {
    qo_global Q, dO;
    kv_global K, V;
    qg_global qg;
    kg_global kg;
    vg_global vg;
    logits_global L;
    denom_global D;
  };

  // Producer load block.
  struct input_block {
    qo_tile q;
    qo_tile og;
    logits_tile l;
    denom_tile d;
  };

  struct scratch_block {
    kv_tile k[BWD_CONSUMER_WARPGROUPS];
    kv_tile v[BWD_CONSUMER_WARPGROUPS];
    attn_tile ds_smem[BWD_CONSUMER_WARPGROUPS];
    dk_tile dk_smem[BWD_CONSUMER_WARPGROUPS];
    dv_tile dv_smem[BWD_CONSUMER_WARPGROUPS];
    dq_tile qg_smem;
  };

  struct common_state {
    int batch, head, seq, kv_head;
  };

  struct consumer_state {
    rt_fl<16, G::tile_width> kg_reg;
    rt_fl<16, G::tile_width> vg_reg;
    rt_fl<16, G::tile_width> qg_reg;

    rt_fl<16, 64> lse_block_t;

    // Softmax and attention blocks.
    rt_fl<16, 64> s_block_t, p_block_t;
    // Attention gradient blocks.
    rt_fl<16, 64> ds_block_t, dp_block_t;
    rt_bf<16, 64> ds_block_t_mma, p_block_t_mma;
  };
};

template <int D, bool is_causal>
struct attn_bwd_template {
  // static constexpr int NUM_CONSUMER_WARPS = 12,
  //                      NUM_WORKERS = NUM_CONSUMER_WARPS / 4,
  //                      INPUT_PIPE_STAGES = 1;
  static constexpr int NUM_CONSUMER_WARPS = 8,
                       NUM_CONSUMER_WARPGROUPS = NUM_CONSUMER_WARPS / 4,
                       INPUT_PIPE_STAGES = 1;
  using layout = attn_bwd_layout<D, NUM_CONSUMER_WARPGROUPS>;

  __device__ static inline void stream_tile(auto& reg_tile, auto& smem_vec) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int base_col = 16 * i + 2 * (kittens::laneid() % 4);
      reg_tile.tiles[0][i].data[0] = *(float2*)&smem_vec[base_col + 0];
      reg_tile.tiles[0][i].data[1] = *(float2*)&smem_vec[base_col + 0];
      reg_tile.tiles[0][i].data[2] = *(float2*)&smem_vec[base_col + 8];
      reg_tile.tiles[0][i].data[3] = *(float2*)&smem_vec[base_col + 8];
    }
  }

  __device__ static inline void stream_sub_tile(auto& reg_tile,
                                                auto& smem_vec) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int base_col = 16 * i + 2 * (laneid() % 4);
      reg_tile.tiles[0][i].data[0] = base_ops::sub::template op<float2>(
          reg_tile.tiles[0][i].data[0], *(float2*)&smem_vec[base_col + 0]);
      reg_tile.tiles[0][i].data[1] = base_ops::sub::template op<float2>(
          reg_tile.tiles[0][i].data[1], *(float2*)&smem_vec[base_col + 0]);
      reg_tile.tiles[0][i].data[2] = base_ops::sub::template op<float2>(
          reg_tile.tiles[0][i].data[2], *(float2*)&smem_vec[base_col + 8]);
      reg_tile.tiles[0][i].data[3] = base_ops::sub::template op<float2>(
          reg_tile.tiles[0][i].data[3], *(float2*)&smem_vec[base_col + 8]);
    }
  }

  __device__ static inline void common_setup(common_setup_args<layout> args) {
    int task_id = gridDim.x * args.task_iter + blockIdx.x;
    int seq_k = (args.globals.K.rows() +
                 NUM_CONSUMER_WARPGROUPS * layout::kv_tile::rows - 1) /
                (NUM_CONSUMER_WARPGROUPS * layout::kv_tile::rows);

    args.common.batch = task_id / (seq_k * args.globals.K.depth());
    task_id -= args.common.batch * seq_k * args.globals.K.depth();
    args.common.head = task_id / seq_k;
    task_id -= args.common.head * seq_k;
    args.common.seq = task_id;

    args.num_iters = args.common.batch < args.globals.Q.batch()
                         ? (args.globals.Q.rows() + layout::qo_tile::rows - 1) /
                               (layout::qo_tile::rows)
                         : -1;

    if (warpgroup::laneid() == 0) {
      if (args.num_iters > 0) {
        printf("common_setup: batch %d, head %d, seq %d, num_iters %d\n",
               args.common.batch, args.common.head, args.common.seq,
               args.num_iters);
      }
    }
  }

  struct producer {
    __device__ static inline void setup(producer_setup_args<layout> args) {
      warpgroup::producer_registers();
    }

    __device__ static inline void load(producer_load_args<layout> args) {
      if (warpgroup::warpid() == 0) {
        // Load Q, O tile and softmax logits and denom
        tma::expect(args.inputs_arrived, args.input);

        tma::load_async(args.input.q, args.globals.Q,
                        {args.common.batch, args.common.head, args.iter, 0},
                        args.inputs_arrived);

        tma::load_async(args.input.og, args.globals.dO,
                        {args.common.batch, args.common.head, args.iter, 0},
                        args.inputs_arrived);

        tma::load_async(args.input.l, args.globals.L,
                        {args.common.batch, args.common.head, 0, 0},
                        args.inputs_arrived);

        tma::load_async(args.input.d, args.globals.D,
                        {args.common.batch, args.common.head, 0, 0},
                        args.inputs_arrived);
      } else if (laneid() == 0) {
        arrive(args.inputs_arrived);
      }
    }
  };

  struct consumer {
    __device__ static inline void setup(consumer_setup_args<layout> args) {
      // 0. Initialize dK, dV registers.
      zero(args.state.kg_reg);
      zero(args.state.vg_reg);
      zero(args.state.qg_reg);

      // 1. Load K/V tiles to shared memory.
      int kv_idx = args.common.head;
      int seq_idx = args.common.seq * NUM_CONSUMER_WARPGROUPS;

      if ((seq_idx + warpgroup::groupid()) * layout::kv_tile::rows <
          args.globals.K.rows()) {
        warpgroup::load(
            args.scratch.k[warpgroup::groupid()], args.globals.K,
            {args.common.batch, kv_idx, seq_idx + warpgroup::groupid(), 0});
        warpgroup::load(
            args.scratch.v[warpgroup::groupid()], args.globals.V,
            {args.common.batch, kv_idx, seq_idx + warpgroup::groupid(), 0});
      }

      // if (warpgroup::groupid() == 0) {
      //   warpgroup::increase_registers<256>();
      // } else {
      //   warpgroup::increase_registers<224>();
      // }
    }

    __device__ static inline void compute(consumer_compute_args<layout> args) {
      zero(args.state.s_block_t);
      warpgroup::mm_ABt(
          args.state.s_block_t,                  // output: S block (float)
          args.scratch.k[warpgroup::groupid()],  // K tile (bfloat16)
          args.input.q                           // Q tile (bfloat16)
      );
      warpgroup::mma_commit_group();
      // warpgroup::mma_async_wait();

      warpgroup::mm_ABt(args.state.dp_block_t,
                        args.scratch.v[warpgroup::groupid()], args.input.og);
      warpgroup::mma_commit_group();
      warpgroup::mma_async_wait();

      // load lse into lse_block_t.
      stream_tile(args.state.lse_block_t, args.input.l);

      // TODO(KuangjuX): handle causal mask

      if constexpr (layout::head_dim == 64) {
        mul(args.state.s_block_t, args.state.s_block_t, 0.125f);
      } else {
        mul(args.state.s_block_t, args.state.s_block_t, 0.08838834764f);
      }

      sub(args.state.s_block_t, args.state.s_block_t,
          args.state.lse_block_t);  // S_ij - LSE_std_i

      mul(args.state.s_block_t, args.state.s_block_t,
          1.44269504089f);  // -S_ij

      // P = exp(S_ij - L_i)
      exp2(args.state.s_block_t, args.state.s_block_t);
      copy(args.state.p_block_t, args.state.s_block_t);
      copy(args.state.p_block_t_mma, args.state.s_block_t);

      stream_sub_tile(args.state.dp_block_t, args.input.d);
      // dS = P * dP
      mul(args.state.ds_block_t, args.state.p_block_t, args.state.dp_block_t);

      if constexpr (layout::head_dim == 64) {
        mul(args.state.ds_block_t, args.state.ds_block_t, 0.125f);
      } else {
        mul(args.state.ds_block_t, args.state.ds_block_t, 0.08838834764f);
      }

      warpgroup::mma_AB(args.state.vg_reg, args.state.p_block_t_mma,
                        args.input.og);
      warpgroup::mma_commit_group();

      // dK = dS^T * Q
      copy(args.state.ds_block_t_mma, args.state.ds_block_t);
      warpgroup::store(args.scratch.ds_smem[warpgroup::groupid()],
                       args.state.ds_block_t);

      warpgroup::mma_AB(args.state.kg_reg, args.state.ds_block_t_mma,
                        args.input.q);
      warpgroup::mma_commit_group();
      warpgroup::mma_async_wait();

      // Ensure all warpgroups have completed the current computation phase.
      // group<8>::sync(10) is a warpgroup-level synchronization primitive.
      // It synchronizes all warps within a group of 8 warps (i.e., a
      // "warpgroup"), ensuring that all warps in the group reach this point
      // before any proceed. The argument '10' is a synchronization tag or phase
      // identifier, which can help avoid conflicts between different sync
      // points in the code.
      group<8>::sync(10);
      // NOTE(KuangjuX): fix warp number to 4
      // group<4>::sync(10);

      // If this is the first consumer warpgroup (warpgroupid == 0), compute
      // Q gradient dQ = dS * K^T
      if (warpgroup::groupid() == 0) {
        // dS (float tile): args.scratch.ds_smem[0] and [1] (if double-buffered)
        // K (bfloat16 tile): args.scratch.k_smem[0] and [1] (if
        // double-buffered) Output dQ to qg_reg (float tile)
        // rt_fl<16, layout::G::tile_width> qg_reg;
        // For reference parity, we assume two K/dS tiles (double - buffered)
        warpgroup::mm_AtB(args.state.qg_reg, args.scratch.ds_smem[0],
                          args.scratch.k[0]);
        warpgroup::mma_AtB(args.state.qg_reg, args.scratch.ds_smem[1],
                           args.scratch.k[1]);
        warpgroup::mma_commit_group();
        warpgroup::mma_async_wait();

        warpgroup::store(args.scratch.qg_smem, args.state.qg_reg);
        group<4>::sync(warpgroup::groupid() + 4);

        coord<typename layout::dq_tile> tiled_idx = {blockIdx.z, blockIdx.y,
                                                     blockIdx.x, 0};
        tma::store_add_async(args.globals.qg, args.scratch.qg_smem, tiled_idx);
        tma::store_async_wait();
      }

      if (laneid() == 0) arrive(args.inputs_finished);
    }

    __device__ static inline void finish(consumer_finish_args<layout> args) {
      group<8>::sync(10);

      // Store K gradient from registers to shared memory.
      warpgroup::store(args.scratch.dk_smem[warpgroup::groupid()],
                       args.state.kg_reg);

      group<4>::sync(warpgroup::groupid() + 4);

      // Perform asynchronous atomic add for dK to global memory.
      // Only one warp (e.g., warpid() % 4 == 0) within each warpgroup
      // should initiate the TMA store to avoid redundant calls and ensure
      // atomicity if needed or simply for load balancing TMA units.
      if (warpgroup::warpid() % 4 == 0) {
        coord<typename layout::dk_tile> tile_idx = {
            blockIdx.z, blockIdx.y,
            blockIdx.x * NUM_CONSUMER_WARPGROUPS + warpgroup::groupid(), 0};
        tma::store_add_async(args.globals.kg,
                             args.scratch.dk_smem[warpgroup::groupid()],
                             tile_idx);
        tma::store_commit_group();
      }

      // Store V gradient from registers to shared memory.
      warpgroup::store(args.scratch.dv_smem[warpgroup::groupid()],
                       args.state.vg_reg);

      group<4>::sync(warpgroup::groupid() + 4);

      if (warpgroup::warpid() % 4 == 0) {
        coord<typename layout::dv_tile> tile_idx = {
            blockIdx.z, blockIdx.y,
            blockIdx.x * NUM_CONSUMER_WARPGROUPS + warpgroup::groupid(), 0};

        tma::store_add_async(args.globals.vg,
                             args.scratch.dv_smem[warpgroup::groupid()],
                             tile_idx);
        tma::store_commit_group();
      }

      tma::store_async_wait();

      // // Store dQ to shared memory for later global store
      // warpgroup::store(args.scratch.qg_smem, args.state.qg_reg);

      // // Optionally, synchronize with other warps in the warpgroup if
      // // needed
      // group<4>::sync(warpgroup::groupid() + 4);

      // if (warpgroup::warpid() % 4 == 0) {
      //   coord<typename layout::dq_tile> tiled_idx = {blockIdx.z, blockIdx.y,
      //                                                blockIdx.x, 0};
      //   tma::store_add_async(args.globals.qg, args.scratch.qg_smem,
      //   tiled_idx); tma::store_async_wait();

      //   // warpgroup::store(args.globals.qg, args.scratch.qg_smem,
      //   tiled_idx);
      // }

      if (laneid() == 0) arrive(args.finish_finished);
    }
  };
};

template <int head_dim>
struct FlashAttentionBwd {
  static void run(bf16* q, bf16* dO, bf16* k, bf16* v, float* dq, float* dk,
                  float* dv, float* l, float* d, uint32_t batch_size,
                  uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t seq_len,
                  cudaStream_t stream, uint32_t num_sms) {
    using ker_template = attn_bwd_template<head_dim, false>;

    typename ker_template::layout::qo_global Qg(q, batch_size, num_q_heads,
                                                seq_len, head_dim);
    typename ker_template::layout::kv_global Kg(k, batch_size, num_kv_heads,
                                                seq_len, head_dim);
    typename ker_template::layout::kv_global Vg(v, batch_size, num_kv_heads,
                                                seq_len, head_dim);
    typename ker_template::layout::qg_global dQg(dq, batch_size, num_q_heads,
                                                 seq_len, head_dim);
    typename ker_template::layout::kg_global dKg(dk, batch_size, num_kv_heads,
                                                 seq_len, head_dim);
    typename ker_template::layout::vg_global dVg(dv, batch_size, num_kv_heads,
                                                 seq_len, head_dim);
    typename ker_template::layout::og_global dOg(dO, batch_size, num_q_heads,
                                                 seq_len, head_dim);
    typename ker_template::layout::logits_global Lg(l, batch_size, num_q_heads,
                                                    seq_len, head_dim);
    typename ker_template::layout::denom_global Dg(d, batch_size, num_q_heads,
                                                   seq_len, head_dim);

    typename ker_template::layout::globals g{.Q = Qg,
                                             .dO = dOg,
                                             .K = Kg,
                                             .V = Vg,
                                             .qg = dQg,
                                             .kg = dKg,
                                             .vg = dVg,
                                             .L = Lg,
                                             .D = Dg};

    cudaFuncSetAttribute(prototype::lcf::kernel<ker_template>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         MAX_SHARED_MEMORY - 1024);
    dim3 block(kittens::prototype::detail::NUM_THREADS_v<ker_template>);
    // printf("thread nums: %d.\n",
    //        kittens::prototype::detail::NUM_THREADS_v<ker_template>);
    prototype::lcf::kernel<ker_template>
        <<<num_sms, block, MAX_SHARED_MEMORY - 1024, stream>>>(g);
  }
};
