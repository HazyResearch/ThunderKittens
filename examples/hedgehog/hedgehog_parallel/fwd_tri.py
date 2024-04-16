import torch 

import triton 
import triton.language as tl 
from torch.cuda.amp import custom_bwd, custom_fwd

@triton.jit
def parallel_based_fwd_kernel_hedgehog(
    q,  # query [B, H, N, D]
    k,  # key [B, H, N, D]
    v,  # value [B, H, N, DV]
    o,  # output [B, H, N, DV]
    s_qk_h,  # stride size: N * D
    s_qk_t,  # stride size: D
    s_qk_d,  # stride size: 1
    s_vo_h,  # stride size: N * DV
    s_vo_t,  # stride size: DV
    s_vo_d,  # stride size: 1
    B,  # batch size
    H,  # n_heads
    N,  # seq_len
    scale,  # D ** -0.5
    BS_q_n: tl.constexpr,  # BLOCK SIZE along the sequence dimension for Q
    BS_kv_n: tl.constexpr,  # BLOCK SIZE along the sequence dimension for K/V
    BS_k_d: tl.constexpr,  # BLOCK SIZE along the K dimension
    BS_v_dv: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D
    DV: tl.constexpr,  # DV
):
    i_kv, i_n, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    NV = tl.cdiv(DV, BS_v_dv)
    i_k = i_kv // (NV)
    i_v = i_kv % (NV)

    p_q = tl.make_block_ptr(q + i_bh * s_qk_h, (N, DK),
                            (s_qk_t, s_qk_d), (i_n * BS_q_n, i_k * BS_k_d), (BS_q_n, BS_k_d), (1, 0))
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, N),
                            (s_qk_d, s_qk_t), (i_k * BS_k_d, 0), (BS_k_d, BS_kv_n), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (N, DV),
                            (s_vo_t, s_vo_d), (0, i_v * BS_v_dv), (BS_kv_n, BS_v_dv), (1, 0))

    # block Q - in the shared memory
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    
    b_o = tl.zeros([BS_q_n, BS_v_dv], dtype=tl.float32)

    # Q block and K block (no mask part)
    for _ in range(0, i_n * BS_q_n, BS_kv_n):
        # [BS_k_d, BS_kv_n]
        b_k = tl.load(p_k, boundary_check=(0, 1))

        # [BS_kv_n, BS_v_dv]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        
        # [BS_q_n, BS_kv_n]
        b_s = tl.dot(b_q, (b_k), allow_tf32=False)

        # [BQ, BD]
        b_o = b_o + tl.dot(b_s.to(b_v.dtype), b_v, allow_tf32=False)
        
        p_k = tl.advance(p_k, (0, BS_kv_n))
        p_v = tl.advance(p_v, (BS_kv_n, 0))

    # rescale interchunk output
    tl.debug_barrier()
    o_q = tl.arange(0, BS_q_n)
    
    # sync threads, easy for compiler to optimize
    tl.debug_barrier()

    o_k = tl.arange(0, BS_kv_n)
    
    p_k = tl.make_block_ptr(k + i_bh * s_qk_h, (DK, N),
                            (s_qk_d, s_qk_t), (i_k * BS_k_d, i_n * BS_q_n), (BS_k_d, BS_kv_n), (0, 1))
    p_v = tl.make_block_ptr(v + i_bh * s_vo_h, (N, DV),
                            (s_vo_t, s_vo_d), (i_n * BS_q_n, i_v * BS_v_dv), (BS_kv_n, BS_v_dv), (1, 0))
    
    # Q block and K block (masked part)
    for _ in range(i_n * BS_q_n, (i_n + 1) * BS_q_n, BS_kv_n):
        # [BS_k_d, BS_kv_n]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        
        # [BS_kv_n, BS_v_dv]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        
        # [BS_q_n, BS_kv_n]
        m_s = o_q[:, None] >= o_k[None, :]
        b_s = tl.dot(b_q, b_k, allow_tf32=False)
        b_s = tl.where(m_s, b_s, 0)
        
        # [BS_q_n, BS_v_dv]
        b_o += tl.dot(b_s.to(b_q.dtype), b_v, allow_tf32=False)

        p_k = tl.advance(p_k, (0, BS_kv_n))
        p_v = tl.advance(p_v, (BS_kv_n, 0))
        o_k += BS_kv_n

    p_o = tl.make_block_ptr(o + (i_bh + B * H * i_k) * s_vo_h, (N, DV),
                            (s_vo_t, s_vo_d), 
                            (i_n*BS_q_n, i_v*BS_v_dv), 
                            (BS_q_n, BS_v_dv), (1, 0))
    
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))