# Copyright (c) 2024 Doraemonzzz

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    Out,
    S,  # log lambda
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    BLOCK_MODEL: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_h = off_bh % h
    off_e = tl.program_id(1)
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    # channel offset
    e_offset = off_e * BLOCK_MODEL

    ##### get block ptr
    Q_block_ptr = Q + qk_offset + tl.arange(0, d)[None, :]
    K_trans_block_ptr = K + qk_offset + tl.arange(0, d)[:, None]
    V_block_ptr = V + v_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    O_block_ptr = Out + o_offset + e_offset + tl.arange(0, BLOCK_MODEL)[None, :]
    S_block_ptr = S + off_h

    ##### init diag decay(Lambda); q, k decay; kv
    s = tl.load(S_block_ptr)
    # q, k decay
    off_block = tl.arange(
        0, BLOCK
    )  # Not bug, this is a bit different from algorithm 1, but is mathematically equivalent
    q_decay = tl.exp(-s.to(tl.float32) * off_block[:, None])
    k_trans_decay = tl.exp(-s.to(tl.float32) * (BLOCK - off_block[None, :]))
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)
    # diag decay
    index = off_block[:, None] - off_block[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)
    
    kv = tl.zeros([d, BLOCK_MODEL], dtype=tl.float32)

    ##### compute
    for i in range(NUM_BLOCK):
        # load
        q = tl.load(
            Q_block_ptr + off_block[:, None] * d, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)
        k_trans = tl.load(
            K_trans_block_ptr + off_block[None, :] * d,
            mask=off_block[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr + off_block[:, None] * e, mask=off_block[:, None] < n, other=0.0
        ).to(tl.float32)

        # compute
        qk = tl.dot(q, k_trans) * diag_decay
        o_intra = tl.dot(qk, v)
        o_inter = tl.dot(q, kv) * q_decay
        o = o_intra + o_inter

        # save and update
        tl.store(
            O_block_ptr + off_block[:, None] * e,
            o.to(O_block_ptr.dtype.element_ty),
            mask=off_block[:, None] < n,
        )
        kv = block_decay * kv + tl.dot(k_trans * k_trans_decay, v)
        off_block += BLOCK


@triton.jit
def _bwd_intra_kernel(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_block = tl.program_id(1)
    off_h = off_bh % h
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    block_offset = off_block * BLOCK + tl.arange(0, BLOCK)

    ##### get block ptr
    Q_trans_block_ptr = (
        Q + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
    )
    K_block_ptr = K + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
    V_trans_block_ptr = (
        V + v_offset + block_offset[None, :] * e + tl.arange(0, e)[:, None]
    )

    DQ_block_ptr = DQ + qk_offset + block_offset[:, None] * d + tl.arange(0, d)[None, :]
    DK_trans_block_ptr = (
        DK + qk_offset + block_offset[None, :] * d + tl.arange(0, d)[:, None]
    )
    DV_block_ptr = DV + v_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]
    DO_block_ptr = DO + o_offset + block_offset[:, None] * e + tl.arange(0, e)[None, :]

    S_block_ptr = S + off_h

    ##### init diag decay(Lambda)
    s = tl.load(S_block_ptr)
    array = tl.arange(0, BLOCK).to(tl.float32)
    # diag
    index = array[:, None] - array[None, :]
    s_index = s * index
    s_index = tl.where(index >= 0, -s_index, float("-inf"))
    diag_decay = tl.exp(s_index)
    diag_decay_trans = tl.trans(diag_decay)

    ##### load block
    k = tl.load(K_block_ptr, mask=block_offset[:, None] < n, other=0.0).to(tl.float32)
    v_trans = tl.load(V_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0).to(
        tl.float32
    )
    do = tl.load(DO_block_ptr, mask=block_offset[:, None] < n, other=0.0).to(tl.float32)
    q_trans = tl.load(Q_trans_block_ptr, mask=block_offset[None, :] < n, other=0.0).to(
        tl.float32
    )

    ##### compute
    dqk = tl.dot(do, v_trans) * diag_decay
    dq_intra = tl.dot(dqk, k)

    dk_intra_trans = tl.dot(q_trans, dqk)

    qk_trans = tl.dot(k, q_trans) * diag_decay_trans
    dv_intra = tl.dot(qk_trans, do)

    dq = dq_intra
    dk_trans = dk_intra_trans
    dv = dv_intra

    # save
    tl.store(
        DQ_block_ptr,
        dq.to(DQ_block_ptr.dtype.element_ty),
        mask=block_offset[:, None] < n,
    )
    tl.store(
        DK_trans_block_ptr,
        dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
        mask=block_offset[None, :] < n,
    )
    tl.store(
        DV_block_ptr,
        dv.to(DV_block_ptr.dtype.element_ty),
        mask=block_offset[:, None] < n,
    )


@triton.jit
def _bwd_inter_kernel(
    Q,
    K,
    V,
    S,
    DO,
    DQ,
    DK,
    DV,
    b: tl.constexpr,
    h: tl.constexpr,
    n: tl.constexpr,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    ##### get offset
    off_bh = tl.program_id(0)
    off_h = off_bh % h

    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e
    S_block_ptr = S + off_h

    ##### get block ptr
    DQ_block_ptr = (
        DQ + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    K_block_ptr = (
        K + qk_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V + v_offset + tl.arange(0, CBLOCK)[None, :] * e + tl.arange(0, e)[:, None]
    )
    DO_block_ptr = (
        DO + o_offset + tl.arange(0, CBLOCK)[:, None] * e + tl.arange(0, e)[None, :]
    )
    # mask
    off_block1 = tl.arange(0, CBLOCK)
    off_block2 = tl.arange(0, CBLOCK)
    # compute block array
    c_array = tl.arange(0, CBLOCK)

    ##### init lambda; kv
    s = tl.load(S_block_ptr)
    block_decay = tl.exp(-s.to(tl.float32) * BLOCK)
    kv_trans = tl.zeros([e, d], dtype=tl.float32)

    ##### compute dq inter
    for i in range(NUM_BLOCK):
        # compute in subblock
        for j in range(NUM_CBLOCK):
            if i > 0:  # if not add this, may have bug
                q_decay = tl.exp(-s.to(tl.float32) * (j * CBLOCK + c_array[:, None]))
                do = tl.load(DO_block_ptr, mask=off_block1[:, None] < n, other=0.0).to(
                    tl.float32
                )
                dq_inter = tl.dot(do, kv_trans) * q_decay
                dq = dq_inter + tl.load(
                    DQ_block_ptr, mask=off_block1[:, None] < n, other=0.0
                )
                tl.store(
                    DQ_block_ptr,
                    dq.to(DQ_block_ptr.dtype.element_ty),
                    mask=off_block1[:, None] < n,
                )

            DQ_block_ptr += CBLOCK * d
            DO_block_ptr += CBLOCK * e
            off_block1 += CBLOCK

        # update kv in subblock
        kv_trans_current = tl.zeros([e, d], dtype=tl.float32)
        for j in range(NUM_CBLOCK):
            v_trans = tl.load(
                V_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0
            ).to(tl.float32)
            k = tl.load(K_block_ptr, mask=off_block2[:, None] < n, other=0.0).to(
                tl.float32
            )
            k_decay = tl.exp(
                -s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[:, None]))
            )
            kv_trans_current += tl.dot(v_trans, k * k_decay)

            K_block_ptr += CBLOCK * d
            V_trans_block_ptr += CBLOCK * e
            off_block2 += CBLOCK

        kv_trans = block_decay * kv_trans + kv_trans_current

    ##### get block ptr
    m = NUM_BLOCK * BLOCK
    off_block1 = m + tl.arange(0, CBLOCK)
    off_block2 = m + tl.arange(0, CBLOCK)

    Q_trans_block_ptr = (
        Q
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    K_block_ptr = (
        K
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    V_trans_block_ptr = (
        V
        + v_offset
        + m * e
        + tl.arange(0, CBLOCK)[None, :] * e
        + tl.arange(0, e)[:, None]
    )

    DK_trans_block_ptr = (
        DK
        + qk_offset
        + m * d
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    DV_block_ptr = (
        DV
        + v_offset
        + m * e
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    DO_block_ptr = (
        DO
        + o_offset
        + m * e
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    ##### init dkv
    dkv = tl.zeros([d, e], dtype=tl.float32)

    ##### compute dk, dv inter
    for i in range(NUM_BLOCK - 1, -1, -1):
        # compute in subblock
        for j in range(NUM_CBLOCK - 1, -1, -1):
            K_block_ptr -= CBLOCK * d
            V_trans_block_ptr -= CBLOCK * e
            DK_trans_block_ptr -= CBLOCK * d
            DV_block_ptr -= CBLOCK * e
            off_block1 -= CBLOCK

            if i < NUM_BLOCK - 1:  # if not add this, may have bug
                k = tl.load(K_block_ptr, mask=off_block1[:, None] < n, other=0.0).to(
                    tl.float32
                )
                v_trans = tl.load(
                    V_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0
                ).to(tl.float32)

                k_decay_trans = tl.exp(
                    -s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[None, :]))
                )
                k_decay = tl.exp(
                    -s.to(tl.float32) * (BLOCK - (j * CBLOCK + c_array[:, None]))
                )
                dk_inter_trans = tl.dot(dkv, v_trans) * k_decay_trans
                dv_inter = tl.dot(k, dkv) * k_decay

                dk_trans = dk_inter_trans + tl.load(
                    DK_trans_block_ptr, mask=off_block1[None, :] < n, other=0.0
                )
                dv = dv_inter + tl.load(
                    DV_block_ptr, mask=off_block1[:, None] < n, other=0.0
                )

                tl.store(
                    DK_trans_block_ptr,
                    dk_trans.to(DK_trans_block_ptr.dtype.element_ty),
                    mask=off_block1[None, :] < n,
                )
                tl.store(
                    DV_block_ptr,
                    dv.to(DV_block_ptr.dtype.element_ty),
                    mask=off_block1[:, None] < n,
                )

        # update dkv in subblock
        dkv_current = tl.zeros([d, e], dtype=tl.float32)
        for j in range(NUM_CBLOCK - 1, -1, -1):
            DO_block_ptr -= CBLOCK * e
            Q_trans_block_ptr -= CBLOCK * d
            off_block2 -= CBLOCK

            do = tl.load(DO_block_ptr, mask=off_block2[:, None] < n, other=0.0).to(
                tl.float32
            )
            q_trans = tl.load(
                Q_trans_block_ptr, mask=off_block2[None, :] < n, other=0.0
            ).to(tl.float32)
            q_decay_trans = tl.exp(-s.to(tl.float32) * (j * CBLOCK + c_array[None, :]))
            dkv_current += tl.dot(q_trans * q_decay_trans, do)

        dkv = block_decay * dkv + dkv_current


class LightningAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, s):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()

        b, h, n, d = q.shape
        e = v.shape[-1]
        o = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        BLOCK = 64
        NUM_BLOCK = triton.cdiv(q.shape[2], BLOCK)
        # parallel over channel
        BLOCK_MODEL = min(triton.next_power_of_2(e), 32)
        grid = (b * h, triton.cdiv(e, BLOCK_MODEL))

        _fwd_kernel[grid](
            q,
            k,
            v,
            o,
            s,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            BLOCK_MODEL=BLOCK_MODEL,
        )

        ctx.save_for_backward(q, k, v, s)

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, s = ctx.saved_tensors

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        s = s.contiguous()
        do = do.contiguous()

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        b, h, n, d = q.shape
        e = v.shape[-1]

        # block size
        BLOCK = 64
        NUM_BLOCK = triton.cdiv(n, BLOCK)
        # compute block size
        CBLOCK = 32
        NUM_CBLOCK = BLOCK // CBLOCK

        # for intra part, compute in parallel
        grid = (b * h, NUM_BLOCK)
        _bwd_intra_kernel[grid](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        # for inter part, compute in sequencial
        grid = (b * h,)
        _bwd_inter_kernel[grid](
            q,
            k,
            v,
            s,
            do,
            dq,
            dk,
            dv,
            b,
            h,
            n,
            d,
            e,
            BLOCK=BLOCK,
            NUM_BLOCK=NUM_BLOCK,
            CBLOCK=CBLOCK,
            NUM_CBLOCK=NUM_CBLOCK,
        )

        return dq, dk, dv, None, None


lightning_attn2 = LightningAttention2.apply