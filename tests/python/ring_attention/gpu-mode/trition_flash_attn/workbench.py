import math
import torch
from flash_attn_triton import flash_attn_func
from flash_attn_triton_og import attention


def naive_attn(q, k, v, scale):
    s = q @ k.mT * scale
    a = torch.softmax(s, dim=-1)
    return a @ v


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    """reference implementation copied from the pytorch documentation:
    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def test_openai_attention():
    dtype = torch.float16
    device = torch.device("cuda:0")

    batch_size, num_heads, head_dim = 2, 2, 64
    for sequence_length in [16, 32, 64, 128, 256, 512, 1024]:

        q = torch.randn(batch_size, num_heads, sequence_length, head_dim, dtype=dtype, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        scale_factor = q.size(-1) ** -0.5
        causal = False

        a = naive_attn(q, k, v, scale_factor)
        b = attention(q, k, v, causal, scale_factor)

        print(f"sequence_length={sequence_length}, all_close={torch.allclose(a, b, atol=0.01)}, delta (mean)={torch.abs(a-b).mean()}")


def main():
    test_openai_attention()


if __name__ == '__main__':
    main()
