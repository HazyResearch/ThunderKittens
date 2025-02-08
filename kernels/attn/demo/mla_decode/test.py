import mla_decode
import torch

torch.manual_seed(0)

from torch.nn.functional import linear, scaled_dot_product_attention as sdpa

B, NEW_TOKENS, H, D_QK, D_VO, PAGE_SIZE, MAX_LENGTH = 2, 1, 1, 576, 512, 64, 32768
MAX_PAGES = MAX_LENGTH//PAGE_SIZE

CACHE_PAGES = 1000

Q = torch.randn((B, NEW_TOKENS, H, D_QK), device='cuda', dtype=torch.bfloat16)
Cache = torch.ones((CACHE_PAGES, PAGE_SIZE, D_QK), device='cuda', dtype=torch.bfloat16)
# Q = torch.randn((B, NEW_TOKENS, H, D_QK), device='cuda', dtype=torch.bfloat16)
# Cache = torch.randn((CACHE_PAGES, PAGE_SIZE, D_QK), device='cuda', dtype=torch.bfloat16)
Lengths = torch.randint(0, MAX_LENGTH, (B,), device='cuda', dtype=torch.int32)
Table = torch.randint(0, CACHE_PAGES, (B, MAX_PAGES), device='cuda', dtype=torch.int32)
O = torch.zeros((B, NEW_TOKENS, H, D_VO), device='cuda', dtype=torch.bfloat16)

sizes = (Lengths + (PAGE_SIZE - 1)).floor_divide(PAGE_SIZE).to(device='cuda', dtype=torch.int32)

total = Lengths.sum().item()
maximum = Lengths.max().item()

sequence_ids, position_ids = (
        torch.arange(MAX_PAGES, dtype=torch.int32, device='cuda')[None, :]
        .expand(B, -1)
        .lt(sizes.view(-1, 1))
        .nonzero(as_tuple=True)
    )

Table[sequence_ids, position_ids] = (
        torch.randperm(CACHE_PAGES, device='cuda')
        [:sequence_ids.size(0)]
        .sort()
        .values
        .int()
    )

latent = torch.ones(total, 1, D_QK, dtype=torch.bfloat16, device='cuda')
expanded = latent.expand(total, H, D_QK)

sequence_ids, position_ids = (
        torch.arange(maximum, device='cuda')[None, :]
        .expand(B, -1)
        .lt(Lengths.view(-1, 1))
        .nonzero(as_tuple=True)
    )

padded_key = torch.zeros(B, maximum, H, D_QK, dtype=torch.bfloat16, device='cuda')
padded_value = torch.zeros(B, maximum, H, D_VO, dtype=torch.bfloat16, device='cuda')

padded_key[sequence_ids, position_ids] = expanded
padded_value[sequence_ids, position_ids] = expanded[..., :D_VO]

entry_ids = Table[sequence_ids, position_ids.floor_divide(PAGE_SIZE)]
column_ids = position_ids.fmod(PAGE_SIZE)

Cache[entry_ids, column_ids] = latent.squeeze(1)

mask = None
R = NEW_TOKENS - 1
if True:
    bounds = (
        torch.arange(R + 1, dtype=torch.int32, device='cuda')[None, :]
        + Lengths[:, None]
        - R
    )

    mask = (
        torch.arange(maximum, dtype=torch.int32, device='cuda')[None, None, None, :]
        .expand(B, H, R + 1, -1)
        .lt(bounds[:, None, :, None].expand(B, H, R + 1, 1))
    )


print('Starting')
mla_decode.mla_decode(Q, Cache, Lengths, Table, O)
torch.cuda.synchronize()
print('Finished')

print(O)
print(torch.mean(O.abs()))

Sz = 1.0 / (D_QK ** 0.5)
padded_o1 = sdpa(
    query=Q.transpose(1, 2),
    key=padded_key.transpose(1, 2),
    value=padded_value.transpose(1, 2),
    attn_mask=mask,
    dropout_p=0.0,
    is_causal=False,
    scale=Sz,
    enable_gqa=False,
)

O_ref = padded_o1.transpose(1, 2)

print("Reference output:")
print(O_ref)
print(torch.mean(O_ref.abs()))

print("\nMax difference:", torch.max(torch.abs(O - O_ref)).item())
print("Mean difference:", torch.mean(torch.abs(O - O_ref)).item())
