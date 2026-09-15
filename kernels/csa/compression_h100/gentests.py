import torch
import numpy as np
import sys

print("Generating tests. This will take a moment.")

torch.random.manual_seed(42)

B = 2
N = 316 

TESTNAME = sys.argv[1] if len(sys.argv) > 1 else 'randn'


def make_tensor(shape, dtype=torch.bfloat16):
    if TESTNAME == 'ones':
        base = torch.ones(shape, device='cuda')
    elif TESTNAME == 'randn':
        base = torch.randn(shape, device='cuda')
    else:
        print('Invalid test name')
        sys.exit(1)
    return base.to(dtype)


def compression_reference(value_a, value_b, score_a, score_b, bias_a, bias_b, M):
    """
    Reference implementation of CSA. Matches correctness.cu's.
    """
    B, N, C = value_a.shape
    num_blocks = N // M
    out = torch.zeros((B, num_blocks, C), dtype=torch.float32, device=value_a.device)
    for b in range(B):
        for i in range(num_blocks):
            za = score_a[b, i * M:(i + 1) * M, :].float() + bias_a.float()  # [M, C]
            z_slices = [za]
            value_slices = [value_a[b, i * M:(i + 1) * M, :].float()]
            if i > 0:
                zb = score_b[b, (i - 1) * M:i * M, :].float() + bias_b.float()  # [M, C]
                z_slices.append(zb)
                value_slices.append(value_b[b, (i - 1) * M:i * M, :].float())
            z = torch.cat(z_slices, dim=0)      # [M, C] (i=0) or [2M, C]
            val = torch.cat(value_slices, dim=0)
            S = torch.softmax(z, dim=0)         # per-channel softmax down the rows
            out[b, i, :] = (S * val).sum(dim=0)
    return out


def write_case(f, C, M):
    value_a = make_tensor((B, N, C), dtype=torch.float8_e4m3fn)
    value_b = make_tensor((B, N, C), dtype=torch.float8_e4m3fn)
    score_a = make_tensor((B, N, C), dtype=torch.float8_e4m3fn)
    score_b = make_tensor((B, N, C), dtype=torch.float8_e4m3fn)
    bias_a = make_tensor((M, C))
    bias_b = make_tensor((M, C))
    ref = compression_reference(value_a, value_b, score_a, score_b, bias_a, bias_b, M)

    # Sanity-check the reference: softmax weights should sum to 1 per channel per block.
    num_blocks = N // M
    for b in range(B):
        for i in range(num_blocks):
            za = score_a[b, i * M:(i + 1) * M, :].float() + bias_a.float()
            z_slices = [za]
            if i > 0:
                zb = score_b[b, (i - 1) * M:i * M, :].float() + bias_b.float()
                z_slices.append(zb)
            z = torch.cat(z_slices, dim=0)
            S = torch.softmax(z, dim=0)
            assert torch.allclose(S.sum(dim=0), torch.ones_like(S.sum(dim=0)), atol=1e-3), \
                f"softmax weights don't sum to 1 at b={b}, i={i}"

    # Order here MUST match run_case<C,M>()'s read order in correctness.cu's
    # standalone main(): value_a, value_b, score_a, score_b, bias_a, bias_b, ref_compressed.
    for name, t in [('value_a', value_a), ('value_b', value_b),
                     ('score_a', score_a), ('score_b', score_b),
                     ('bias_a', bias_a), ('bias_b', bias_b),
                     ('ref_compressed', ref)]:
        arr = t.to(torch.float32).flatten().detach().cpu().numpy()
        print(f"  Writing {name} ({arr.size} values)...")
        np.savetxt(f, arr.reshape(1, -1), fmt='%.8g', delimiter=' ', newline=' ')


fn = f'{TESTNAME}_compression_test.txt'
with open(fn, 'w') as f:
    # Real DeepSeek-V4 CSA paper values
    print(f"C=512, M=4, B={B}, N={N}")
    write_case(f, C=512, M=4)

print(f"Wrote {fn}")
