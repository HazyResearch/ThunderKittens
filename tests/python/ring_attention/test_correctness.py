import jax
import numpy as np
import torch

from implementations import (
    generate_mha_inputs,
    mha_torch,
    mha_jax,
    ring_mha_torch,
    ring_mha_orig
)
from tk_ring_attention import ring_mha_forward


# Parameters
NUM_DEVICES = 8
B = 4
H = 8
N = 768 * NUM_DEVICES # sequence length (must be 768 * i)
D_h = 64 # head dimension (do not change)
dtype = 'bf16'
causal = False

assert NUM_DEVICES >= 1, 'NUM_DEVICES must be >= 1'
assert N % NUM_DEVICES == 0, 'N must be divisible by NUM_DEVICES'


def check_correctness(A: np.ndarray, B: np.ndarray, TOL: float = 1e-2):

    assert(A.shape == B.shape)
    assert(A.dtype == B.dtype)
    abs_diff = np.abs(B - A)
    max_error = np.max(abs_diff)
    num_errors = np.sum(abs_diff > TOL)
    print(f'Max abs diff: {max_error}')
    print(f'Num errors: {num_errors} out of {B.size} (TOL: {TOL})')


if __name__ == '__main__':

    torch.cuda.set_device(0)

    print('\nRunning Reference MHA...')
    Q, K, V, dL = generate_mha_inputs(B, H, N, D_h, dtype=dtype, target='mha_torch')
    out_ref = mha_torch(Q, K, V, causal)
    out_ref = out_ref.detach().to(dtype=torch.float32, device='cpu').numpy()

    print('\nRunning Jax MHA...')
    Q, K, V, dL = generate_mha_inputs(B, H, N, D_h, dtype=dtype, target='mha_jax')
    out_jax = mha_jax(Q, K, V, causal)
    out_jax = np.array(jax.device_get(out_jax)).astype(np.float32)
    check_correctness(out_ref, out_jax)

    print('\nRunning PyTorch Ring Attention...')
    Q, K, V, dL = generate_mha_inputs(B, H, N, D_h, dtype=dtype, target='mha_torch')
    out_torch_ring = ring_mha_torch(Q, K, V, causal, num_devices=NUM_DEVICES)
    out_torch_ring = out_torch_ring.detach().to(dtype=torch.float32, device='cpu').numpy()
    check_correctness(out_ref, out_torch_ring)

    print('\nRunning Original Ring Attention...')
    Q, K, V, dL = generate_mha_inputs(B, H, N, D_h, dtype=dtype, target='ring_mha_orig', num_devices=NUM_DEVICES)
    out_ring_orig = ring_mha_orig(Q, K, V, causal, num_devices=NUM_DEVICES)
    out_ring_orig = np.array(jax.device_get(out_ring_orig)).astype(np.float32)
    check_correctness(out_ref, out_ring_orig)

    print('\nRunning ThunderKittens Ring Attention...')
    Qs, Ks, Vs, dLs = generate_mha_inputs(B, H, N, D_h, dtype=dtype, target='ring_mha_tk', num_devices=NUM_DEVICES)
    out_tk_ring = ring_mha_forward(Qs, Ks, Vs, causal)
    out_tk_ring = [t.detach().to(dtype=torch.float32, device='cpu').numpy() for t in out_tk_ring]
    out_tk_ring = np.concatenate(out_tk_ring, axis=2)
    check_correctness(out_ref, out_tk_ring)
