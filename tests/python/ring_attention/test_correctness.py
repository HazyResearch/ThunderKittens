from functools import partial
import gc

import jax
import numpy as np
import torch

from implementations import (
    generate_mha_inputs,
    mha_torch,
    mha_jax,
    ring_mha_orig
)


# Parameters
NUM_DEVICES = 8
B = 8
H = 16
N = 1024
D_h = 32
dtype = 'bf16'
causal = False


def run_and_clean(fn, target):

    print(f'Running {target}...')

    # Run
    Q, K, V, dL = generate_mha_inputs(B, H, N, D_h, dtype=dtype, target=target, num_devices=NUM_DEVICES)
    result = fn(Q, K, V, causal)

    # Clean up
    for i in range(NUM_DEVICES):
        torch.cuda.synchronize(i)
    del Q, K, V, dL
    gc.collect() # this will do for jax
    torch.cuda.empty_cache()

    return result


if __name__ == '__main__':

    assert NUM_DEVICES >= 1, 'NUM_DEVICES must be >= 1'
    assert N % NUM_DEVICES == 0, 'N must be divisible by NUM_DEVICES'

    # Torch MHA, Single Device
    out_torch = run_and_clean(mha_torch, 'mha_torch')

    # Jax MHA, Single Device
    out_jax = run_and_clean(mha_jax, 'mha_jax')

    # Original Ring MHA
    out_ring_orig = run_and_clean(partial(ring_mha_orig, num_devices=NUM_DEVICES), 'ring_mha_orig')

    # Verify correctness. Output shape is (batch, seq, feature)
    TOL = 1e-2 # large due to bf16
    out_torch = out_torch.detach().to(dtype=torch.float32, device='cpu').numpy()
    out_jax = np.array(jax.device_get(out_jax)).astype(np.float32)
    out_ring_orig = np.array(jax.device_get(out_ring_orig)).astype(np.float32)
    max_error = np.max(np.abs(out_torch - out_ring_orig))
    num_errors = np.sum(np.abs(out_torch - out_ring_orig) > TOL)
    print(f'Max abs diff: {max_error}')
    print(f'Num errors: {num_errors} out of {out_torch.size} (TOL: {TOL})')
