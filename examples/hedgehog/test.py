# -*- coding: utf-8 -*-

import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from examples.hedgehog.naive import HedgehogBased


def main():
    batch_size = 4
    n_heads = 4
    seq_lens = [300, 512]
    hidden_sizes = [8, 15]
    dtypes = [torch.float16, torch.bfloat16, torch.float32]

    for seq_len in seq_lens:
        for hidden_size in hidden_sizes:
            for dtype in dtypes:
                
                torch.manual_seed(42)
                q = (torch.randn(batch_size, n_heads, seq_len, 16).cuda().to(dtype)).requires_grad_(True)
                k = (torch.randn(batch_size, n_heads, seq_len, 16).cuda().to(dtype)).requires_grad_(True)
                v = torch.randn(batch_size, n_heads, seq_len, 128).cuda().to(dtype).requires_grad_(True)
                            
                model = HedgehogBased(num_heads=n_heads, head_dim=16, feature_dim=128, input_dim=16, dtype=dtype)
                ref = model(q, k, v, False, True)
                
                assert q.shape == (batch_size, n_heads, seq_len, 16)
                assert k.shape == (batch_size, n_heads, seq_len, 16)
                assert v.shape == (batch_size, n_heads, seq_len, 128)
                assert ref.shape == (batch_size, n_heads, seq_len, 128)
                
                print(f"Test passed for seq_len={seq_len}, hidden_size={hidden_size}, dtype={dtype}")
                print(q.shape, k.shape, v.shape, ref.shape)


if __name__ == "__main__":
    main()
