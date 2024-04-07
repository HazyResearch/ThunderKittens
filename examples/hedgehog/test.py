# -*- coding: utf-8 -*-

import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from examples.hedgehog.hedgehog import HedgehogBased


def main():
    batch_size = 2
    n_heads = 4
    seq_lens = [1024]
    dtypes = [torch.float16, torch.bfloat16, torch.float32]
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    for seq_len in seq_lens:
        for dtype in dtypes:
            
            torch.manual_seed(42)
            
            q = (torch.randn(batch_size, n_heads, seq_len, 16).cuda().to(dtype)).requires_grad_(True)
            k = (torch.randn(batch_size, n_heads, seq_len, 16).cuda().to(dtype)).requires_grad_(True)
            v = torch.randn(batch_size, n_heads, seq_len, 128).cuda().to(dtype).requires_grad_(True)
            
            q_tri = q.clone().detach().requires_grad_(True)
            k_tri = k.clone().detach().requires_grad_(True)
            v_tri = v.clone().detach().requires_grad_(True)
                        
            # model_ref = HedgehogBased(num_heads=n_heads, head_dim=16, feature_dim=128, input_dim=16, dtype=dtype, zero_init=False, use_triton=False)
            # ref = model_ref(q, k, v, False, False)
            
            model_tri = HedgehogBased(num_heads=n_heads, head_dim=16, feature_dim=128, input_dim=16, dtype=dtype, zero_init=False, use_triton=True)
            tri = model_tri(q_tri, k_tri, v_tri, False, False)
            
            # assert q.shape == (batch_size, n_heads, seq_len, 16)
            # assert k.shape == (batch_size, n_heads, seq_len, 16)
            # assert v.shape == (batch_size, n_heads, seq_len, 128)
            # assert ref.shape == (batch_size, n_heads, seq_len, 128)
            
            # save q, k, v and output in a single .txt file in a folder named outputs
            # with open(f"outputs/naive_{seq_len}_{dtype}.txt", "w") as f:
            #     f.write(f"q: {q}\n\nk: {k}\n\nv: {v}\n\nref: {ref}")
            
            print(f"Test completed for seq_len={seq_len}, dtype={dtype}")
            print("Shapes:")
            print(q.shape, k.shape, v.shape, ref.shape)


if __name__ == "__main__":
    main()
