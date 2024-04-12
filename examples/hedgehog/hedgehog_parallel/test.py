# -*- coding: utf-8 -*-

import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from hedgehog import HedgehogBased


def main():
    batch_size = 1
    n_heads = 1
    seq_lens = [256, 512, 1024]
    dtypes = [torch.bfloat16]
    
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    for seq_len in seq_lens:
        for dtype in dtypes:
            
            torch.manual_seed(0)
            
            q = (torch.randn(batch_size, n_heads, seq_len, 16).cuda().to(dtype) / 16).requires_grad_(True)
            k = (torch.randn(batch_size, n_heads, seq_len, 16).cuda().to(dtype) / 16).requires_grad_(True)
            v = (torch.randn(batch_size, n_heads, seq_len, 128).cuda().to(dtype) / 128).requires_grad_(True)
            
            q_tri = q.clone().detach().requires_grad_(True)
            k_tri = k.clone().detach().requires_grad_(True)
            v_tri = v.clone().detach().requires_grad_(True)
            
            torch.manual_seed(1)
            model_ref = HedgehogBased(num_heads=n_heads, head_dim=16, feature_dim=128, input_dim=16, dtype=dtype, zero_init=False, use_triton=False)
            ref = model_ref(q, k, v, False, False)
            
            torch.manual_seed(1)
            model_tri = HedgehogBased(num_heads=n_heads, head_dim=16, feature_dim=128, input_dim=16, dtype=dtype, zero_init=False, use_triton=True)
            tri = model_tri(q_tri, k_tri, v_tri, False, False)
            
            assert q.allclose(q_tri, 1e-1)
            assert k.allclose(k_tri, 1e-1)
            assert v.allclose(v_tri, 1e-1)
            
            # here, 1e-1 comes from error seen when 
            # we use a fake einsum implementation in pytorch
            if not ref.allclose(tri, atol=1e-1):
                # print out idx of mismatched values
                print(f"Test failed for seq_len={seq_len}, dtype={dtype}")
                print("Shapes:")
                print(q.shape, k.shape, v.shape, ref.shape)
                
                print("Unacceptable Difference:")
                print(torch.abs(ref - tri).max())
            else:
                print(f"Test passed for seq_len={seq_len}, dtype={dtype}")
                print("Shapes:")
                print(q.shape, k.shape, v.shape, ref.shape)
                
                print("Acceptable Difference:")
                print(torch.abs(ref - tri).max())
            print("---------------------------------------------------")
            
                
            # print out q, k, v, ref in a file in outputs/
            with open(f"{output_dir}/seq_len_{seq_len}_dtype_{dtype}.txt", "w") as f:
                f.write(f"ref: {ref}\n")
            
            # print out q_tri, k_tri, v_tri, ref in a file in outputs/
            with open(f"{output_dir}/seq_len_{seq_len}_dtype_{dtype}_tri.txt", "w") as f:
                f.write(f"tri: {tri}\n")


if __name__ == "__main__":
    main()
