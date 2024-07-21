import torch
import sys

# These installs are for pulling in the TK source
sys.path.append('../../../')
from src.common.pyutils.test_build_utils import __eq
sys.path.append('build/lib.linux-x86_64-cpython-311')
try:
    sys.path.append("../H100")
    import add as mod
    print(f"Successfully imported TK kernel")
except:
    mod = None
    print("Could not import TK  kernel.")

def pytorch_test(K):
    k_a1_cumsum =  K.to(torch.float32).cumsum(dim=2)
    return k_a1_cumsum

def tk_test(K, use_reg=True, dt=torch.bfloat16):
    b, h, n, d = K.shape
    k_state_a1 = torch.zeros((b, h, n, d), dtype=dt, device='cuda')
    mod.tk_cumulative_sum(int(use_reg), K, k_state_a1)
    return k_state_a1

B, H, N, D = 4, 16, 1024, 16
k = torch.randn((B, H, N, D), dtype=torch.bfloat16, device='cuda')/D
k_a1_cumsum_ref = pytorch_test(k)
k_a1_cumsum = tk_test(k)

print(f"{k_a1_cumsum_ref.shape=}, {k_a1_cumsum.shape=}")
diff = torch.norm(k_a1_cumsum_ref-k_a1_cumsum).item()
print(f"{diff=}")

print("\nToken 0:")
print(k_a1_cumsum_ref[0,0,0,:8])
print(k_a1_cumsum[0,0,0,:8])

print("\nToken 1:")
print(k_a1_cumsum_ref[0,0,1,:8])
print(k_a1_cumsum[0,0,1,:8])

print("\nTokens 62-66:")
print(k_a1_cumsum_ref[1,1,-4:,])
print(k_a1_cumsum[1,1,-4:,])






