
import torch
import thunderkittens as tk

N = 1024
hidden_states = torch.randn(N, N, device="cuda").to(torch.float8_e4m3fn).contiguous()
weights = torch.randn(N, N, device="cuda").to(torch.float8_e4m3fn).contiguous()
output = tk.fp8_gemm(hidden_states, weights)

