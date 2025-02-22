import torch
import thunderkittens

b, N, K, M = 2, 64, 512, 64
A = torch.rand([b, N, K], dtype=torch.bfloat16, device="cuda")
B = torch.rand([b, M, K], dtype=torch.bfloat16, device="cuda")
assert thunderkittens.batch_matmul(A, B).allnear(A@B)

