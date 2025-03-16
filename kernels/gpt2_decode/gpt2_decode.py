import torch
import gpt2_decode
import torch.nn.functional as F

dtype = torch.bfloat16
device = 'cuda'

num_instructions = 3
instructions = torch.zeros(132, num_instructions, 4, dtype=torch.int, device='cuda')
instructions[0,0,0] = 1

instructions[0,1,0] = 2
instructions[1,1,0] = 2
instructions[1,1,1] = 1

layer_input = torch.rand(256, 256, dtype=dtype, device=device)
after_first_norm = torch.zeros(256, 256, dtype=dtype, device=device)
qkv_weights = torch.rand(256, 256, dtype=dtype, device=device)
qkv = torch.zeros(256, 256, dtype=dtype, device=device)

ref = F.layer_norm(layer_input, (256, )) @ qkv_weights

gpt2_decode.gpt2_decode(instructions, layer_input, after_first_norm, qkv_weights, qkv)
print(ref)
print(after_first_norm)
print(qkv)
print((qkv - ref).abs().max())

# print(torch.allclose(A @ B, C))

# gpt2_decode.gpt2_decode(instructions, hidden)

# print(torch.allclose(ref, hidden, atol=1e-1))
