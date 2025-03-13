import torch
import gpt2_decode
import torch.nn.functional as F

num_instructions = 2
instructions = torch.zeros(132, num_instructions, 4, dtype=torch.int, device='cuda')
instructions[0][0] = 1

batch_size = 64
hidden_dim = 64
hidden = torch.arange(batch_size * hidden_dim, dtype=torch.bfloat16, device='cuda').view(batch_size, hidden_dim)

ref = F.layer_norm(hidden, (hidden_dim, ))

gpt2_decode.gpt2_decode(instructions, hidden)

print(torch.allclose(ref, hidden, atol=1e-1))
