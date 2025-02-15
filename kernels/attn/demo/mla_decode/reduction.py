import torch

# Parameters.
batch = 1
num_heads = 1
seq_q = 3
seq_k = 6   
d_model = 4

chunk_size = seq_k // 2

torch.manual_seed(0)
Q = torch.randn(batch, num_heads, seq_q, d_model)
K = torch.randn(batch, num_heads, seq_k, d_model)
V = torch.randn(batch, num_heads, seq_k, d_model)

scale = 1.0 / (d_model ** 0.5)

chunk_max = torch.randn(batch, num_heads, chunk_size, 1).fill_(-torch.inf)
sum_exp   = torch.zeros_like(chunk_max)

# --- First chunk ---
K0 = K[..., :chunk_size, :]
V0 = V[..., :chunk_size, :]

scores0     = torch.matmul(Q, K0.transpose(-1, -2)) * scale  # [B, H, seq_q, 3]
chunk_max   = torch.max(chunk_max, scores0.max(dim=-1, keepdim=True).values)
exp_scores0 = torch.exp(scores0 - chunk_max)
sum_exp    += exp_scores0.sum(dim=-1, keepdim=True)

l0   = chunk_max + torch.log(sum_exp)  # LSE_0: [B, H, seq_q, 1]
out0 = torch.matmul(exp_scores0, V0)   # O_0:   [B, H, seq_q, d_model]

# --- Second chunk ---
K1 = K[..., chunk_size:, :]
V1 = V[..., chunk_size:, :]

scores1     = torch.matmul(Q, K1.transpose(-1, -2)) * scale  # [B, H, seq_q, 3]
chunk_max   = torch.max(chunk_max, scores1.max(dim=-1, keepdim=True).values)
exp_scores1 = torch.exp(scores1 - chunk_max)
sum_exp    += exp_scores1.sum(dim=-1, keepdim=True)

l1 = chunk_max + torch.log(sum_exp)  # LSE_1: [B, H, seq_q, 1]
out1 = torch.matmul(exp_scores1, V1) # O_1:   [B, H, seq_q, d_model]

########################################################
# --- Reduction ---
########################################################

# LSE_final = log(sum(exp(LSE_i)))
lse_final = torch.log(torch.exp(l1))

# Compute the final log-sum-exp value.
final_output = (torch.exp(l0 - lse_final) * out0) + (torch.exp(l1 - lse_final) * out1)

print("Final Attention Output:\n", final_output)
