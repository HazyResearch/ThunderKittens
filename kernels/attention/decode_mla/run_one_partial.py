import torch
import mla_decode
from timings import save_gantt_chart

torch.manual_seed(0)

batch = 1
heads = 16
new_seq = 4

L = 2048

instructions = torch.zeros((batch,1,32), dtype=torch.int32, device='cuda')

q = torch.ones((batch, new_seq, heads, 512), dtype=torch.bfloat16, device='cuda')
q_rot = torch.ones((batch, new_seq, heads, 64), dtype=torch.bfloat16, device='cuda')

kv_cache = torch.randn((256*batch, 256, 512), dtype=torch.bfloat16, device='cuda')
k_rot_cache = torch.ones((256*batch, 256, 64), dtype=torch.bfloat16, device='cuda')

table = torch.arange(256*batch, dtype=torch.int32, device='cuda').reshape((batch,256))

o = torch.zeros((batch, new_seq, heads, 512), dtype=torch.bfloat16, device='cuda')
o_scratch = torch.zeros((1, new_seq, heads, 512), dtype=torch.float32, device='cuda')
lvec_scratch = torch.zeros((1, 1, new_seq, heads), dtype=torch.float32, device='cuda')

semaphore = torch.zeros((1, new_seq), dtype=torch.int32, device='cuda')

Softmax_scale = (1/576)**.5
tic = 1

timings = torch.zeros((batch, 1, 64), dtype=torch.int32, device='cuda')

instructions[0,0,:9] = torch.tensor([
  1, # Opcode
  0, # Uid
  0, # dst.batch_idx
  0, # dst.seq_idx
  0, # q_batch_idx
  0, # q_seq_idx
  0, # start_pos
  L, # end_pos
  L, # length
], dtype=torch.int32, device='cuda')

def compute_ref(q, q_rot, k_rot_cache, kv_cache, L):
    q_cat = torch.cat((q, q_rot), dim=-1)
    k_cat = torch.cat((kv_cache, k_rot_cache), dim=-1).reshape((-1,576))[:L]
    # print('k_cat sum 0', k_cat[:,:64].sum(dim=-1))
    # print('k_cat sum ALL', k_cat.sum(dim=-1))
    logits = torch.einsum('bnhd,ld->bnhl', q_cat, k_cat).to(torch.float32)
    # print('LOGITS', logits)
    logits *= Softmax_scale
    # probs = torch.nn.functional.softmax(logits * Softmax_scale, dim=-1)
    logits -= logits.max(dim=-1, keepdim=True)[0]
    logits = torch.exp(logits)
    # print(logits.to(torch.bfloat16))
    probs = logits / logits.sum(dim=-1, keepdim=True)
    # print(probs)
    v   = kv_cache.reshape((-1,512))[:L]
    out = torch.einsum('bnhl,ld->bnhd', probs.to(torch.bfloat16), v)
    return out


mla_decode.mla_decode(instructions, q_rot, q, k_rot_cache, kv_cache, table, o, o_scratch, lvec_scratch, semaphore, Softmax_scale, tic, timings)
torch.cuda.synchronize()
ref = compute_ref(q, q_rot, k_rot_cache, kv_cache, L)

print(f'o mean: {o.abs().mean()}, ref mean: {ref.abs().mean()}')
print(f'o max: {o.abs().max()}, ref max: {ref.abs().max()}')

print('avg diff:', (o-ref).abs().mean())


# save_gantt_chart(timings, instructions, save_all=True, name='single')

breakpoint()