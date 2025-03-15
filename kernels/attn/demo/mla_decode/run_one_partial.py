import torch
import mla_decode
from timings import save_gantt_chart

batch = 1
heads = 16
new_seq = 8

instructions = torch.zeros((batch,1,32), dtype=torch.int32, device='cuda')

q = torch.zeros((batch, new_seq, heads, 512), dtype=torch.bfloat16, device='cuda')
q_rot = torch.zeros((batch, new_seq, heads, 64), dtype=torch.bfloat16, device='cuda')

kv_cache = torch.ones((256*batch, 256, 512), dtype=torch.bfloat16, device='cuda')
k_rot_cache = torch.zeros((256*batch, 256, 64), dtype=torch.bfloat16, device='cuda')

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
  256, # end_pos
  256, # length
], dtype=torch.int32, device='cuda')

mla_decode.mla_decode(instructions, q_rot, q, k_rot_cache, kv_cache, table, o, o_scratch, lvec_scratch, semaphore, Softmax_scale, tic, timings)
torch.cuda.synchronize()
print(o)
print('Timings:', timings)
save_gantt_chart(timings, instructions, save_all=True, name='single')
