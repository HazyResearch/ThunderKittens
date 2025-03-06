import mha_decode
import torch
import numpy as np
import math
import heapq
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from scheduler_v2 import backward_schedule
from scheduler import sample_schedule_generator, priority_schedule_tasks, visualize_schedule, create_arguments_from_task_schedule
from timings import save_gantt_chart
from pprint import pprint
torch.manual_seed(0)

# D_Main, D_Rot = 512, 64
HEAD_DIM = 128
PAGE_SIZE = 256
H = 2                   # H heads
NUM_PAGES = 1000        # number of pages in cache
NUM_PROCESSORS = 132    # number of processors
MAX_NUM_PAGES = 65536 // PAGE_SIZE

ENABLE_TIMINGS = True

def init_arguments(seq_lengths: List[int], NEW_TOKENS: int):

    B = len(seq_lengths)

    # Need to initialize QRot, QV, K_cache, V_cache, Lengths, Table    
    
    # QRot    = torch.randn(B, NEW_TOKENS, H, D_Rot, dtype=torch.bfloat16, device='cuda')
    # QV      = torch.randn(B, NEW_TOKENS, H, D_Main, dtype=torch.bfloat16, device='cuda')
    Q       = torch.randn(B, NEW_TOKENS, H, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    # K_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D_Rot, dtype=torch.bfloat16, device='cuda')
    # V_cache = torch.randn(NUM_PAGES, PAGE_SIZE, D_Main, dtype=torch.bfloat16, device='cuda')
    K_cache = torch.randn(NUM_PAGES, PAGE_SIZE, H, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    V_cache = torch.randn(NUM_PAGES, PAGE_SIZE, H, HEAD_DIM, dtype=torch.bfloat16, device='cuda')
    
    Lengths = torch.tensor(seq_lengths, dtype=torch.int32, device='cuda')
    # Table = torch.arange(MAX_NUM_PAGES, dtype=torch.int32, device='cuda').reshape(1, -1).repeat(B, 1)
    Table = torch.randint(0, NUM_PAGES, (B, MAX_NUM_PAGES), dtype=torch.int32, device='cuda')

    # return QRot, QV, K_cache, V_cache, Lengths, Table
    return Q, K_cache, V_cache, Lengths, Table

def create_thundermha_arguments(seq_lengths, new_tokens, num_heads):
    # Processor assignment heuristic: assign processors proportionally to sequence lengths.
    t0 = time.time()
    # chunking = 128
    chunking = max(32, (round(1.05*sum(seq_lengths)*num_heads//NUM_PROCESSORS)//32)*32)
    print(f'Chunking: {chunking}')
    tasks = sample_schedule_generator(new_tokens=new_tokens, num_heads=num_heads, lengths=seq_lengths, chunkings=chunking)
    scheduled_tasks = priority_schedule_tasks(tasks, NUM_PROCESSORS)
    t1 = time.time()
    from pprint import pprint
    pprint(scheduled_tasks)
    print(f'Time taken to create schedule: {(t1-t0)*1000} ms')
    visualize_schedule(scheduled_tasks, NUM_PROCESSORS)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_arguments_from_task_schedule(
        scheduled_tasks, new_tokens, num_processors=NUM_PROCESSORS, num_heads=num_heads, enable_timings=ENABLE_TIMINGS
    )
    pprint('Instructions:')
    pprint(Instructions[:6].tolist())
    return Instructions, O_scratch, Lvec_scratch, Semaphore, Timings

def run_thundermha(Q, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings, tic=None):
    if tic is None:
        Semaphore.zero_()
        tic = 1
    O = torch.zeros_like(Q)
    softmax_scale = 1.0 / math.sqrt(HEAD_DIM)
    
    torch.cuda.synchronize()
    mha_decode.mha_decode(Instructions, Q, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic, Timings)
    # mla_decode.mla_decode(Instructions,QRot, QV, K_cache, V_cache, Table, O, O_scratch, Lvec_scratch, Semaphore, softmax_scale, tic, Timings)
    torch.cuda.synchronize()
    
    return O

def run_mha_torch(Q, K_cache, V_cache, Lengths, Table):
    B, q_tokens, H, head_dim = Q.shape
    full_K = K_cache[Table].reshape(B, -1, H, head_dim)
    full_V = V_cache[Table].reshape(B, -1, H, head_dim)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    O = torch.zeros_like(Q)
    for b in range(B):
        l = Lengths[b].item() 
        Q_b = Q[b:b+1].transpose(1, 2)
        K_b = full_K[b, :l].unsqueeze(0).transpose(1, 2)
        V_b = full_V[b, :l].unsqueeze(0).transpose(1, 2)
        mask = torch.ones(Q.shape[1], l, dtype=torch.bool).tril(diagonal=l-Q.shape[1]).to(Q.device)
        O_b = torch.nn.functional.scaled_dot_product_attention(
            Q_b,
            K_b, 
            V_b, 
            attn_mask=mask, 
            dropout_p=0.0, 
            is_causal=False, 
            scale=softmax_scale
        )
        
        O[b:b+1] = O_b.transpose(1, 2)    
    return O

# def run_mla_torch(QRot, QV, K_cache, V_cache, Lengths, Table):
#     Q = torch.concat([QRot, QV], dim=-1)
#     full_K = torch.cat([K_cache, V_cache], dim=-1)[Table].reshape(Q.shape[0], -1, Q.shape[-1])
#     full_V = V_cache[Table].reshape(Q.shape[0], -1, QV.shape[-1])
#     softmax_scale = 1.0 / math.sqrt(D_Main+D_Rot)
#     O = torch.zeros_like(QV)
#     for b, l in enumerate(Lengths):
#         assert Q.shape[1] == 1, "Q must have shape (B, 1, H, D) for the time being."
#         O[b:b+1] = torch.nn.functional.scaled_dot_product_attention(
#             Q[b:b+1].transpose(1, 2),
#             full_K[b:b+1, :l].unsqueeze(-2).repeat((1,1,H,1)).transpose(1, 2),
#             full_V[b:b+1, :l].unsqueeze(-2).repeat((1,1,H,1)).transpose(1, 2),
#             is_causal=False,
#             scale=softmax_scale
#         ).transpose(1, 2)
#     return O

def main():
    # seq_lengths=sorted([32768*2])
    seq_lengths=sorted([65536])
    # seq_lengths=sorted([4641,45118,1730,1696])
    
    NEW_TOKENS = 16
    Q, K_cache, V_cache, Lengths, Table = init_arguments(seq_lengths, NEW_TOKENS)
    ref = run_mha_torch(Q, K_cache, V_cache, Lengths, Table)
    Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_thundermha_arguments(seq_lengths, NEW_TOKENS, H)
    O = run_thundermha(Q, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings)
    # QRot, QV, K_cache, V_cache, Lengths, Table = init_arguments(seq_lengths, NEW_TOKENS)
    # ref = run_mla_torch(QRot, QV, K_cache, V_cache, Lengths, Table)
    # Instructions, O_scratch, Lvec_scratch, Semaphore, Timings = create_thundermla_arguments(seq_lengths, NEW_TOKENS)
    # O = run_thundermla(QRot, QV, K_cache, V_cache, Lengths, Table, Instructions, O_scratch, Lvec_scratch, Semaphore, Timings)
    
    print('O', O)
    print('Ref', ref)
    for b in range(len(seq_lengths)):
        print("ref mean:", torch.mean(ref[b].abs()))
        print("Kernel output mean:", torch.mean(O[b].abs()))
        print("Max absolute diff:", torch.max(torch.abs(O[b] - ref[b])))
        print("Avg absolute diff:", torch.mean(torch.abs(O[b] - ref[b])))
    save_gantt_chart(Timings, Instructions)
    breakpoint()

if __name__ == "__main__":
    main()