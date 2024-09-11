import torch
import torch.nn as nn
from tqdm import trange
import numpy as np
import sys
import math
from einops import rearrange

B = 1
multiplier = 6
num_heads = 1
img_in_dim, txt_in_dim = 3072, 512
head_dim = 128
print(f"{head_dim=}")

TESTNAME = sys.argv[1]

# Modules and Weights
q_img_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()
k_img_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()
q_txt_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()
k_txt_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda()

if TESTNAME == 'ones':
    rms_input_img_q = torch.randn(B, num_heads, img_in_dim, head_dim, dtype=torch.bfloat16, device='cuda')/head_dim
    rms_input_img_k = torch.randn(B, num_heads, img_in_dim, head_dim, dtype=torch.bfloat16, device='cuda')/head_dim
    rms_input_txt_q = torch.randn(B, num_heads, txt_in_dim, head_dim, dtype=torch.bfloat16, device='cuda')/head_dim
    rms_input_txt_k = torch.randn(B, num_heads, txt_in_dim, head_dim, dtype=torch.bfloat16, device='cuda')/head_dim

else:
    print('Invalid test name')
    sys.exit(0)

def get_output(rms_input_img_q, rms_input_img_k, rms_input_txt_q, rms_input_txt_k):

    rrms = torch.rsqrt(torch.mean(rms_input_img_q**2, dim=-1, keepdim=True) + 1e-6)
    img_q = (rms_input_img_q * rrms) * q_img_rms_norm_scale

    rrms = torch.rsqrt(torch.mean(rms_input_img_k**2, dim=-1, keepdim=True) + 1e-6)
    img_k = (rms_input_img_k * rrms) * k_img_rms_norm_scale

    rrms = torch.rsqrt(torch.mean(rms_input_txt_q**2, dim=-1, keepdim=True) + 1e-6)
    txt_q = (rms_input_txt_q * rrms) * q_txt_rms_norm_scale

    rrms = torch.rsqrt(torch.mean(rms_input_txt_k**2, dim=-1, keepdim=True) + 1e-6)
    txt_k = (rms_input_txt_k * rrms) * k_txt_rms_norm_scale

    q = torch.cat((txt_q, img_q), dim=2)
    k = torch.cat((txt_k, img_k), dim=2)

    return q.float(), k.float(), q_img_rms_norm_scale.float(), k_img_rms_norm_scale.float(), q_txt_rms_norm_scale.float(),  k_img_rms_norm_scale.float()

q_out, k_out, rms_q_scale, rms_k_scale, rms_q_scale_txt, rms_k_scale_txt = get_output(rms_input_img_q, rms_input_img_k, rms_input_txt_q, rms_input_txt_k)

print(f"{q_out.shape=}, {k_out.shape=}")

with open(f'{TESTNAME}.txt', 'w') as f:
    rms_input_img_q_f = rms_input_img_q.to(torch.float32).flatten().detach().cpu().numpy()
    rms_q_scale_f = rms_q_scale.to(torch.float32).flatten().detach().cpu().numpy()
    # img_q_f = img_q.to(torch.float32).flatten().detach().cpu().numpy()

    rms_input_img_k_f = rms_input_img_k.to(torch.float32).flatten().detach().cpu().numpy()
    rms_k_scale_f = rms_k_scale.to(torch.float32).flatten().detach().cpu().numpy()
    # img_k_f = img_k.to(torch.float32).flatten().detach().cpu().numpy()

    rms_input_txt_q_f = rms_input_txt_q.to(torch.float32).flatten().detach().cpu().numpy()
    rms_q_scale_f_txt = rms_q_scale_txt.to(torch.float32).flatten().detach().cpu().numpy()
    # txt_q_f = txt_q.to(torch.float32).flatten().detach().cpu().numpy()

    rms_input_txt_k_f = rms_input_txt_k.to(torch.float32).flatten().detach().cpu().numpy()
    rms_k_scale_f_txt = rms_k_scale_txt.to(torch.float32).flatten().detach().cpu().numpy()
    # txt_k_f = txt_k.to(torch.float32).flatten().detach().cpu().numpy()

    q_f = q_out.to(torch.float32).flatten().detach().cpu().numpy()
    k_f = k_out.to(torch.float32).flatten().detach().cpu().numpy()

    for i in trange(head_dim):
        f.write(repr(rms_q_scale_f[i]))
        f.write(' ')
    for i in trange(B*num_heads*img_in_dim*head_dim):
        f.write(repr(rms_input_img_q_f[i]))
        f.write(' ')
    # for i in trange(B*num_heads*img_in_dim*head_dim):
    #     f.write(repr(img_q_f[i]))
    #     f.write(' ')

    for i in trange(head_dim):
        f.write(repr(rms_k_scale_f[i]))
        f.write(' ')
    for i in trange(B*num_heads*img_in_dim*head_dim):
        f.write(repr(rms_input_img_k_f[i]))
        f.write(' ')
    # for i in trange(B*num_heads*img_in_dim*head_dim):
    #     f.write(repr(img_k_f[i]))
    #     f.write(' ')

    for i in trange(head_dim):
        f.write(repr(rms_q_scale_f_txt[i]))
        f.write(' ')
    for i in trange(B*num_heads*txt_in_dim*head_dim):
        f.write(repr(rms_input_txt_q_f[i]))
        f.write(' ')
    # for i in trange(B*num_heads*txt_in_dim*head_dim):
    #     f.write(repr(txt_q_f[i]))
    #     f.write(' ')
    
    for i in trange(head_dim):
        f.write(repr(rms_k_scale_f_txt[i]))
        f.write(' ')
    for i in trange(B*num_heads*txt_in_dim*head_dim):
        f.write(repr(rms_input_txt_k_f[i]))
        f.write(' ')
    # for i in trange(B*num_heads*txt_in_dim*head_dim):
    #     f.write(repr(txt_k_f[i]))
    #     f.write(' ')


    for i in trange(B*num_heads*(img_in_dim+txt_in_dim)*head_dim):
        f.write(repr(q_f[i]))
        if ( i < 3 ): print(q_f[i])
        f.write(' ')
    for i in trange(B*num_heads*(img_in_dim+txt_in_dim)*head_dim):
        f.write(repr(k_f[i]))
        f.write(' ')
   