
import torch 
from torch import Tensor, nn
from einops import rearrange
from dataclasses import dataclass
import time
from statistics import mean
import thunderkittens as tk

b, s, dim = 1, 4096, 3072
img_in_dim, txt_in_dim = 3072, 512
img = torch.randn(b, img_in_dim, dim, dtype=torch.float, device='cuda')
txt = torch.randn(b, txt_in_dim, dim, dtype=torch.float, device='cuda')
vec = torch.randn(b, dim, dtype=torch.float, device='cuda')
pe = torch.randn(b, 1, (img_in_dim + txt_in_dim), 64, 2, 2, dtype=torch.float, device='cuda')


def simplified_pytorch(img=img, txt=txt, vec=vec, pe=pe):
    multiplier = 6
    dim = 3072
    num_heads = 24
    head_dim = dim // num_heads

    # Modules and Weights
    lin1 = nn.Linear(dim, 6 * dim, bias=True).cuda().to(torch.bfloat16)
    lin2 = nn.Linear(dim, 6 * dim, bias=True).cuda().to(torch.bfloat16)

    norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda() 
    txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda()
    norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda()
    txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6).cuda()

    q_img_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda().to(torch.bfloat16)
    k_img_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda().to(torch.bfloat16)
    q_txt_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda().to(torch.bfloat16)
    k_txt_rms_norm_scale = nn.Parameter(torch.ones(head_dim)).cuda().to(torch.bfloat16)

    img_attn_qkv = nn.Linear(dim, dim * 3, bias=True).cuda().to(torch.bfloat16)
    txt_attn_qkv = nn.Linear(dim, dim * 3, bias=True).cuda().to(torch.bfloat16)

    img_proj = nn.Linear(dim, dim).cuda().to(torch.bfloat16)
    txt_proj = nn.Linear(dim, dim).cuda().to(torch.bfloat16)
    img_mlp = nn.Sequential(
        nn.Linear(dim, 4 * dim, bias=True),
        nn.GELU(approximate="tanh"),
        nn.Linear(4 * dim, dim, bias=True),
    ).cuda().to(torch.bfloat16)
    txt_mlp = nn.Sequential(
        nn.Linear(dim, 4 * dim, bias=True),
        nn.GELU(approximate="tanh"),
        nn.Linear(4 * dim, dim, bias=True),
    ).cuda().to(torch.bfloat16)

    # Part 1: Layer norm 
    out_img = lin1(nn.functional.silu(vec).to(torch.bfloat16))[:, None, :].chunk(multiplier, dim=-1)
    img_mod1, img_mod2 = out_img[:3], out_img[3:] 

    out_txt = lin2(nn.functional.silu(vec).to(torch.bfloat16))[:, None, :].chunk(multiplier, dim=-1)
    txt_mod1, txt_mod2 = out_txt[:3], out_txt[3:] 
    txt_modulated = (1 + txt_mod1[1].to(torch.bfloat16)) * txt_norm1(txt.to(torch.bfloat16)).to(torch.bfloat16) + txt_mod1[0].to(torch.bfloat16) 

    torch.cuda.synchronize()
    start = time.time()
    txt_modulated = (1 + txt_mod1[1].to(torch.bfloat16)) * txt_norm1(txt.to(torch.bfloat16)).to(torch.bfloat16) + txt_mod1[0].to(torch.bfloat16) 
    img_modulated = norm1(img.to(torch.bfloat16))
    img_modulated = (1 + img_mod1[1].to(torch.bfloat16)) * img_modulated.to(torch.bfloat16) + img_mod1[0].to(torch.bfloat16)
    torch.cuda.synchronize()
    end = time.time()
    # print(f"torch (s) = {end-start}")

    # print(f"Starting the layrnorm step.")
    # img_modulated_tk = torch.empty_like(img_modulated)
    # print(f"Created an empty tensor of shape: {img_modulated_tk.shape}")
    # torch.cuda.synchronize()
    # start = time.time()
    # tk.fused_flux_layernorm(
    #     img.to(torch.bfloat16).contiguous(), 
    #     img_mod1[0][0][0].to(torch.bfloat16).contiguous(), 
    #     img_mod1[1][0][0].to(torch.bfloat16).contiguous(),
    #     img_modulated_tk.to(torch.bfloat16).contiguous()
    # )
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"tk (s) = {end-start}")
    # diff = torch.norm(img_modulated - img_modulated_tk).max()
    # print(f"Diff: {diff}; TODO: Convert to floats.")


    # Part 2: RMS norms and concats   
    print(f"\nStarting the rmsnorm:")
    img_qkv = img_attn_qkv(img_modulated)
    img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=num_heads)
    txt_qkv = txt_attn_qkv(txt_modulated)
    txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=num_heads)

    print(f"Torch dtypes: {img_q.dtype}, {q_img_rms_norm_scale.dtype}")

    timings = []
    for i in range(10):
        img_q = torch.randn_like(img_q)
        img_k = torch.randn_like(img_k)
        txt_q = torch.randn_like(txt_q)
        txt_k = torch.randn_like(txt_k)
        torch.cuda.synchronize()
        start = time.time()

        rrms = torch.rsqrt(torch.mean(img_q**2, dim=-1, keepdim=True) + 1e-6)
        img_q_ref = (img_q * rrms) * q_img_rms_norm_scale.to(torch.bfloat16)
        rrms = torch.rsqrt(torch.mean(img_k**2, dim=-1, keepdim=True) + 1e-6)
        img_k_ref = (img_k * rrms) * k_img_rms_norm_scale.to(torch.bfloat16)
        rrms = torch.rsqrt(torch.mean(txt_q**2, dim=-1, keepdim=True) + 1e-6)
        txt_q_ref = (txt_q * rrms) * q_txt_rms_norm_scale.to(torch.bfloat16)
        rrms = torch.rsqrt(torch.mean(txt_k**2, dim=-1, keepdim=True) + 1e-6)
        txt_k_ref = (txt_k * rrms) * k_txt_rms_norm_scale.to(torch.bfloat16)

        q_ref = torch.cat((txt_q_ref, img_q_ref), dim=2) # torch.Size([1, 24, 4592, 128])
        k_ref = torch.cat((txt_k_ref, img_k_ref), dim=2) # torch.Size([1, 24, 4592, 128])

        torch.cuda.synchronize()
        end = time.time()
        # print(end-start)
        if i > 1: timings.append(end-start)
    print(f"torch (ms) = {mean(timings) *1000}")

    img_q_tk = torch.empty_like(q_ref).to(torch.bfloat16).contiguous()
    img_k_tk = torch.empty_like(k_ref).to(torch.bfloat16).contiguous()
    print(f"Created an empty tensor of shape: {img_q_tk.shape}")

    timings = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.time()
        tk.fused_flux_rmsnorm(
            img_q.to(torch.bfloat16).contiguous(), 
            img_k.to(torch.bfloat16).contiguous(), 
            txt_q.to(torch.bfloat16).contiguous(), 
            txt_k.to(torch.bfloat16).contiguous(), 

            q_img_rms_norm_scale.to(torch.bfloat16).contiguous(),
            k_img_rms_norm_scale.to(torch.bfloat16).contiguous(),
            q_txt_rms_norm_scale.to(torch.bfloat16).contiguous(),
            q_txt_rms_norm_scale.to(torch.bfloat16).contiguous(), 

            img_q_tk,
            img_k_tk
        )
        torch.cuda.synchronize()
        end = time.time()
        timings.append(end-start)
        # print(end-start)
    print(f"tk (ms) = {mean(timings)*1000}")
    diff = torch.norm(q_ref - img_q_tk).max() 
    print(f"Diff: {diff=}")

    # Attention
    # attn = attention(q, k, v, pe=pe)
    attn = torch.randn(b, (txt_in_dim + img_in_dim), dim, dtype=q_ref.dtype, device=q_ref.device) 
    txt_attn, img_attn = attn[:, : txt_in_dim], attn[:, txt_in_dim :]

    timings = []
    for i in range(10):
        torch.cuda.synchronize()
        start = time.time()

        img_proj_out = img_proj(img_attn).to(torch.bfloat16)
        txt_proj_out = txt_proj(txt_attn).to(torch.bfloat16)

        img = img + img_mod1[2] * img_proj_out
        img_mlp_in = (1 + img_mod2[1]) * norm2(img) + img_mod2[0]
        img = img + img_mod2[2] * img_mlp(img_mlp_in.to(torch.bfloat16))

        txt = txt + txt_mod1[2] * txt_proj_out
        txt_mlp_in = (1 + txt_mod2[1]) * txt_norm2(txt) + txt_mod2[0]
        txt = txt + txt_mod2[2] * txt_mlp(txt_mlp_in.to(torch.bfloat16))

        torch.cuda.synchronize()
        end = time.time()
        timings.append(end-start)
        # print(end-start)
    # print(f"torch (ms) = {mean(timings)*1000}")s

    return img_q_ref

simplified_pytorch(img=img, txt=txt, vec=vec, pe=pe)

