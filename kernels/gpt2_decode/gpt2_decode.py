import itertools
import torch
from gpt2_decode import gpt2_decode, OpCode
import torch.nn.functional as F
from transformers import GPT2Model

dtype = torch.bfloat16
device = 'cuda'
torch.manual_seed(0)

# General params
SM_COUNT = 1
INST_SIZE = 4

# MM params
TILE_DIM = 64
M_BLOCK = 2
N_BLOCK = 4

# GPT2 params
NUM_HEADS = 12
HEAD_DIM = 64
EMBED_DIM = NUM_HEADS * HEAD_DIM

def get_layernorm_inst(opcode):
    instructions = torch.zeros(SM_COUNT, 1, INST_SIZE, dtype=torch.int, device=device)
    instructions[0,0,0] = opcode.value
    return instructions

def get_mm_inst(opcode, m, n):
    
    # Compute the tiles and the number of instructions required
    tiles = list(itertools.product(range(m // (TILE_DIM * M_BLOCK)), range(n // (TILE_DIM * N_BLOCK))))
    num_inst = (len(tiles) + SM_COUNT - 1) // SM_COUNT
    
    # Allocate and fill in the instructions
    instructions = torch.zeros(SM_COUNT, num_inst, INST_SIZE, dtype=torch.int, device=device)
    for inst in range(num_inst):
        for i, tile in enumerate(tiles[inst * SM_COUNT: (inst + 1) * SM_COUNT]):
            instructions[i, inst, 0] = opcode.value
            instructions[i, inst, 1], instructions[i, inst, 2] = tile
    
    return instructions

def get_attn_inst():
    instructions = torch.zeros(SM_COUNT, 1, INST_SIZE, dtype=torch.int, device=device)
    instructions[0,0,0] = OpCode.ATTENTION.value
    return instructions

def get_null_inst():
    return torch.zeros(SM_COUNT, 1, INST_SIZE, dtype=torch.int, device=device)

if __name__ == '__main__':
    
    model = GPT2Model.from_pretrained('gpt2').to(device=device, dtype=dtype)
    
    seq_len = 256

    instructions = torch.cat(
        (
            get_layernorm_inst(OpCode.FIRST_NORM), 
            get_mm_inst(OpCode.QKV, seq_len, 3 * EMBED_DIM), 
            get_attn_inst(),
            get_mm_inst(OpCode.PROJECTION, seq_len, EMBED_DIM), 
            get_layernorm_inst(OpCode.SECOND_NORM), 
            get_mm_inst(OpCode.FF_EXPAND, seq_len, 4 * EMBED_DIM),
            get_mm_inst(OpCode.FF_CONTRACT, seq_len, EMBED_DIM),
            get_null_inst()
         ), 
        dim=1
    )
    
    input_hidden = torch.rand(seq_len, EMBED_DIM, dtype=dtype, device=device)
    input_residual = torch.rand(seq_len, EMBED_DIM, dtype=dtype, device=device)
    gamma_first_norm = model.h[0].ln_1.weight.detach()
    beta_first_norm = model.h[0].ln_1.bias.detach()
    mid_residual = torch.empty(seq_len, EMBED_DIM, dtype=dtype, device=device)
    mid_first_norm = torch.empty(seq_len, EMBED_DIM, dtype=dtype, device=device)
    weight_qkv = model.h[0].attn.c_attn.weight.detach()
    bias_qkv = model.h[0].attn.c_attn.bias.detach()
    mid_qkv = torch.empty(seq_len, 3 * EMBED_DIM, dtype=dtype, device=device)
    mid_attn = torch.empty(seq_len, EMBED_DIM, dtype=dtype, device=device)
    weight_proj = model.h[0].attn.c_proj.weight.detach()
    bias_proj = model.h[0].attn.c_proj.bias.detach()
    mid_proj = torch.empty(seq_len, EMBED_DIM, dtype=dtype, device=device)
    gamma_second_norm = model.h[0].ln_2.weight.detach()
    beta_second_norm = model.h[0].ln_2.bias.detach()
    output_residual = torch.empty(seq_len, EMBED_DIM, dtype=dtype, device=device)
    mid_second_norm = torch.empty(seq_len, EMBED_DIM, dtype=dtype, device=device)
    weight_ff_expand = model.h[0].mlp.c_fc.weight.detach()
    bias_ff_expand = model.h[0].mlp.c_fc.bias.detach()
    mid_ff_expand = torch.empty(seq_len, 4 * EMBED_DIM, dtype=dtype, device=device)
    weight_ff_contract = model.h[0].mlp.c_proj.weight.detach()
    bias_ff_contract = model.h[0].mlp.c_proj.bias.detach()
    output_hidden = torch.empty(seq_len, EMBED_DIM, dtype=dtype, device=device)

    gpt2_decode(instructions, 
                input_hidden, 
                input_residual, 
                gamma_first_norm,
                beta_first_norm,
                mid_residual, 
                mid_first_norm, 
                weight_qkv, 
                bias_qkv,
                mid_qkv, 
                mid_attn, 
                weight_proj, 
                bias_proj,
                mid_proj, 
                gamma_second_norm,
                beta_second_norm,
                output_residual, 
                mid_second_norm,
                weight_ff_expand,
                bias_ff_expand,
                mid_ff_expand,
                weight_ff_contract,
                bias_ff_contract,
                output_hidden)
    
    print('mid_residual:', ((input_hidden + input_residual) - mid_residual).abs().max().item(), mid_residual.std().item())
    print('mid_first_norm:', (model.h[0].ln_1(mid_residual) - mid_first_norm).abs().max().item(), mid_first_norm.std().item())
    print('mid_qkv:', (model.h[0].attn.c_attn(mid_first_norm) - mid_qkv).abs().max().item(), mid_qkv.std().item())
    print('mid_attn:', (mid_qkv[:, :EMBED_DIM] - mid_attn).abs().max().item(), mid_attn.std().item())
    print('mid_proj:', (model.h[0].attn.c_proj(mid_attn) - mid_proj).abs().max().item(), mid_proj.std().item())
    print('output_residual:', ((mid_proj + mid_residual) - output_residual).abs().max().item(), output_residual.std().item())
    print('mid_second_norm:', (model.h[0].ln_2(output_residual) - mid_second_norm).abs().max().item(), mid_second_norm.std().item())
    print('mid_ff_expand:', (F.gelu(model.h[0].mlp.c_fc(mid_second_norm)) - mid_ff_expand).abs().max().item(), mid_ff_expand.std().item())
    print('output_hidden:', (model.h[0].mlp.c_proj(mid_ff_expand) - output_hidden).abs().max().item(), output_hidden.std().item())
    