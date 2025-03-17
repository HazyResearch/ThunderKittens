import itertools
import torch
from gpt2_decode import gpt2_decode, OpCode
import torch.nn.functional as F

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
    mid_residual = torch.zeros(seq_len, EMBED_DIM, dtype=dtype, device=device)
    mid_first_norm = torch.zeros(seq_len, EMBED_DIM, dtype=dtype, device=device)
    weight_qkv = torch.rand(EMBED_DIM, 3 * EMBED_DIM, dtype=dtype, device=device)
    bias_qkv = torch.rand(3 * EMBED_DIM, dtype=dtype, device=device)
    mid_qkv = torch.zeros(seq_len, 3 * EMBED_DIM, dtype=dtype, device=device)
    mid_attn = torch.zeros(seq_len, EMBED_DIM, dtype=dtype, device=device)
    weight_proj = torch.rand(EMBED_DIM, EMBED_DIM, dtype=dtype, device=device)
    mid_proj = torch.zeros(seq_len, EMBED_DIM, dtype=dtype, device=device)
    output_residual = torch.zeros(seq_len, EMBED_DIM, dtype=dtype, device=device)
    mid_second_norm = torch.zeros(seq_len, EMBED_DIM, dtype=dtype, device=device)
    weight_ff_expand = torch.rand(EMBED_DIM, 4 * EMBED_DIM, dtype=dtype, device=device)
    mid_ff_expand = torch.zeros(seq_len, 4 * EMBED_DIM, dtype=dtype, device=device)
    weight_ff_contract = torch.rand(4 * EMBED_DIM, EMBED_DIM, dtype=dtype, device=device)
    output_hidden = torch.zeros(seq_len, EMBED_DIM, dtype=dtype, device=device)

    gpt2_decode(instructions, 
                input_hidden, 
                input_residual, 
                mid_residual, 
                mid_first_norm, 
                weight_qkv, 
                bias_qkv,
                mid_qkv, 
                mid_attn, 
                weight_proj, 
                mid_proj, 
                output_residual, 
                mid_second_norm,
                weight_ff_expand,
                mid_ff_expand,
                weight_ff_contract,
                output_hidden)
    
    print('mid_residual:', ((input_hidden + input_residual) - mid_residual).abs().max().item(), mid_residual.std().item())
    print('mid_first_norm:', (F.layer_norm(mid_residual, (EMBED_DIM, )) - mid_first_norm).abs().max().item(), mid_first_norm.std().item())
    print('mid_qkv:', ((mid_first_norm @ weight_qkv + bias_qkv) - mid_qkv).abs().max().item(), mid_qkv.std().item())
    print('mid_attn:', (mid_qkv[:, :EMBED_DIM] - mid_attn).abs().max().item(), mid_attn.std().item())
    print('mid_proj:', (mid_attn @ weight_proj - mid_proj).abs().max().item(), mid_proj.std().item())
    print('output_residual:', ((mid_proj + mid_residual) - output_residual).abs().max().item(), output_residual.std().item())
    print('mid_second_norm:', (F.layer_norm(output_residual, (EMBED_DIM, )) - mid_second_norm).abs().max().item(), mid_second_norm.std().item())
    print('mid_ff_expand:', (F.gelu(mid_second_norm @ weight_ff_expand) - mid_ff_expand).abs().max().item(), mid_ff_expand.std().item())
    print('output_hidden:', (mid_ff_expand @ weight_ff_contract - output_hidden).abs().max().item(), output_hidden.std().item())
    