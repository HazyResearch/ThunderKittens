import itertools
import torch
import gpt2_decode
import torch.nn.functional as F
from enum import Enum

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

class OpCode(Enum):
    STOP = 0
    LAYER_NORM = 1
    MM = 2

def get_layernorm_inst():
    instructions = torch.zeros(SM_COUNT, 1, INST_SIZE, dtype=torch.int, device=device)
    instructions[0,0,0] = OpCode.LAYER_NORM.value
    return instructions

def get_mm_inst(m, n):
    
    # Compute the tiles and the number of instructions required
    tiles = list(itertools.product(range(m // (TILE_DIM * M_BLOCK)), range(n // (TILE_DIM * N_BLOCK))))
    num_inst = (len(tiles) + SM_COUNT - 1) // SM_COUNT
    
    # Allocate and fill in the instructions
    instructions = torch.zeros(SM_COUNT, num_inst, INST_SIZE, dtype=torch.int, device=device)
    for inst in range(num_inst):
        for i, tile in enumerate(tiles[inst * SM_COUNT: (inst + 1) * SM_COUNT]):
            instructions[i, inst, 0] = OpCode.MM.value
            instructions[i, inst, 1], instructions[i, inst, 2] = tile
    
    return instructions

def get_null_inst():
    return torch.zeros(SM_COUNT, 1, INST_SIZE, dtype=torch.int, device=device)

if __name__ == '__main__':
    
    seq_len = 256

    instructions = torch.cat(
        (
            get_layernorm_inst(), 
            get_mm_inst(seq_len, 3 * EMBED_DIM), 
            get_null_inst()
         ), 
        dim=1
    )

    layer_input = torch.rand(seq_len, EMBED_DIM, dtype=dtype, device=device)
    after_first_norm = torch.zeros(seq_len, EMBED_DIM, dtype=dtype, device=device)
    qkv_weights = torch.rand(EMBED_DIM, 3 * EMBED_DIM, dtype=dtype, device=device)
    qkv = torch.zeros(seq_len, 3 * EMBED_DIM, dtype=dtype, device=device)

    ref = F.layer_norm(layer_input, (EMBED_DIM, )) @ qkv_weights

    gpt2_decode.gpt2_decode(instructions, layer_input, after_first_norm, qkv_weights, qkv)
    
    print(after_first_norm)
    print(qkv)
    print(ref)
    print((qkv - ref).abs().max())
