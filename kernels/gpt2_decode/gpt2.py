import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from gpt2_decode import gpt2_decode, OpCode
from gpt2_run import get_layernorm_inst, get_mm_inst, get_attn_inst, get_null_inst, EMBED_DIM, dtype, device

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', device=device)
model = GPT2LMHeadModel.from_pretrained('gpt2').to(dtype=dtype, device=device)

seq_len = 256

class CustomGPT2Block(nn.Module):
    
    def __init__(self, block):
        super(CustomGPT2Block, self).__init__()
        self.block = block
    
    def forward(self, hidden_states, *args, **kwargs):
        
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
        
        input_hidden = torch.zeros(seq_len, EMBED_DIM, dtype=dtype, device=device)
        input_hidden[:hidden_states.shape[1], :] = hidden_states.squeeze(0)
        input_residual = torch.zeros_like(input_hidden)
        gamma_first_norm = self.block.ln_1.weight.detach()
        beta_first_norm = self.block.ln_1.bias.detach()
        mid_residual = torch.empty(seq_len, EMBED_DIM, dtype=dtype, device=device)
        mid_first_norm = torch.empty(seq_len, EMBED_DIM, dtype=dtype, device=device)
        weight_qkv = self.block.attn.c_attn.weight.detach()
        bias_qkv = self.block.attn.c_attn.bias.detach()
        mid_qkv = torch.empty(seq_len, 3 * EMBED_DIM, dtype=dtype, device=device)
        mid_attn = torch.empty(seq_len, EMBED_DIM, dtype=dtype, device=device)
        weight_proj = self.block.attn.c_proj.weight.detach()
        bias_proj = self.block.attn.c_proj.bias.detach()
        mid_proj = torch.empty(seq_len, EMBED_DIM, dtype=dtype, device=device)
        gamma_second_norm = self.block.ln_2.weight.detach()
        beta_second_norm = self.block.ln_2.bias.detach()
        output_residual = torch.empty(seq_len, EMBED_DIM, dtype=dtype, device=device)
        mid_second_norm = torch.empty(seq_len, EMBED_DIM, dtype=dtype, device=device)
        weight_ff_expand = self.block.mlp.c_fc.weight.detach()
        bias_ff_expand = self.block.mlp.c_fc.bias.detach()
        mid_ff_expand = torch.empty(seq_len, 4 * EMBED_DIM, dtype=dtype, device=device)
        weight_ff_contract = self.block.mlp.c_proj.weight.detach()
        bias_ff_contract = self.block.mlp.c_proj.bias.detach()
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
        
        output_hidden += output_residual
        
        my_output = (output_hidden[:hidden_states.shape[1], :].unsqueeze(0),)
        
        expected_output = self.block(hidden_states, *args, **kwargs)
        
        return my_output

for i in range(12):
    model.transformer.h[i] = CustomGPT2Block(model.transformer.h[i])

text = "A kitten struck by thunder is known as a"
encoded_input = tokenizer(text, return_tensors='pt').input_ids.to(device)
print(tokenizer.decode(model.generate(encoded_input, use_cache=False)[0].tolist()))
