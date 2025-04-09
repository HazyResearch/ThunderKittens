# Evaluation of Triton Flash-Attention Implementations

## Files from Dao-AILab / flash-attention

The following files were copied form Tri Dao's flash-attention repository to keep this repo self-contained:

- flash_attn_triton_og.py copied from [06-fused-attention.py](https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py)
- flash_attn_triton.py
  - applied fix for "dot() got an unexpected keyword argument 'trans_b'" found in [flash-attention#PR232](https://github.com/Dao-AILab/flash-attention/pull/232/files)

