
CONFIG_DIR='../configs/'   

# Using huggingface checkpoints 
CUDA_VISIBLE_DEVICES=0 python -Wignore demo_lolcats_hf.py \
    --model_config_path ${CONFIG_DIR}/llama_3.1_8b_model_config.yaml \
    --distill_config_path ${CONFIG_DIR}/llama_3.1_8b_distill_config.yaml \
    --finetune_config_path ${CONFIG_DIR}/llama_3.1_8b_finetune_config.yaml \
    --attn_mlp_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-distill' \
    --finetune_checkpoint_path 'hazyresearch/lolcats-llama-3.1-8b-ft-lora' \
    --num_generations 1 \
    --max_new_tokens 50 \
    --use_cuda_kernels 1 \
    --benchmark

