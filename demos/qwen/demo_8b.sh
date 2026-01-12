

CONFIG_DIR='../configs/'

CUDA_VISIBLE_DEVICES=0 python -Wignore demo_qwen_hf.py \
    --model_config_path ${CONFIG_DIR}/llama_3.1_8b_model_config.yaml \
    --num_generations 1 \
    --max_new_tokens 50  \
    --model_type tk_attention # options are: [tk_attention, eager, sdpa] 


