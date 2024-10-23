
CONFIG_DIR='/home/bfs/simran/attention/lolcats/configs/'   # update to your path

CUDA_VISIBLE_DEVICES=0 python -Wignore demo_llama_hf.py \
    --model_config_path ${CONFIG_DIR}/model/distill_llama3_1_8b_lk_smd_wtk64_fd64_w01.yaml \
    --num_generations 1 \
    --max_new_tokens 50  \
    --model_type tk_attention 
    


