# ring-transformer

This is a minimal transformer training implementation using `ring_flash_attn_qkvpacked_func` of [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention).

To run it make sure torch is installed.

Run `pip install -r requirements.txt` or manually clone & install [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) (e.g. run `pip install .` where it was cloned).

The script `./run.sh` launches the script for 2 GPUs, e.g. `torchrun --nproc_per_node 2 main.py`.
