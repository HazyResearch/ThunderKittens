# You must set the environment variable GPU
# export GPU=H100 | B200

SRC := ulysses_attn.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 benchmark.py
CONFIG := pytorch

include ../../common.mk
