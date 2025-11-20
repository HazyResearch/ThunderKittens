# You must set the environment variable GPU
# export GPU=H100 | B200

ifeq ($(GPU),B200)
$(error B200 is unsupported.)
else ifeq ($(GPU),H100)
SRC := ag_gemm_h100.cu
else
$(error Unsupported GPU: $(GPU). Please set GPU to B200 or H100.)
endif

OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 benchmark.py
CONFIG := pytorch

include ../../common.mk
