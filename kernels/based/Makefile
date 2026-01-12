GPU := H100

# PyTorch extension mode
SRC := linear_attn.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python benchmark.py
NVCCFLAGS := -DTORCH_COMPILE
CONFIG := pytorch

# Standalone mode
# SRC := linear_attn.cu
# OUT := linear_attn.out
# CMD := python gentests.py randn_all; ./linear_attn.out randn_all_1x1x1024x16x64.txt
# CONFIG := standalone

include ../common.mk
