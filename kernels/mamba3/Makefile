GPU := H100


# PyTorch extension mode
SRC := mamba3.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python benchmark.py
NVCCFLAGS := -DTORCH_COMPILE
CONFIG := pytorch

# Standalone mode
# SRC := mamba3.cu
# OUT := mamba3.out
# CMD := python gentests.py; ./mamba3.out randn_1024N_64D_64S.txt
# CONFIG := standalone

include ../common.mk
