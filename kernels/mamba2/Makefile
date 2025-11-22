$(error Mamba2 kernel is currently broken and is in the process of being rewritten!)

GPU := H100

# PyTorch extension mode
SRC := mamba2.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python benchmark.py # or test_correctness.py
NVCCFLAGS := -DTORCH_COMPILE
CONFIG := pytorch

# Standalone mode
# SRC := mamba2.cu
# OUT := mamba2.out
# CMD := python gentests.py; ./mamba2.out randn_1024N_64D_64S.txt
# CONFIG := standalone

include ../common.mk
