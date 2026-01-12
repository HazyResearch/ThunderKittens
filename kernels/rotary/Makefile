GPU := H100

# PyTorch extension mode
SRC := rotary.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python benchmark.py # or test_correctness.py
NVCCFLAGS := -DTORCH_COMPILE
CONFIG := pytorch

# Standalone mode
# SRC := rotary.cu
# OUT := rotary.out
# CMD := python gentests.py randn; ./rotary.out randn.txt
# CONFIG := standalone

include ../common.mk
