GPU := H100
SRC := layernorm.cu

# PyTorch extension mode
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python benchmark.py # or test_correctness.py
NVCCFLAGS := -DTORCH_COMPILE
CONFIG := pytorch

# Harness mode
# OUT := layernorm.out
# CMD := python gentests.py randn; ./layernorm.out randn.txt
# CONFIG := standalone

include ../common.mk
