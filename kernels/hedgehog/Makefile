$(error Hedgehog kernel is currently broken and is in the process of being rewritten!)

GPU := H100

# PyTorch extension mode
SRC := hedgehog.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python benchmark.py # or test_correctness.py
NVCCFLAGS := -DTORCH_COMPILE
CONFIG := pytorch

# Standalone mode
# SRC := hedgehog.cu
# OUT := hedgehog.out
# CMD := python gentests.py randn; ./hedgehog.out randn.txt
# CONFIG := standalone

include ../common.mk
