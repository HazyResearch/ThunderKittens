GPU := H100

# PyTorch extension mode
SRC := mha_h100.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python benchmark.py # or test_correctness.py
NVCCFLAGS := -DTORCH_COMPILE
CONFIG := pytorch

# Standalone mode
# SRC := mha_h100.cu
# OUT := mha_h100.out
# CMD := python gentests.py; ./mha_h100.out randn_inputs.txt
# CONFIG := standalone

include ../../common.mk
