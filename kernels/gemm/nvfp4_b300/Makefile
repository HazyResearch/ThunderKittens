ARCH := SM103

# Standalone mode
SRC := nvfp4_b300_gemm.cu
# SRC := nvfp4_b300_gemm_optimized.cu
OUT := nvfp4_b300_gemm.out
CMD := ./nvfp4_b300_gemm.out
CONFIG := standalone

# PyTorch extension mode
# This is only to demonstrate binding to PyTorch and does not reflect actual performance.
# SRC := nvfp4_b300_gemm.cu
# OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# CMD := python test_gemm.py # or test_quantize.py
# NVCCFLAGS := -DTORCH_COMPILE
# CONFIG := pytorch

include ../../common.mk
