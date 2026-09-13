ARCH := SM90
SRC := rotary.cu

# PyTorch extension mode
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python benchmark.py # or test_correctness.py
NVCCFLAGS := -DTORCH_COMPILE
CONFIG := pytorch

# tvm-ffi mode
# OUT := rotary_tvm_ffi.so
# CMD := python test_tvm_ffi.py
# NVCCFLAGS := -DTVM_FFI_COMPILE
# CONFIG := tvm_ffi

# Standalone mode
# OUT := rotary.out
# CMD := python gentests.py randn; ./rotary.out randn.txt
# CONFIG := standalone

include ../common.mk
