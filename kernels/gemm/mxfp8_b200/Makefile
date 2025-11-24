GPU := B200
SRC := mxfp8_b200_gemm.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python test_quantize.py
CONFIG := pytorch
include ../../common.mk
