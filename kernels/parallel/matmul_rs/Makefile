ifeq ($(GPU),B200)
SRC := matmul_rs_b200.cu
else ifeq ($(GPU),H100)
SRC := matmul_rs_h100.cu
else
$(error Unsupported GPU: $(GPU). Please set GPU to B200 or H100.)
endif

include ../common.mk
