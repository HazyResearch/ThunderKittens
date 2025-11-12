ifeq ($(GPU),B200)
$(error B200 is unsupported.)
else ifeq ($(GPU),H100)
SRC := moe_dispatch_matmul_h100.cu
else
$(error Unsupported GPU: $(GPU). Please set GPU to B200 or H100.)
endif

include ../common.mk
