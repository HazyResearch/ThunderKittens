ifeq ($(GPU),B200)
$(error B200 is unsupported.)
else ifeq ($(GPU),H100)
SRC := matmul_ar_h100.cu # change this to compile the LCSC version
else
$(error Unsupported GPU: $(GPU). Please set GPU to B200 or H100.)
endif

include ../common.mk
