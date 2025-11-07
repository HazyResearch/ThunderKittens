# Example usage:
#   SRC := all_to_all.cu
#   GPU := H100
#   include common.mk

THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))
COMMON_DIR := $(dir $(THIS_MAKEFILE))
THUNDERKITTENS_ROOT := $(abspath $(COMMON_DIR)/../..)

NVCC := nvcc

# These take a while to load. You can bypass them by defining them as environment variables in advance
PYTHON_VERSION ?= $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))")
PYTHON_INCLUDES ?= $(shell python3 -c "import sysconfig; print('-I', sysconfig.get_path('include'), sep='')")
PYBIND_INCLUDES ?= $(shell python3 -m pybind11 --includes)
PYTORCH_INCLUDES ?= $(shell python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
PYTHON_LIBDIR ?= $(shell python3 -c "import sysconfig; print('-L', sysconfig.get_config_var('LIBDIR'), sep='')")
PYTORCH_LIBDIR ?= $(shell python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))")

NVCCFLAGS := -DNDEBUG -lineinfo
NVCCFLAGS += --expt-extended-lambda --expt-relaxed-constexpr 
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing 
NVCCFLAGS += -forward-unknown-to-host-compiler -ftemplate-backtrace-limit=0
NVCCFLAGS += -std=c++20 -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static
NVCCFLAGS += -O3 --use_fast_math # VERY important to include this flag
NVCCFLAGS += -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills 
NVCCFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype
NVCCFLAGS += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
NVCCFLAGS += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
NVCCFLAGS += -DTORCH_API_INCLUDE_EXTENSION_H
NVCCFLAGS += $(PYTHON_INCLUDES) $(PYTORCH_INCLUDES) $(PYBIND_INCLUDES)
NVCCFLAGS += ${PYTHON_LIBDIR} ${PYTORCH_LIBDIR} -lpython${PYTHON_VERSION}
NVCCFLAGS += -ltorch_python -ltorch_cuda -ltorch_cpu -ltorch -lc10_cuda -lc10
NVCCFLAGS += -diag-suppress 3189
NVCCFLAGS += -shared -fPIC
NVCCFLAGS += -DTORCH_EXTENSION_NAME=_C

# Expect the including Makefile to set SRC and GPU
GPU ?= NOT_SET
SRC ?= NOT_SET
OUT ?= _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
SCRIPT ?= benchmark.py
RUN_CMD ?= OMP_NUM_THREADS=1 torchrun --nproc_per_node=8 $(SCRIPT)

ifeq ($(GPU),B200)
NVCCFLAGS += -DKITTENS_HOPPER -DKITTENS_BLACKWELL -gencode arch=compute_100a,code=sm_100a
else ifeq ($(GPU),H100)
NVCCFLAGS += -DKITTENS_HOPPER -gencode arch=compute_90a,code=sm_90a
else
$(error Unsupported GPU: $(GPU). Please set GPU to B200 or H100.)
endif

all: $(OUT)

run: $(OUT)
	$(RUN_CMD)

bench:
	$(RUN_CMD)

ncu: $(OUT)
	ncu \
		--replay-mode kernel \
		--set full -f --export ./$(basename $(SRC)).ncu-rep \
		$(RUN_CMD)

nsys: $(OUT)
	nsys profile \
		--stats=true \
		--trace cuda,osrt,nvtx,python-gil,syscall \
		--gpu-metrics-devices=all \
		--cuda-memory-usage true \
		--force-overwrite=true \
		-o ./$(basename $(SRC)).nsys-rep \
		$(RUN_CMD)

$(OUT): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(OUT)

clean:
	rm -f $(OUT)

.PHONY: all run bench clean
