# GPU Selection: 4090, A100, H100
GPU_TARGET=H100

# Compiler
NVCC?=nvcc
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE -Xcompiler -fopenmp --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_HOPPER -arch=sm_90a -lcuda -lcudadevrt -lcudart_static -lcublas -lgomp -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype # H100

DEBUG ?= 0
ifneq ($(DEBUG), 0)
    NVCCFLAGS += -G -g -dopt=on
endif

PYTHON?=0
SUFFIX=""
ifeq ($(PYTHON), 1)
    NVCCFLAGS+=$(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC -lpython3.11
    SUFFIX=$(shell python3-config --extension-suffix)
endif

TARGET=matmul
SRC=matmul.cu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(SUFFIX)

# Clean target
clean:
	rm -f $(TARGET)
