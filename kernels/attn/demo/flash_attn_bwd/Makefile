# GPU Selection: 4090, A100, H100
GPU_TARGET=H100

PROJ_ROOT=${PWD}
THUNDERKITTENS_ROOT=$(PROJ_ROOT)

PYTORCH_ROOT := $(shell python -c "import torch; import os; print(os.path.dirname(torch.__file__))")

# Compiler
NVCC?=nvcc

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE -Xcompiler -fopenmp --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_HOPPER -arch=sm_90a \
         -I/usr/local/cuda/include \
         -I${THUNDERKITTENS_ROOT}/include \
         -I${THUNDERKITTENS_ROOT}/prototype \
         -I$(PYTORCH_ROOT)/include \
         -I$(PYTORCH_ROOT)/include/torch/csrc/api/include \
         -I$(PYTORCH_ROOT)/include/ATen \
         -I$(PYTORCH_ROOT)/include/ATen/cuda \
         -L$(PYTORCH_ROOT)/lib \
         -Wl,-rpath,$(PYTORCH_ROOT)/lib \
         -ltorch \
         -ltorch_cuda \
         -ltorch_cpu \
         -lc10 \
         -lc10_cuda \
         -lcuda \
         -lcudart_static \
         -lcublas \
         -lcudadevrt \
         -lgomp

TARGET=bench_h100_lcsf_bwd
SRC=bench_h100_lcsf_bwd.cu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p build
	$(NVCC) $(SRC) $(NVCCFLAGS) -o build/$(TARGET)

# Clean target
clean:
	rm -f build/$(TARGET)
	rm -rf build/*.o
