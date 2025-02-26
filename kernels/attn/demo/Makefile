# Compiler
NVCC?=nvcc
GPU=H100


ifeq ($(GPU),4090)
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -I../../../include -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_4090 -arch=sm_89 -lcuda -lcudadevrt -lcudart_static -lcublas
TARGET=attn_fwd
SRC=4090.cu
else ifeq ($(GPU),H100)
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -I../../../include -I../../../prototype -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_HOPPER -arch=sm_90a -lcuda -lcudadevrt -lcudart_static -lcublas
TARGET=attn_fwd
SRC=h100_lcf.cu
endif

TARGET_PYBIND=attn_fwd_pybind
PYBIND_FLAGS=$(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC -lpython3.12
SRC_PYBIND=h100_lcf_pybind.cu

TARGET_CAUSAL=attn_fwd_causal
SRC_CAUSAL=h100_lcf_causal.cu

# Default target
all: $(TARGET) $(TARGET_PYBIND) $(TARGET_CAUSAL)

pybind: $(TARGET_PYBIND)

causal: $(TARGET_CAUSAL)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

$(TARGET_CAUSAL): $(SRC_CAUSAL)
	$(NVCC) $(SRC_CAUSAL) $(NVCCFLAGS) -o $(TARGET_CAUSAL)

$(TARGET_PYBIND): $(SRC_PYBIND)
	$(NVCC) $(SRC_PYBIND) $(NVCCFLAGS) $(PYBIND_FLAGS) -o $(TARGET_PYBIND)$(shell python3-config --extension-suffix)

# Clean target
clean:
	rm -f $(TARGET) $(TARGET_PYBIND)$(shell python3-config --extension-suffix) $(TARGET_CAUSAL)