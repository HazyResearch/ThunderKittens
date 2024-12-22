GPU_TARGET=H100 # FYI

# Compiler
NVCC=nvcc

# Conditional setup based on the target GPU
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -I../../include -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_HOPPER -arch=sm_90a -lcuda -lcudadevrt -lcudart_static # H100

TARGET=cylon
SRC=cylon.cu

# # Python and Torch include paths
# PYTHON_INCLUDE := $(shell python -c "import sysconfig; print(sysconfig.get_path('include'))")
# TORCH_INCLUDE := $(shell python -c "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
# # Add Python and Torch include paths to NVCCFLAGS
# NVCCFLAGS += -I$(PYTHON_INCLUDE) $(TORCH_INCLUDE)

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)
