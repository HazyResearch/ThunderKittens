GPU_TARGET=B100

# Compiler
NVCC=nvcc
PYTHON_VERSION=3.10


# Conditional setup based on the target GPU
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -I../../../include -I../../../prototype -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_BLACKWELL -DKITTENS_HOPPER -arch=sm_100a -lcuda -lcudadevrt -lcudart_static -lineinfo $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC -lpython$(PYTHON_VERSION)

TARGET=b200_attn
SRC=b200.cu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(shell python3-config --extension-suffix)

# Clean target
clean:
	rm -f $(TARGET)$(shell python3-config --extension-suffix)