# Compiler
NVCC=nvcc

# Compiler flags
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE -DKITTENS_HOPPER --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -arch=sm_90a -lcuda -lcudadevrt -lcudart_static -lcublas

# Target binary name
TARGET=naive

# Source files
SRC=naive.cu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)