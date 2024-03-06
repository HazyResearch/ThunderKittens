# Compiler
NVCC=nvcc

# Compiler flags
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++17 -MD -MT -MF -x cu -lrt -lpthread -ldl -arch=sm_70 -lcuda -lcudadevrt -lcudart_static -lcublas -I/home/ubuntu/thunderKittens/src

# Target binary name
TARGET=tictoc_ker

# Source files
SRC=tictoc_ker.cu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)
