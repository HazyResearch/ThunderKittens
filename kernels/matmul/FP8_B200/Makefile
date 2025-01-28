# GPU Selection: 4090, A100, H100
GPU_TARGET=B200

# Compiler
NVCC=nvcc

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE -Xcompiler -fopenmp --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_BLACKWELL -DKITTENS_HOPPER -arch=sm_100a -lcuda -lcudadevrt -lcudart_static -lcublas -lgomp -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype
TARGET=matmul
SRC=matmul.cu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)