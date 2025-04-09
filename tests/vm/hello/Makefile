# Compiler
NVCC?=nvcc
TARGET_ARCH?=sm_100a

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -I../../../include -I../../../prototype -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_HOPPER -DKITTENS_BLACKWELL -arch=$(TARGET_ARCH) -lcuda -lcudadevrt -lcudart_static -lcublas -I$(THUNDERKITTENS_ROOT)/include -I$(THUNDERKITTENS_ROOT)/prototype -lineinfo
TARGET=hello
SRC=hello.cu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)