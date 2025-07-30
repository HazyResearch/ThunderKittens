# PGL is only supported on Hopper and Blackwell
GPU := H100

# Compiler
NVCC := nvcc

# NVCC flags
THUNDERKITTENS_ROOT := ../../
NVCCFLAGS := -DNDEBUG -Xcompiler=-fPIE -Xcompiler -fopenmp --expt-extended-lambda --expt-relaxed-constexpr 
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler 
NVCCFLAGS += -O3  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt 
NVCCFLAGS += -lpthread -ldl -DKITTENS_HOPPER -lcuda -lcudadevrt -lcudart_static -lcublas -lgomp 
NVCCFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype

# Architecture-specific flags
ifeq ($(GPU), H100)
	NVCCFLAGS += -DKITTENS_HOPPER -gencode arch=compute_90a,code=sm_90a
else ifeq ($(GPU), B200)
	NVCCFLAGS += -DKITTENS_HOPPER -DKITTENS_BLACKWELL -gencode arch=compute_100a,code=sm_100a
endif

# Target
SRC := all_reduce.cu
TARGET := all_reduce

all: $(TARGET)

run: $(TARGET)
	./$(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)
