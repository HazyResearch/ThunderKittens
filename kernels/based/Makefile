# GPU Selection: 4090, H100
GPU_TARGET=4090

# Compiler
NVCC=nvcc

ifeq ($(GPU_TARGET),H100)
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE -Xcompiler -fopenmp --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_HOPPER -arch=sm_90a -lcuda -lcudadevrt -lcudart_static -lgomp -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype # H100
TARGET=based
SRC=lin_attn_h100.cu
else ifeq ($(GPU_TARGET),4090)
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE -Xcompiler -fopenmp --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_4090 -arch=sm_89 -lcuda -lcudadevrt -lcudart_static -lgomp -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype # 4090
TARGET=based
SRC=lin_attn_4090.cu
endif

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)

