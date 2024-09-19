# GPU Selection: Set to either 4090 or H100
GPU_TARGET=4090

# Compiler
NVCC=nvcc

# Conditional setup based on the target GPU
ifeq ($(GPU_TARGET),4090)
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_4090 -arch=sm_89 -lcuda -lcudadevrt -lcudart_static -lcublas -I${THUNDERKITTENS_ROOT}
TARGET=fftconv_tk
SRC=4090/fftconv_tk.cu
else ifeq ($(GPU_TARGET),A100)
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_A100 -arch=sm_80 -lcuda -lcudadevrt -lcudart_static -lcublas -I${THUNDERKITTENS_ROOT} # A100
TARGET=fftconv_tk
SRC=4090/fftconv_tk.cu
else ifeq ($(GPU_TARGET),H100)
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3  -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_HOPPER -arch=sm_90a -lcuda -lcudadevrt -lcudart_static -lcublas -I${THUNDERKITTENS_ROOT} # H100
TARGET=fftconv_tk_h100
SRC=H100/fftconv_tk_h100.cu
endif

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)
