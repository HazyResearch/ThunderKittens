# GPU Selection: 4090, A100, H100
GPU_TARGET=H100

PURPOSE=inference
# PURPOSE=training

# Compiler
NVCC=nvcc

# Conditional setup based on the target GPU
NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_HOPPER -arch=sm_90a -lcuda -lcudadevrt -lcudart_static -lcublas # H100
	ifeq ($(PURPOSE),inference)
	TARGET=attn # H100
	SRC=h100_fwd_c.cu # H100
	else ifeq ($(PURPOSE),training)
	TARGET=attn # H100
	SRC=h100_train_c.cu # H100
	endif

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)