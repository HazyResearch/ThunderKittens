# GPU Selection: Set to either 4090 or H100
GPU_TARGET=4090

# Compiler
NVCC=nvcc

# Compiler flags
ifeq ($(GPU_TARGET),4090)
NVCCFLAGS=-g -G -std=c++20 --expt-relaxed-constexpr -arch=sm_89 -DKITTENS_4090 -lcuda -lcudart # for 4090
else ifeq ($(GPU_TARGET),H100)
NVCCFLAGS=-g -G -std=c++20 --expt-relaxed-constexpr -arch=sm_90a -DKITTENS_HOPPER -lcuda -lcudart
endif

# Target binary name
TARGET=unit_tests

# Source files
SRC=unit_tests.cu

# Default target
all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Run target
run: all
	./$(TARGET)

# Clean target
clean:
	rm -f $(TARGET)