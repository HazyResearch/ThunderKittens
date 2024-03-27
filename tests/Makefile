# GPU Selection: Set to either 4090 or H100
GPU_TARGET=4090
COMP_LEVEL=fast

# Compiler
NVCC=nvcc
NVCCFLAGS=-std=c++20 --expt-relaxed-constexpr -Itesting_commons -I../src

# At what level to run tests?
NVCCFLAGS+= -DTEST_INTENSITY=2 # this is still on the testing todo list
# Which tests to run?
NVCCFLAGS+= -DTEST_ALL # Run all tests. But be warned, this will take a year to link.
# NVCCFLAGS+= -DTEST_ALL_WARP_REGISTER # Or look in testing_commons/flags.cuh for specific tests to significantly reduce compilation time.

ifeq ($(COMP_LEVEL),fast)
NVCCFLAGS+= -O0 --threads=0
else ifeq ($(COMP_LEVEL),debug)
NVCCFLAGS+= -g -G
else ifeq ($(COMP_LEVEL),profile)
NVCCFLAGS+= -O3 --threads=0 --dlto
endif

# Compiler flags based on GPU target
ifeq ($(GPU_TARGET),4090)
NVCCFLAGS+= -arch=sm_89 -DKITTENS_4090
else ifeq ($(GPU_TARGET),A100)
NVCCFLAGS+= -arch=sm_80 -DKITTENS_A100
else ifeq ($(GPU_TARGET),H100)
NVCCFLAGS+= -arch=sm_90a -DKITTENS_HOPPER
endif

# Target binary name
TARGET=unit_tests
BUILD_DIR=build

# Test source files
TESTS_SRC=$(shell find . -name '*.cu')

# Object files
OBJS=$(patsubst %.cu,$(BUILD_DIR)/%.o,$(TESTS_SRC))

# Default target
all: build $(TARGET)

# Create the build directory
build:
	mkdir -p $(BUILD_DIR)

# Rule to compile each CU file
$(BUILD_DIR)/%.o: %.cu
	mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Link object files to create the final executable
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $^ -o $(TARGET)

# Run target
run: all
	./$(TARGET)

# Clean target
clean:
	rm -rf $(BUILD_DIR) $(TARGET)