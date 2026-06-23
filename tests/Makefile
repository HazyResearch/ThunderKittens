# NOTE: WE HIGHLY RECOMMEND RUNNING WITH `make -j32` or however many threads your machine has.
# If you want to run lots of tests without parallel compilation, be prepared
# to spend a *long* time compiling. (like, HOURS single-threaded.)

# Architecture Selection: Set to SM80, SM90, SM100, SM103, SM120, or SM121 (GB10)
ARCH=SM90
COMP_LEVEL=fast

# Compiler
NVCC?=nvcc
NVCCFLAGS=-std=c++20 --expt-relaxed-constexpr -Itesting_commons -I../include --extended-lambda

# At what level to run tests?
# Level 1 takes 15 seconds to compile 1,000 tests.
# (Default) Level 2 takes a minute to compile 3,000 tests.
# Level 3 takes 5 minutes to compile 10,000 tests.
# Level 4 takes 15 while to compile 25,000 tests
NVCCFLAGS+= -DTEST_INTENSITY=2
# Which tests to run?
# -DTEST_ALL means run all tests.
# You can also specify subsections, e.g. -DTEST_WARP_MEMORY
# Or individual tests, like -DTEST_WARP_MEMORY_VEC_DSMEM. Useful for debugging!
NVCCFLAGS+= -DTEST_ALL -DTEST_ALL_MULTI_GPU
# NVCCFLAGS+= -DTEST_ALL_WARP_MEMORY

ifeq ($(COMP_LEVEL),fast)
NVCCFLAGS+= -O0 --threads=0 -diag-suppress 20054
else ifeq ($(COMP_LEVEL),debug)
NVCCFLAGS+= -g -G
else ifeq ($(COMP_LEVEL),profile)
NVCCFLAGS+= -O3 --threads=0 -diag-suppress 20054 --dlto
endif

# Compiler flags based on architecture target
ifeq ($(ARCH),SM80)
NVCCFLAGS+= -gencode arch=compute_80a,code=sm_80 -DKITTENS_SM80 -lcuda -lcudart
else ifeq ($(ARCH),SM90)
NVCCFLAGS+= -gencode arch=compute_90a,code=sm_90a -DKITTENS_SM90 -lcuda -lcudart
else ifeq ($(ARCH),SM100)
NVCCFLAGS+= -gencode arch=compute_100a,code=sm_100a -DKITTENS_SM100 -lcuda -lcudart
else ifeq ($(ARCH),SM103)
NVCCFLAGS+= -gencode arch=compute_103a,code=sm_103a -DKITTENS_SM103 -lcuda -lcudart
else ifeq ($(ARCH),SM121)
# GB10 (DGX Spark): SM120 feature set, native sm_121a SASS (sm_120a won't load on sm_121).
# GB10 is single-GPU, so skip the multi-GPU (PGL) tests.
NVCCFLAGS+= -gencode arch=compute_121a,code=sm_121a -DKITTENS_SM120 -DNUM_GPUS=1 -lcuda -lcudart
else ifeq ($(ARCH),SM120)
NVCCFLAGS+= -gencode arch=compute_120a,code=sm_120a -DKITTENS_SM120 -lcuda -lcudart
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