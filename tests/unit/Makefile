# NOTE: WE HIGHLY RECOMMEND RUNNING WITH `make -j32` or however many threads your machine has.
# If you want to run lots of tests without parallel compilation, be prepared
# to spend a *long* time compiling. (like, HOURS single-threaded.)

# GPU Selection: Set to either 4090 or H100
GPU_TARGET=H100
COMP_LEVEL=fast

# Compiler
NVCC=nvcc
NVCCFLAGS=-std=c++20 --expt-relaxed-constexpr -Itesting_commons -I../../include

# At what level to run tests?
# Level 1 takes a 15 seconds to compile 1,000 tests.
# (Default) Level 2 takes a minute to compile 3,000 tests.
# Level 3 takes 5 minutes to compile 10,000 tests.
# Level 4 takes 15 while to compile 25,000 tests
NVCCFLAGS+= -DTEST_INTENSITY=2
# Which tests to run?
# -DTEST_ALL means run all tests.
# You can also specify subsections, e.g. -DTEST_WARP_MEMORY
# Or individual tests, like -DTEST_WARP_MEMORY_VEC_DSMEM. Useful for debugging!
# NVCCFLAGS+= -DTEST_ALL_WARP_MEMORY -DTEST_ALL_GROUP_MEMORY
# NVCCFLAGS+= -DTEST_ALL_WARP_REGISTER_VEC
# NVCCFLAGS+= -DTEST_WARP_MEMORY_TILE_GLOBAL_TO_SHARED
# NVCCFLAGS+= -DTEST_WARP_REGISTER_TILE_MMA_COMPLEX
# NVCCFLAGS+= -DTEST_WARP_REGISTER_TILE_MUL_COMPLEX
# NVCCFLAGS+= -DTEST_WARP_REGISTER_TILE_MAPS_COMPLEX
NVCCFLAGS+= -DTEST_ALL_GROUP_WGMMA

ifeq ($(COMP_LEVEL),fast)
NVCCFLAGS+= -O0 --threads=0 -diag-suppress 20054
else ifeq ($(COMP_LEVEL),debug)
NVCCFLAGS+= -g -G
else ifeq ($(COMP_LEVEL),profile)
NVCCFLAGS+= -O3 --threads=0 -diag-suppress 20054 --dlto
endif

# Compiler flags based on GPU target
ifeq ($(GPU_TARGET),4090)
NVCCFLAGS+= -arch=sm_89 -DKITTENS_4090
else ifeq ($(GPU_TARGET),A100)
NVCCFLAGS+= -arch=sm_80 -DKITTENS_A100
else ifeq ($(GPU_TARGET),H100)
NVCCFLAGS+= -arch=sm_90a -DKITTENS_HOPPER -lcuda -lcudart
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