# Compiler
NVCC=nvcc

# Compiler flags
NVCCFLAGS=-g -G -std=c++20 --expt-relaxed-constexpr -arch=sm_90a

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