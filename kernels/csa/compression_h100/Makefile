ARCH := SM90
SRC := correctness.cu
OUT := correctness.out
CMD := python gentests.py randn; ./correctness.out randn_compression_test.txt
CONFIG := standalone
include ../../common.mk

# Timing sweep (NUM_WORKERS x C_CHUNK x INPUT_PIPE_STAGES) -- see benchmark.cu.
benchmark.out: benchmark.cu
	$(NVCC) benchmark.cu $(NVCCFLAGS) -o benchmark.out

benchmark: benchmark.out
	./benchmark.out

.PHONY: benchmark
