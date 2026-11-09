# You must set the environment variable ARCH
# export ARCH=SM90 | SM100 | SM103
# Optional: NUM_DEVICES=<n> (default 8) builds and runs for n GPUs

NUM_DEVICES ?= 8
NVCCFLAGS += -DTK_NUM_DEVICES=$(NUM_DEVICES)

ifneq ($(filter $(ARCH),SM100 SM103),)
SRC := gemm_rs_b200.cu
else ifeq ($(ARCH),SM90)
SRC := gemm_rs_h100.cu
else
$(error Unsupported ARCH: $(ARCH). Please set ARCH to SM90, SM100 or SM103.)
endif

OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := OMP_NUM_THREADS=1 torchrun --nproc_per_node=$(NUM_DEVICES) benchmark.py
CONFIG := pytorch

include ../../common.mk
