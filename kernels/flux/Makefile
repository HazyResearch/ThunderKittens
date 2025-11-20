GPU := H100

# Flux gate + PyTorch extension mode
SRC := flux_gate.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python benchmark.py
NVCCFLAGS := -DTORCH_COMPILE
CONFIG := pytorch

# Flux gate + standalone mode
# SRC := flux_gate.cu
# OUT := flux_gate.out
# CMD := ./flux_gate.out
# NVCCFLAGS := -fopenmp -lgomp
# CONFIG := standalone

# Flux gelu + PyTorch extension mode
# SRC := flux_gelu.cu
# OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# CMD := python benchmark.py
# NVCCFLAGS := -DTORCH_COMPILE
# CONFIG := pytorch

# Flux gelu + standalone mode
# SRC := flux_gelu.cu
# OUT := flux_gelu.out
# CMD := ./flux_gelu.out
# NVCCFLAGS := -fopenmp -lgomp
# CONFIG := standalone

include ../common.mk
