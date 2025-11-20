# Change this!
LEVEL := 01

###############################################
###############################################
###############################################

GPU := H100
SRC := level_$(LEVEL).cu
OUT := level_$(LEVEL).out
CMD := ./level_$(LEVEL).out
CONFIG := standalone
NVCCFLAGS := -fopenmp -lgomp
include ../../common.mk
