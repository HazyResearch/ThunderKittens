# Change this or override it on the command line:
#   make LEVEL=04 run
LEVEL ?= 01

ARCH := SM120
SRC := level_$(LEVEL).cu
OUT := level_$(LEVEL).out
CMD := ./level_$(LEVEL).out
CONFIG := standalone

ifeq ($(LEVEL),00)
NVCCFLAGS += -lcublas
endif

include ../../common.mk
