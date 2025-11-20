# H100 and B200 are both supported
$(warning Warning: Linear attention kernel is currently broken.)
GPU := H100
SRC := linear_attn.cu
OUT := linear_attn.out
CMD := python gentests.py; ./linear_attn.out randn_1024.txt
CONFIG := standalone
include ../common.mk
