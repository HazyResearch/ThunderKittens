$(warning Warning: Linear attention kernel is currently broken. A new, better algorithm is coming soon!)
GPU := H100
SRC := linear_attention.cu
OUT := linear_attention.out
CMD := python gentests.py; ./linear_attention.out randn_1024.txt
CONFIG := standalone
include ../common.mk
