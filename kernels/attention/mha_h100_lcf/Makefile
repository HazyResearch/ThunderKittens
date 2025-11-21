GPU := H100
SRC := mha_h100_lcf.cu
OUT := mha_h100_lcf.out
CMD := python gentests.py randn; ./mha_h100_lcf.out randn_924_128.txt
CONFIG := standalone
include ../../common.mk
