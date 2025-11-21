GPU := H100

# Producer-consumer kernel + PyTorch extension mode
SRC := fftconv_pc.cu
OUT := _C$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CMD := python benchmark.py # or test_correctness.py
NVCCFLAGS := -DTORCH_COMPILE
CONFIG := pytorch

# Producer-consumer kernel + standalone mode
# SRC := fftconv_pc.cu
# OUT := fftconv_pc.out
# CMD := python gentests.py randn; ./fftconv_pc.out randn_f4096_H1024_B4.txt
# CONFIG := standalone

# Non-producer-consumer kernel + standalone mode (PyTorch extension mode does not exist for this kernel)
# SRC := fftconv_non_pc.cu
# OUT := fftconv_non_pc.out
# CMD := python gentests.py randn; ./fftconv_non_pc.out randn_f4096_H1024_B4.txt
# CONFIG := standalone

include ../common.mk
