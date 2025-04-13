
NVCC?=nvcc

TARGET=tk_ring_attention
SRC=tk_ring_attention.cu

THUNDERKITTENS_ROOT=$(shell pwd)/../../..

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas
NVCCFLAGS+= -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype $(shell python3 -m pybind11 --includes) $(shell python3-config --ldflags) -shared -fPIC -lpython3.11
NVCCFLAGS+= -DKITTENS_HOPPER -arch=sm_90a

torch:
	python3 setup.py build_ext --inplace --verbose
	mv tk_ring_attention.*so ../
	rm -rf build/

# If there is an error with libc10 not found, try:
#   import torch; print(torch.__path__[0])
# Say whatever was printed above was PATH, run:
#  export LD_LIBRARY_PATH=PATH/lib:$LD_LIBRARY_PATH
# For me it was:
#  export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH

# For quick compilation (didn't use "all" because I want "torch" to be the default)
build: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)$(shell python3-config --extension-suffix)

clean:
	rm -rf build/ ../*.so __pycache__/ $(TARGET)
