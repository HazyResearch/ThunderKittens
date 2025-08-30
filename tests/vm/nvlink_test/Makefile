NVCC?=nvcc

NVCCFLAGS=-DNDEBUG -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing \
		  --use_fast_math -I../../../include -I../../../prototype -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose \
		  -Xptxas=--warn-on-spills -std=c++20 -MD -MT -MF -x cu -lrt -lpthread -ldl -DKITTENS_HOPPER -arch=sm_90a -lcuda -lcudadevrt -lcudart_static -lcublas \
		  -Xptxas -dlcm=cg # Disable L1 caching ('cv' for no caching at all)

TARGET=nvlink_test
SRC=nvlink_test.cu

# Default target
all: $(TARGET)

run: $(TARGET)
	./$(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(TARGET)

# Clean target
clean:
	rm -f $(TARGET)
