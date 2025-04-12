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

clean:
	rm -rf build/ ../*.so __pycache__/
