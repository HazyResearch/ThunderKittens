torch:
	python3 setup.py build_ext --inplace --verbose
	mv tk_ring_attention.*so ../
	rm -rf build/

clean:
	rm -rf build/ ../*.so __pycache__/
