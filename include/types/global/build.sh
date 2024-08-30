#!/bin/bash
nvcc test.cu -arch=sm_89 -std=c++20 --expt-relaxed-constexpr -o test -lcuda -lcudadevrt -lcudart_static