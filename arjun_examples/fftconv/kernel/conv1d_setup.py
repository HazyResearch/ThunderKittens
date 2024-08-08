# This is the commands for the pytorch jit...
# https://pytorch.org/tutorials/advanced/cpp_extension.html
import torch
name = "fftconv_tk"
gpu = 'H100'
assert(gpu in ['4090', 'A100', 'H100'])

import test_build_utils as tbu
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
debug     = False
if(debug): print(f"WARNING DEBUG IS TRUE")
cuda_ext  = tbu.cuda_extension(name, debug, gpu)
setup(name=f"{name}", 
      ext_modules=[cuda_ext], 
      cmdclass={'build_ext': BuildExtension})
if(debug): print(f"WARNING DEBUG IS TRUE")