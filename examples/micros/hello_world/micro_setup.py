# This is the commands for the pytorch jit...
# https://pytorch.org/tutorials/advanced/cpp_extension.html
# This is the commands for the pytorch jit...
# https://pytorch.org/tutorials/advanced/cpp_extension.html
import torch

import test_build_utils as tbu
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

name      = "micro"
debug     = False 
if(debug): print(f"WARNING DEBUG IS TRUE")
cuda_ext  = tbu.cuda_extension(name, debug)
setup(name=f"{name}", 
      ext_modules=[cuda_ext], 
      cmdclass={'build_ext': BuildExtension})
if(debug): print(f"WARNING DEBUG IS TRUE")