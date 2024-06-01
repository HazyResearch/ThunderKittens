# This is the commands for the pytorch jit...
# https://pytorch.org/tutorials/advanced/cpp_extension.html
# This is the commands for the pytorch jit...
# https://pytorch.org/tutorials/advanced/cpp_extension.html
import torch

import test_build_utils as tbu
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

name      = "tk_batch_add"
debug     = False
if(debug): print(f"WARNING DEBUG IS TRUE")
cuda_ext  = tbu.cuda_extension(name, debug, 'A100')
setup(name=f"{name}", 
      ext_modules=[cuda_ext],
      cmdclass={'build_ext': BuildExtension})
if(debug): print(f"WARNING DEBUG IS TRUE")