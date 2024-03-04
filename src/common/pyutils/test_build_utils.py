import torch
from torch.utils.cpp_extension import load
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
tile = 16
##################
#
# Extension help
#
# This is the commands for the pytorch jit...
# https://pytorch.org/tutorials/advanced/cpp_extension.html
import os
project_root = os.getenv("CUDATASTIC_ROOT")
if project_root is None:
    print("There is no project root set (env: cudatastic_root) did you run env.src?")
    os._exit(-1)   

def _sources(name): return [f"{name}_frontend.cpp", f"{name}.cu"]
def jit_build(name, debug=False):
    _cuda_flags  = ['-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-arch=native', '--generate-line-info', '--restrict', 
                    f"-I {project_root}"]
    if(debug): _cuda_flags += ['-D__DEBUG_PRINT', '-g', '-G', '-D TORCH_USE_CUDA_DSA']
    return load(name=f"{name}", sources=_sources(name), 
            extra_cflags=[],
            extra_cuda_cflags=_cuda_flags)


def cuda_extension(name, debug):
    _cuda_flags  = [
                    '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', 
                    '-use_fast_math',
                    '-arch=native', 
                    '--generate-line-info', 
                    '--restrict', '-std=c++20',
                    f"-I {project_root}"
                    ]
    if(debug): _cuda_flags += ['-D__DEBUG_PRINT', '-g', '-G']
    return CUDAExtension(f'{name}', 
                        sources=_sources(name), 
                        extra_compile_args={'cxx' : ['-std=c++20'],
                                            'nvcc' : ['-O3'] + _cuda_flags})

def library_build(name, debug=False):
    setup(name=f"{name}", 
        ext_modules=[cuda_extension(name)], 
        cmdclass={'build_ext': BuildExtension})

#####
# Helpers
#   
def __eq(str, x,y, tol=1e-5, debug=False): 
    err = torch.abs(x-y).max()
    pass_str = "pass" if err < tol else "fail" 
    print(f"{str} : {pass_str} [err={err:0.5f}]")
    if(debug and (err > tol)):
        print(f"x\n{x}")
        print(f"y\n{y}")
        print(f"diff\n{x-y}")
        
    return err <= tol

def _rtile(b,n,d,dt): return torch.randn(b,n,d,device='cuda', dtype=dt)/(n*d)
def _rhtile(b,h,n,d,dt): return torch.randn(b,h,n,d,device='cuda', dtype=dt)/(n*d)
def _rones(b,n,d,dt): return torch.ones(b,n,d,device='cuda', dtype=dt)

def print_tiles(str, t):
    for i in range(t.size(0)):
        for j in range(t.size(1)//tile):
            print(f"{str} TILE batch={i} tile={j}")
            print(f"{t[i,j*tile:(j+1)*tile,:]}")