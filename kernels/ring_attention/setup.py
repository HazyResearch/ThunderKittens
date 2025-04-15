import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Set environment variables
thunderkittens_root = os.getenv('THUNDERKITTENS_ROOT', os.path.abspath(os.path.join(os.getcwd(), '../../')))
python_include = subprocess.check_output(['python3', '-c', "import sysconfig; print(sysconfig.get_path('include'))"]).decode().strip()
torch_include = subprocess.check_output(['python3', '-c', "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))"]).decode().strip()

# CUDA flags
cuda_flags = [
    '-DNDEBUG',
    '-Xcompiler=-Wno-psabi',
    '-Xcompiler=-fno-strict-aliasing',
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '-forward-unknown-to-host-compiler',
    '--use_fast_math',
    '-std=c++20',
    '-O3',
    '-Xnvlink=--verbose',
    '-Xptxas=--verbose',
    '-Xptxas=--warn-on-spills',
    f'-I{thunderkittens_root}/include',
    f'-I{thunderkittens_root}/prototype',
    f'-I{python_include}',
    '-DTORCH_COMPILE',
    '-DKITTENS_HOPPER', # assume H100 for ring attn
    '-arch=sm_90a',     # assume H100 for ring attn
] + torch_include.split()
cpp_flags = [
    '-std=c++20',
    '-O3'
]
source_files = ['tk_ring_attention.cu']

setup(
    name='tk_ring_attention',
    ext_modules=[
        CUDAExtension(
            'tk_ring_attention',
            sources=source_files, 
            extra_compile_args={'cxx' : cpp_flags,
                                'nvcc' : cuda_flags}, 
            libraries=['cuda']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
