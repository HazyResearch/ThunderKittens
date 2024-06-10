import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
import subprocess

def get_last_arch_torch():
    arch = torch.cuda.get_arch_list()[-1]
    print(f"Found arch: {arch} from existing torch installation")
    return arch

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
    if int(bare_metal_major) >= 11 and int(bare_metal_minor) >= 2:
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args

arch = get_last_arch_torch()
sm_num = arch[-2:]
# SA: Needs to be compute_90a for H100
cc_flag = ['--generate-code=arch=compute_90a,code=compute_90a']
# SA: Needs to be compute_80 for A100
# cc_flag = ['--generate-code=arch=compute_80,code=compute_80']


setup(
    name='causal_attention_cuda_cpp',
    ext_modules=[
        CUDAExtension('causal_attention_cuda', [
            #'causal_attention.cpp',
            'causal_attention_cuda.cu',
        ],
        extra_compile_args={'cxx': ['-O3'],
                             'nvcc': append_nvcc_threads(['-O3', '-lineinfo', '--use_fast_math', '-std=c++17'] + cc_flag)
                             # 'nvcc': append_nvcc_threads(['-O3', '-lineinfo', '--use_fast_math', '-std=c++14'] + cc_flag)
                            })
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
