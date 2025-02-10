### ADD TO THIS TO REGISTER NEW KERNELS
sources = {
    'attn': {
        'source_files': {
            'h100': 'kernels/attn/h100/h100.cu' # define these source files for each GPU target desired.
        }
    },
    'hedgehog': {
        'source_files': {
            'h100': 'kernels/hedgehog/hh.cu'
        }
    },
    'based': {
        'source_files': {
            'h100': [
                'kernels/based/lin_attn_h100.cu',
            ],
            '4090': [
                'kernels/based/lin_attn_4090.cu',
            ]
        }
    },
    'cylon': {
        'source_files': {
            'h100': 'kernels/cylon/cylon.cu'
        }
    },
    'flux': {
        'source_files': {
            'h100': [
                'kernels/flux/flux_gate.cu',
                'kernels/flux/flux_gelu.cu'
            ]
        }
    },
    'fftconv': {
        'source_files': {
            'h100': 'kernels/fftconv/pc/pc.cu'
        }
    },
    'fused_rotary': {
        'source_files': {
            'h100': 'kernels/rotary/pc.cu'
        }
    },
    'fused_layernorm': {
        'source_files': {
            'h100': 'kernels/layernorm/non_pc/layer_norm.cu'
        }
    },
    'mamba2': {
        'source_files': {
            'h100': 'kernels/mamba2/pc.cu'
        }
    },
    'fp8_gemm': {
        'source_files': {
            'h100': 'kernels/matmul/FP8/matmul.cu'
        }
    },
    "test": {"source_files": {"h100": "kernels/test/matmul.cu"}},
}

### WHICH KERNELS DO WE WANT TO BUILD?
# (oftentimes during development work you don't need to redefine them all.)
# kernels = ['attn', 'mamba2', 'hedgehog', 'fftconv', 'fused_rotary', 'based', 'fused_layernorm']
kernels = ['test']

### WHICH GPU TARGET DO WE WANT TO BUILD FOR?
target = 'h100'
