### ADD TO THIS TO REGISTER NEW KERNELS
sources = {
    'attn': {
        'source_files': {
            'h100': 'kernels/attn/h100/h100.cu' # define these source files for each GPU target desired. (they can be the same.)
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
                'kernels/based/linear_prefill/linear_prefill.cu',
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
    'mamba2': {
        'source_files': {
            'h100': 'kernels/mamba2/pc.cu'
        }
    }
}

### WHICH KERNELS DO WE WANT TO BUILD?
# (oftentimes during development work you don't need to redefine them all.)
# kernels = ['attn', 'mamba2', 'hedgehog', 'cylon', 'fftconv', 'fused_rotary']
kernels = ['attn']

### WHICH GPU TARGET DO WE WANT TO BUILD FOR?
target = 'h100'
