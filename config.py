### ADD TO THIS TO REGISTER NEW KERNELS
sources = {
    'attn_inference': {
        'source_files': {
            'h100': 'kernels/attn/h100/h100.cu' # define these source files for each GPU target desired. (they can be the same.)
        }
    },
    'attn_training': {
        'source_files': {
            'h100': 'kernels/attn/h100/h100.cu'
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
    'fused_layernorm': {
        'source_files': {
            'h100': [
                'kernels/fused_layernorm/layer_norm.cu',
            ]
        }
    },
    'fused_rotary': {
        'source_files': {
            'h100': [
                'kernels/fused_rotary/rotary.cu',
            ]
        }
    }
}

### WHICH KERNELS DO WE WANT TO BUILD?
# (oftentimes during development work you don't need to redefine them all.)
# kernels = ['attn_inference', 'attn_training', 'hedgehog', 'fused_rotary']
kernels = ['based']

### WHICH GPU TARGET DO WE WANT TO BUILD FOR?
target = 'h100'

