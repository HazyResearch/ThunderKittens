### ADD TO THIS TO REGISTER NEW KERNELS
sources = {
    'attn_inference': {
        'source_files': {
            'h100': 'kernels/attn/h100/h100_fwd.cu' # define these source files for each GPU target desired. (they can be the same.)
        }
    },
    'attn_training': {
        'source_files': {
            'h100': 'kernels/attn/h100/h100_train.cu'
        }
    },
    'attn_causal_inference': {
        'source_files': {
            'h100': 'kernels/attn_causal/h100/h100_fwd.cu'
        }
    },
    'attn_causal_training': {
        'source_files': {
            'h100': 'kernels/attn_causal/h100/h100_train.cu'
        }
    }
}

### WHICH KERNELS DO WE WANT TO BUILD?
# (oftentimes during development work you don't need to redefine them all.)
kernels = ['attn_inference', 'attn_causal_inference', 'attn_training', 'attn_causal_training']
# kernels = ['attn_training', 'attn_inference']

### WHICH GPU TARGET DO WE WANT TO BUILD FOR?
target = 'h100'