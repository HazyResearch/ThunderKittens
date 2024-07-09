### ADD TO THIS TO REGISTER NEW KERNELS
sources = {
    'attn_inference': {
        'source_files': {
            'h100': 'attn/h100/h100_fwd.cu' # define these source files for each GPU target desired. (they can be the same.)
        }
    },
    'attn_training': {
        'source_files': {
            'h100': 'attn/h100/h100_train.cu'
        }
    }
}

### WHICH KERNELS DO WE WANT TO BUILD?
kernels = ['attn_inference', 'attn_training']

### WHICH GPU TARGET DO WE WANT TO BUILD FOR?
target = 'h100'