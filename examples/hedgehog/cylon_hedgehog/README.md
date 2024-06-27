## How to build for PyTorch

First make sure you ran `source env.src` in the TK root directory.

Then run `python hh_fuse_setup.py build`

Your library should appear in the build/ directory after about a minute.

## How to build harness otherwise

First comment out the `TORCH_COMPILE` defined at the top of `hh_fused.cu`. Then run `make`