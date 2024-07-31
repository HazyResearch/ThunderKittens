![alt text](https://github.com/HazyResearch/ThunderKittens/blob/main/examples/attn/h100/image?raw=true)

# TK Kernels + C++ Harness Steps

## Setup

Before running the harness, select the purpose of the run:

- **Inference**: Set `PURPOSE=inference` in `Makefile`.
- **Training**: Set `PURPOSE=training` in `Makefile`.

## Generate Test Data

Run the following command to generate the `.txt` files required for running the harness. Replace `INSERT_ATTN_N` and `INSERT_ATTN_D` with the desired dimensions:

```bash
python gentests.py randn INSERT_ATTN_N INSERT_ATTN_D
```

## Configuration

Configure the parameters `ATTN_B`, `ATTN_H`, `ATTN_N`, `ATTN_D` in the implementation files (ATTN_D = 64, 128 supported for inference, ATTN_D = 64 supported for training)

- For inference: Modify `harness_h100_fwd.impl`
- For training (forward + backward): Modify `harness_h100_bwd.impl`

If defined at the top of the corresponding .cu file, make sure to comment out `TORCH_COMPILE`

## Execution

To compile and run the harness with `randn_4096_64.txt` (ATTN_N = 4096, ATTN_D = 64) for example, use the following commands:

```bash
make clean && make && ./attn randn_4096_64.txt
```

## Output Verification 

Though the C++ harness provides an approximate view of correctness with its smiley face, numerics in floating point arithmetic on the GPU can sometimes arise: to manually check the reference output, generated output, and diff: 

- Create a folder called `printouts` in the same directory as the `.cu` file you are executing
- Running `./attn randn_INSERT_INSERT.txt` will now dump .txt files of reference, generation, and diff in the `printouts` directory

# TK Kernels + PyTorch Bindings/Demo Steps

## Environment Setup

First, source the environment from the Thunderkittens directory:

```bash
source env.src
```

## Build Setup

If not defined at the top of the corresponding .cu file, make sure `TORCH_COMPILE` is defined: 

```bash
#define TORCH_COMPILE
```

Build the Python bindings/setup for both forward and training setups:

```bash
python h100_fwd_setup.py build && python h100_train_setup.py build
```

## Check Correctness

To verify the correctness of the implementations:

- For inference: 
  ```bash
  python h100_fwd_check.py
  ```
- For training (forward + backward):
  ```bash
  python h100_train_check.py
  ```

## Performance Measurement

To measure the approximate performance of the implementations (ATTN_D = 64, 128 supported for inference, ATTN_D = 64 supported for training):

- For inference:
  ```bash
  python h100_fwd_atn.py
  ```
- For training (forward + backward):
  ```bash
  python h100_train_atn.py
  ```
