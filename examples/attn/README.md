
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

Configure the parameters `ATTN_B`, `ATTN_H`, `ATTN_N`, `ATTN_D` in the implementation files:

- For inference: Modify `harness_h100_fwd.impl`
- For training (forward + backward): Modify `harness_h100_bwd.impl`

## Execution

To compile and run the harness with `randn_4096_64.txt` (ATTN_N = 4096, ATTN_D = 64) for example, use the following commands:

```bash
make clean && make && ./attn randn_4096_64.txt
```

## Output Verification 

Though the C++ harness provides an approximate view of correctness with its smiley face, numerics in floating point arithmetic on the GPU can sometimes arise: to manually check the reference output, generated output, and diff: 

- Create a folder called `printouts` in the same directory as the `.cu` file you are executing
- Running `./attn randn_INSERT_INSERT.txt` will now dump .txt files of reference, generation, and diff in the `printouts` directory

## Sample Output

- Inference
```bash
Entered main!
Starting to enter!
Finished loading Q
Finished loading K
Finished loading V
Finished loading O_REF
Finished loading file from randn_4096_64.txt!
Allocated and set memory on GPU!
Set max dynamic memory!
Starting warmup
Starting kernel
Finished kernel
Average execution time: 5226 us
Correct :)
Efficiency: 420.785 TFLOPS
```
- Training
```bash
Starting to enter file!
Finished loading Q
Finished loading K
Finished loading V
Finished loading O_REF
Finished loading OG
Finished loading QG_REF
Finished loading KG_REF
Finished loading VG_REF
Finished computing D_REF
Finished loading file from randn_4096_64.txt!

Starting forward pass!
Allocated and set memory on GPU for forward!
Set max dynamic memory!
Starting fwd kernel
Finished fwd kernel
Average fwd execution time: 6457 us
FWD Correct :)

Starting backward pass!
Allocated and set memory on GPU for backward prep!
Set max dynamic memory!
Starting bwd prep kernel
Finished bwd prep kernel
Average bwd prep execution time: 188 us
BWD Prep Correct :)

Allocated and set memory on GPU for backward!
Set max dynamic memory!
Starting bwd kernel
Finished bwd kernel
Average bwd execution time: 17544 us
BWD Correct :)

Backwards efficiency: 310.036 TFLOPS
```


# TK Kernels + PyTorch Bindings/Demo Steps

## Environment Setup

First, source the environment from the Thunderkittens directory:

```bash
source env.src
```

## Build Setup

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

To measure the approximate performance of the implementations:

- For inference:
  ```bash
  python h100_fwd_atn.py
  ```
- For training (forward + backward):
  ```bash
  python h100_train_atn.py
  ```
