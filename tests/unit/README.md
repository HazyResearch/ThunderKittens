# Kittens Unit Tests

This directory contains unit tests for ThunderKittens.

### Requirements

- NVIDIA GPU
- CUDA Toolkit (12.3 or 12.4 preferred.)
- C++20 compatible compiler

### Compiling the Tests

To compile the unit tests, use the provided Makefile in this directory. It is highly recommended to run the compilation with multiple threads to speed up the process. For example, if your machine has 32 threads, run:

```bash
make -j32
```

### Compilation Options
The Makefile provides several options to customize the compilation:
- `GPU_TARGET`: Set to either `4090`, `A100`, or `H100` to specify the target GPU architecture (default: H100).
- `COMP_LEVEL`: Set the compiler optimization level. Available options are `fast`, `debug`, and `profile` (default: fast).
- `TEST_INTENSITY`: Set the level of test intensity. Higher levels compile more tests but take longer. Available options are 1, 2, 3, and 4 (default: 2).
- `TEST_ALL`: Compile and run all available tests. You can also specify individual test sections or tests using flags like -DTEST_WARP_MEMORY or -DTEST_WARP_MEMORY_VEC_DSMEM.

### Running the Tests
After successful compilation, run the tests using:

```bash
mkdir outputs
./unit_tests printout
```
This will execute the compiled unit tests and dump results of any failed tests to the `outputs/` folder. As a quick note, it is expected for mma tests to occasionally fail. Careful inspection of the output will usually show just a single element differing by a small amount, which we think is due to how floating-point arithmetic is implemented within the tensor cores.

### Cleaning the Build
To clean the build directory and remove the compiled binary, run:

```bash
make clean
```

## Contributing

If you would like to contribute new tests or improve existing ones, please follow the established coding style and naming conventions. Make sure to test your changes thoroughly before submitting a pull request.

The unit tests directly mirror the file structure of the main repo. This makes it much easier to track coverage. and identify untested regions of code.

For more information on contributing to the Kittens project, please refer to the main repository's contributing guidelines.