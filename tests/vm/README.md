# Kittens Virtual Machine

Kernel launches are super expensive. So let's run the entire model in a single kernel!

## Project Organization

Currently a bit disorganized, but largely everything is in:

- This directory (`./tests/vm`) - contains all relevant TK kernels
- `./demos/kvm-runner` - contains the Python framework that actually runs this thing
- Google Docs. But this repo should contain everything up-to-date

## Milestones

### By Apr 23 (Wed)

- Get `meta-llama/Llama-3.2-1B-Instruct` forward pass to run.
- Main to-do is to implement the instructions defined in `./demos/kvm-runner/kvm_runner/instructions.py`
