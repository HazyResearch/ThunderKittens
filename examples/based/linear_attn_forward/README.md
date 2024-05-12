

Here we provide instructions to test and benchmark the Based kernel. 


**Baselines.** We consider four baselines:
1. Pure PyTorch

2. Fast Transformers causal dot product kernel: 
```
cd csrc/causal_dot_prod
python setup.py install
```

3. Flash Linear Attention triton kernels:
```
git clone https://github.com/sustcsonglin/flash-linear-attention
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
```

4. Flash Attention
```
pip install flash-attn ==2.5
```


**Testing** We provide a test harness in C++ and in PyTorch. 
Testing in C++. First, go to the Makefile and select the correct GPU option for your hardware. To run the test:
```
python generate_tests.py randn_all
# Ensure that the based_tk_fwd function and its imports are commented out
# Ensure harness.impl is being imported
make clean && make && ./debug_linear_attend randn_all.txt
```

Testing in PyTorch. Below, you can install for the 4090 or H100 depending on your hardware:
```
cd examples/based/linear_attn_forward/4090
# Ensure that the based_tk_fwd function and its imports are uncommented
# Ensure harness.impl is commented out
python setup.py install # ensure that you have run ```source env.src''' prior to this
python lin_attn_profile.py
```
*Note* that the test may output an error, which we observe is due to numerical differences in the computation approaches as opposed to correctness.


**Benchmarking.** We provide scripts to compare these methods on the H100 and 4090. 
```
cd examples/based/linear_attn_forward/4090
python setup.py install # ensure that you have run ```source env.src''' prior to this
python lin_attn_profile.py
```




