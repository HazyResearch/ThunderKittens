

Here we provide details to test and benchmark the Based kernel's *forward pass / inference prefill*. Note this kernel assumes feature dimension 16. 

Please checkout these resources to learn more about the Based architecture:
- [Code](https://github.com/HazyResearch/based)
- [Paper](https://arxiv.org/abs/2402.18668)


**Baselines.** We consider four baselines. You can toggle which ones you consider in ```lin_attn_profile.py```. 
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


**Testing** We provide a lightweight test harness in C++ and in PyTorch. 
Testing in C++. First, go to the Makefile and select the correct GPU option for your hardware. To run the test:
```
python generate_tests.py randn_all
# Ensure that the based_tk_fwd function and its imports are commented out in 4090/lin_attn.cu and H100/lin_attn_h100.cu
# Ensure harness.impl is being imported (line is uncommented)
make clean && make && ./debug_linear_attend randn_all.txt
```

Testing in PyTorch. Below, shows how you can install the TK kernel for the 4090:
```
cd examples/based/linear_attn_forward/4090
# Ensure that the based_tk_fwd function and its imports are uncommented
# Ensure harness.impl is commented out

python setup.py install # ensure that you have run ```source env.src''' prior to this
python lin_attn_profile.py
```
*Note* that the test may output an error, which we observe is due to numerical differences in the computation approaches as opposed to correctness.


**Benchmarking.** We provide scripts to compare the above baseline methods against the ThunderKittens kernels. 
```
cd examples/based/linear_attn_forward/4090
python setup.py install # ensure that you have run ```source env.src''' prior to this
python lin_attn_profile.py 
```

We also include functions to compute the TFLOPs in the ```harness.impl``` files and in the ```lin_attn_profile.py``` file. 



