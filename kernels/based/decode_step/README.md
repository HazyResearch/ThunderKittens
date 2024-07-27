

Here we provide instructions to test and benchmark a Based architecture (or linear attention more broadly!) inference kernel for the *decoding step*. Please see the ```based/linear_attn_forward/`` folder for the *prefill / forward* pass. Note this kernel assumes feature dimension 16. 

Please checkout these resources to learn more about the Based architecture:
- [Code](https://github.com/HazyResearch/based)
- [Paper](https://arxiv.org/abs/2402.18668)


**Testing** We provide a lightweight PyTorch test. 
Testing in PyTorch. Below, shows how you can install the TK kernel for the 4090:
```
cd examples/based/based_inference/
python based_inference_setup.py install 
python based_inference_test.py
```

**Benchmarking.** We provide scripts to compare PyTorch against the ThunderKittens kernels in wall clock speed.
```
cd examples/based/based_inference/
python based_inference_setup.py install 
python based_inference_profile.py
```

