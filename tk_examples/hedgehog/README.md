

This kernel is for the [Hedgehog linear attention architecture](https://arxiv.org/abs/2402.04347) prefill stage. The structure of this kernel is similar to the Based architecture kernel -- you can read the README of that example for more background!

You can test it out via our C++ harness:
```
python generate_tests.py randn
# Ensure that the hedgehog_fwd_tk function and its imports are commented out in hedgehog.cu
# Ensure harness.impl is being imported (line is uncommented)
make clean && make && ./hedgehog randn.txt
```

You can also try it with PyTorch:
```
# Ensure harness.impl is commented out
python setup.py install # ensure that you have run ```source env.src''' prior to this
python hedgehog_profile.py
```

