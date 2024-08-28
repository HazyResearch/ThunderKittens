

Here we provide details to test and benchmark the Based kernel's *forward pass / inference prefill*. Note this kernel assumes feature dimension 16. 


## Overview of kernel
Using TK, we achieve a fast implementation of linear attention for the Based architecture! You can checkout these resources to learn more about the Based architecture: [Code](https://github.com/HazyResearch/based), [Paper](https://arxiv.org/abs/2402.18668). 

Standard attention computes an $O(N^2)$ matrix of query and key interactions $\exp(q_i^Tk_j/\sqrt{d})$. The idea in [linear attention](https://arxiv.org/abs/2006.16236) is to remove the softmax around the query-key dot product: 

![equation](https://latex.codecogs.com/png.image?%5Cinline%20%5Chuge%20%5Cdpi%7B110%7D%5Cbg%7Bwhite%7Dy_i=%5Csum_%7Bj=1%7D%5Ei%5Cfrac%7B%5Cexp(q_i%5ET%20k_j/%5Csqrt%7Bd%7D)v_j%7D%7B%5Csum_%7Bn=1%7D%5E%7Bi%7D%5Cexp(q_i%5ET%20k_n/%5Csqrt%7Bd%7D)%7D%5Crightarrow%20y_i=%5Csum_%7Bj=1%7D%5Ei%5Cfrac%7B%5Cphi(q_i)%5Cphi(k_j)v_j%7D%7B%5Csum_%7Bn=1%7D%5E%7Bi%7D%5Cphi(q_i)%5Cphi(k_n))

where $\phi$ is a *feature map* that transforms the keys and queries. This is like a kernel trick where we want $\exp(qk^T) \approx \phi(q)\phi(k)^T$. Letting the sequence length be $n$, model dimension $d$ and *feature dimension* after applying $\phi$ be $D$, note that we can now multiply keys and values first in $O(ndD)$ (instead of queries and keys in $O(n^2d)$

Another nice property of linear attention is that we can compute the outputs $y_i$ using a [recursive computation](https://arxiv.org/abs/2006.16236). We'll let $s_i = \sum_{j}^{i} \phi(k_j)^Tv_j$ be our "KV-state" and $z_i \sum_{j=1}^{i} \phi(k_j)^T$ be our "K-state". The update rule becomes:
$$s_i = s_{i-1} + \phi(k_i)^Tv_i, z_i = z_{i-1} + \phi(k_i)^T$$
And output computation becomes: 
$$y_i = \frac{\phi(q_i)s_i}{\phi(q_i)z_i}$$

In Based, our feature map computes a 2nd order Taylor approximation to the $\exp$ function:
$$\exp(x) \approx 1 + x + x^2/2$$
We compute a *concatenation* of the 0th, 1st, and 2nd order terms: 
$$\phi(q_i)^T\phi(k_j) = 1 + q_i^Tk_j + \frac{(q_i^Tk_j)^2}{2}$$
We use a feature dimension of $16$ when projecting queries and keys, so the resulting shape of $\phi(q), \phi(k)$ has dimension $273 = 1 + 16 + 16^2$. we need careful memory management to compute this feature map and outputs efficiently on hardware!

Details of this prefill kernel are provided in [Algorithm 1 of the Based paper](https://arxiv.org/pdf/2402.18668), and the implementation released today reflects a further improved algorithm (including extensions to H100 features)! We provide a high level description here as well. We compute $y_i$ using a combination of the *parallel* and *recurrent* views. Now letting $y_i$ be a $16 \times 16$ *chunk* of tokens, focusing on the numerator:

$$y_i = (\phi(q_i)^T\phi(k_i))v_i + \phi(q_i)\sum_{j=1}^{i-1}\phi(k_j)^Tv_j$$

The left-hand term computes the parallel view (multiplying queries and keys first), then applies causal masking on the tile, and muiltiplies by values. This is handled in the kernel as follows:
```
load(q, q_s[warpid]);
load(k, k_s[warpid]);

zero(local_attn);
mma_ABt(local_attn, q, k, local_attn);
make_causal(local_attn_bf, local_attn_bf, kittens::base_types::constants<bf16>::zero());

load(v, v_s[warpid]);
auto &v_col = swap_layout_inplace(v); // prepare for MMA

zero(o);
mma_AB(o, local_attn_bf, v_col, o);
```
The right-hand term is a simple matrix multiply (cuasality has already been handled from previous iterations over chunks of the sequence)! We partition across workers to store state $s$ in registers throughput! After streaming each chunk of tokens and computing chunks of output as shown above, we update the state: 
```
// Updating the KV state using the keys and values for the current chunk
mma_AB(a2, kt, v_col, a2); // accumulate onto a2
```


## Testing Correctness

We provide a lightweight test harness in C++ and in PyTorch. 
Testing in C++. First, go to the Makefile and select the correct GPU option for your hardware. To run the test:
```
python generate_tests.py randn_all
# 1. ensure that the based_tk_fwd function and its imports are commented out in 4090/lin_attn.cu and H100/lin_attn.cu
# 2. ensure harness.impl is being imported (line is uncommented) in that .cu file
# 3. in the Makefile, select the correct device option
make clean && make && ./debug_linear_attend randn_all.txt
```

Testing in PyTorch. Below, shows how you can install the TK kernel for the H100. The file checks that the output and the kv state are correctly computed and saved.
```
# 1. ensure that the based_tk_fwd function and its imports are uncommented (this is our PyTorch hook!) in H100/lin_attn.cu
# 2. ensure harness.impl is commented out in H100/lin_attn.cu (we don't want to use it!)

cd examples/based/linear_attn_forward/H100
python lin_attn_setup.py install     # ensure that you have run ```source env.src''' prior to this
cd ..
python test_correctness.py   
```
*Note* that the test may output 'fail', which we observe is due to numerical errors. A good way to confirm is to inspect the Tensors. 


## Benchmarking
We consider four baselines for benchmarking. You can toggle which ones you want to compare to in ```lin_attn_profile.py```. 
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


We provide scripts to compare the above baseline methods against the ThunderKittens kernels. 
```
cd examples/based/linear_attn_forward/4090
python setup.py install # ensure that you have run ```source env.src''' prior to this
python lin_attn_profile.py 
```

We also include functions to compute the TFLOPs in the ```harness.impl``` files and in the ```lin_attn_profile.py``` file. 



