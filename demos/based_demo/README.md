# Based Architecture with ThunderKittens 

Here we provide a demo for running inference for pretrained [Based architecture language models](https://arxiv.org/abs/2402.18668) using TK kernels on an NVidia H100 GPU! 
The Based architecture is notable for (1) showing the value of **hybridizing local attention (window sizes $\leq 128$)** with linear attention and (2) building a method to **increase the linear attention state size in hardware efficient ways**. 

Emphasizing this, Based expands the Pareto frontier of the quality-efficiency tradeoff space. In the figure below, for **end-to-end pretrained models**, Based computes inference prefill over $2.3\times$ as fast as Mamba (with $8\times$ smaller recurrent state), and is also faster than Mamba-2 and Based implemented with Flash Linear Attention! Explore our hardware-efficient architecture and algorithm in the demo below!

<div align="center" >
    <img src="plots/benchmark_input8000_output1.png" height=350 alt="Benchmark models" style="margin-bottom:px"/> 
</div>


Setup python environment:
```bash
conda create -n dev python=3.11
pip3 install torch torchvision torchaudio
pip install transformers
pip install einops
pip install hydra-core
pip install flash-attn
```

Setup ThunderKittens kernels. First, in ``ThunderKittens/config.py`` select "based". Then:
```bash 
python setup.py install 
```

## Run the generation demo with pretrained LMs!

Run benchmarks, like document QA and information extraction, that stress test the *in context learning ability* of different architetures! We compute model accuracy and the total time to complete the task. Each of the models below is trained on the exact same data to make it easy to compare.
```bash
cd ThunderKittens/demos/based_demo/
python document_ie_based.py -- model_name based
python document_ie_based.py -- model_name attn
python document_ie_based.py -- model_name mamba
```

Run generation with prompts of your choice:
```bash
python generate_based.py
```

## Benchmarking!
As a baseline for TK, you can install Flash Linear Attention CUDA kernels.
```
git clone https://github.com/sustcsonglin/flash-linear-attention.git
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
```

To run the benchmarking script and produce a plot to compare different (linear) attention approaches *on the end-to-end models*, run:
```bash
python benchmark.py
```

To benchmark the standalone kernel:
```bash
python benchmark_kernel.py
```


Please cite the following if you use this code or build off of our linear attention kernel:
```
@article{arora2024simple,
  title={Simple linear attention language models balance the recall-throughput tradeoff},
  author={Arora, Simran and Eyuboglu, Sabri and Zhang, Michael and Timalsina, Aman and Alberti, Silas and Zinsley, Dylan and Zou, James and Rudra, Atri and Ré, Christopher},
  journal={arXiv:2402.18668},
  year={2024}
}
```

