
Here we provide a demo for running inference for pretrained [Based architecture language models](https://arxiv.org/abs/2402.18668) using TK kernels on an NVidia H100 GPU!

Setup python environments:
```bash
conda create -n dev python=3.11
pip3 install torch torchvision torchaudio
pip install transformers
pip install einops
pip install hydra-core
pip install flash-attn
```

Setup ThunderKittens kernels:
```bash 
# step 1: in ThunderKittens/config.py select "based"
# step 2: run the following
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
As a baseline, you can install FLA, handcrafted Triton kernels for linear attention
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

