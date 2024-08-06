
Here we provide a demo for running inference for pretrained [Based architecture language models](https://arxiv.org/abs/2402.18668) using TK kernels on an NVidia H100 GPU!

Setup python environments:
```bash
conda create -n dev python=3.11

pip3 install torch torchvision torchaudio
pip install transformers
pip install einops
pip install hydra-core
pip install flash-attn

git clone https://github.com/sustcsonglin/flash-linear-attention.git
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention

# auxiliary kernels
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/
cd csrc/layer_norm      && python setup.py install && cd ../../   # this step can take a little bit of time :(
cd csrc/rotary          && python setup.py install && cd ../../
cd csrc/fused_dense_lib && python setup.py install && cd ../../
```

Setup ThunderKittens kernels:
```bash 
cd ThunderKittens/
source env.src

# core prefill kernel
cd ThunderKittens/examples/based/linear_attn_forward/H100/
python lin_attn_setup.py install 

# core decode kernel
cd ThunderKittens//examples/based/based_inference/
python based_inference_setup.py install
```

## Run the generation demo with pretrained LMs!

Run benchmarks that really stress test the *in context learning ability* of different pretrained models. Examples of these tasks are documentQA and document information extraction! We report accuracy and the total time to complete the task.
```bash
cd ThunderKittens/examples/based/demo/
python document_ie_based.py -- model_name based
python document_ie_based.py -- model_name attn
python document_ie_based.py -- model_name mamba
```

Provide prompts of your choice!
```bash
cd ThunderKittens/examples/based/demo/
python generate_based.py
```

## Benchmarking!
```bash
cd ThunderKittens/examples/based/demo/
python benchmark/benchmark_long_prefill.py
```

