

## TK Demos: play with kittens!

<div align="center" >
    <img src="assets/kittens.png" height=350 alt="Kitten workers" style="margin-bottom:px"/> 
</div>

<br>


### General setup 

Several of these demos are setup to use large 8b models from Hugging Face. To setup, run login:
```bash 
huggingface-cli login
```
Set the directory at which you want the models to download in the `_model_config.yaml` file in the `demos/configs/` directory.

Next, install the TK kernels: 
1. From the `ThunderKittens/` directory, run `source env.src` to set the environment variables.
2. In `ThunderKittens/config.py` select `hedgehog` and `attn` and run `python setup.py install` to install the TK kernels.


### Attention 

Attention powers a large number of current LLMs. TK includes forwards / prefill and backwards kernels. We include causal, non-causal, and GQA variants.

We include:
1. Try training with kittens! Checkout [tk-training](https://github.com/HazyResearch/train-tk)
2. LLM inference integration: 
- Run [Llama 3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) with TK GQA attention
- Run [Qwen 2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) with TK attention
```bash
cd llama_demo/
bash demo_8b.sh
```
And enter your prompt, e.g., "The capital of America is"

### LoLCATS

[LoLCATS](https://github.com/HazyResearch/lolcats) is a recent state-of-the-art method for converting quadratic attention Transformer LLMs to linear attention LLMs. TK includes a forwards / prefill kernel. 

We include: 
1. LLM inference integration:
- Run [LoLCATS-Llama 3.1 8B](https://huggingface.co/collections/hazyresearch/lolcats-670ca4341699355b61238c37) with TK 
```bash
cd lolcats_demo/
bash demo_8b.sh
```
And enter your prompt, e.g., "The capital of America is"

### Based

[Based](https://github.com/HazyResearch/based/tree/main) is a linear attention architecture that combines short sliding window attentions with large-state-size linear attentions. TK includes a forwards / prefill kernel.

Added installs:
```bash
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
```

We include:
LLM inference integration:
- Run [Based 1.3B](https://huggingface.co/hazyresearch/my-awesome-model) with TK on a series of recall-intensive in-context learning tasks



### Your Demos!

If you use TK to build any demos, please reach out / make a PR! We'd love to feature it here!!

- DeltaNet: https://github.com/proger/accelerated-scan/tree/delta

