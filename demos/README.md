

## TK Demos 

We include a series of demos to show how TK kernels plug in for speed ups on AI workloads. 


### Attention 

Attention powers a large number of current LLMs. TK includes forwards / prefill and backwards kernels. We include causal, non-causal, and GQA variants.

We include:
1. nanoGPT integration: train Transformer models using TK kernels
2. LLM inference integration: 
- Run [Llama 3 8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) with TK GQA attention
- Run [Qwen 2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) with TK attention

### LoLCATS

LoLCATS is a recent state-of-the-art method for converting quadratic attention Transformer LLMs to linear attention LLMs. TK includes a forwards / prefill kernel. 

We include: 
1. LLM inference integration:
- Run [LoLCATS-Llama 3.1 8B](https://huggingface.co/collections/hazyresearch/lolcats-670ca4341699355b61238c37) with TK 

### Based

Based is a state-of-the-art linear attention architecture that combines short sliding window attentions with large-state-size linear attentions. TK includes a forwards / prefill kernel.

Added installs:
```bash
pip install -U git+https://github.com/sustcsonglin/flash-linear-attention
```

We include:
LLM inference integration:
- Run [Based 1.3B](https://huggingface.co/hazyresearch/my-awesome-model) with TK on a series of recall-intensive in-context learning tasks


