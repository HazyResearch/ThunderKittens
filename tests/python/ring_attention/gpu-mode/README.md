# ring-attention

Ring Attention leverages blockwise computation of self-attention on multiple GPUs and enables training and inference of sequences that would be too long for a single devices.

This repository contains notebooks, experiments and a collection of links to papers and other material related to Ring Attention.

### Reserach / Material

- Paper: [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)
  - code: [lhao499/ring-attention](https://github.com/lhao499/ring-attention)
- Paper: [World Model on Million-Length Video And Language With RingAttention](https://arxiv.org/abs/2402.08268)
  - code: [LargeWorldModel/LWM](https://github.com/LargeWorldModel/LWM),
  - project site: [largeworldmodel.github.io](https://largeworldmodel.github.io/)
  - models: [HF/LargeWorldModel](https://huggingface.co/LargeWorldModel)
- Paper: [Striped Attention: Faster Ring Attention for Causal Transformers](https://arxiv.org/abs/2311.09431), code: [exists-forall/striped_attention](https://github.com/exists-forall/striped_attention)
- Paper (2022): 4D parallelism: [Sequence Parallelism: Long Sequence Training from System Perspective](https://arxiv.org/abs/2105.13120)
- related: [Flash-Decoding for long-context inference](https://www.together.ai/blog/flash-decoding-for-long-context-inference) (together.ai blog)

- Paper: [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (NVIDIA, 2018)
- [ELI5: FlashAttention](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad) by [Aleksa Gordić](https://twitter.com/gordic_aleksa)
- LWM model in ollama: https://ollama.com/ifioravanti/lwm
- Phil Wang's (lucidrain) pytorch impl: [lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch)
- Zilin Zhu's nice [zhuzilin/ring-flash-attention](https://github.com/zhuzilin/ring-flash-attention) implementation


### Notebooks
- [Incremental Softmax](https://colab.research.google.com/drive/1PNDTLx2UYYk8XmTb9e_ZBxPx8P6eByvx?usp=sharing) (to understand the algorithm in 'high-level' pytorch)
- [Naive flash-attn](https://colab.research.google.com/drive/1X-x6PCRydNY9LZBPLA0DZh3Tj2Dyz60M?usp=sharing) (to understand the algorithm in 'high-level' pytorch)


### Development References
- [NVIDIA Collective Communication Library (NCCL) Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [PyTorch Distributed Overview](https://pytorch.org/docs/stable/distributed.html)
- [Distributed communication package - torch.distributed](https://pytorch.org/docs/stable/distributed.html) (`send()`, `recv()`, `broadcast()`, etc.)

## How to contribute

Contact us on the **GPU MODE** discord server: [https://discord.gg/gpumode](https://discord.gg/gpumode), PRs are welcome (please create an issue first).
