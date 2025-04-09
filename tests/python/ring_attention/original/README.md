## GPU/TPU Jax implementation of RingAttention

This codebase provides the implementation of the Ring Attention with Blockwise Transformers. The model is described in the paper [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/pdf/2310.01889.pdf) and [Blockwise Parallel Transformer for Large Context Models](https://arxiv.org/pdf/2305.19370.pdf).

Ring Attention with Blockwise Parallel Transformers enables training sequences up to a length of 'number of devices' times longer than those possible with BPT. This is achieved by distributing the attention and feedforward computation across multiple devices and overlapping the communication with computation. Thanks to the blockwise computing of the attention and feedforward network, it is possible to train with tens of millions of tokens in context size without adding any communication or computation overhead.


### Example usage here with code snippets
First, install the package:
```bash
pip install ringattention
```
Then, `ringattention` and `blockwise_feedforward` can be imported and used as follows:
```python
from ringattention import ringattention, blockwise_feedforward
```
You can use the `ringattention` function by wrapping it with `shard_map` to shard the computation across multiple devices. Here is an example of how to use the `ringattention` function with sharding:
```python
ring_attention_sharded = shard_map(
    partial(
        ringattention,
        axis_name="sp",
        float32_logits=True,
        cache_idx=None,
        blockwise_kwargs=dict(
            causal_block_size=1,
            deterministic=True,
            dropout_rng=None,
            attn_pdrop=0.0,
            query_chunk_size=512,
            key_chunk_size=512,
            policy=jax.checkpoint_policies.nothing_saveable,
            dtype=jax.numpy.float32,
            precision=None,
            prevent_cse=True,
        )
    ),
    mesh=LLaMAConfig.get_jax_mesh(self.config.mesh_dim),
    in_specs=(
        PS(("dp", "fsdp"), "sp", "tp", None),
        PS(("dp", "fsdp"), "sp", "tp", None),
        PS(("dp", "fsdp"), "sp", "tp", None),
        PS(("dp", "fsdp"), None, None, None),
        PS(("dp", "fsdp"), None),
    ),
    out_specs=PS(("dp", "fsdp"), "sp", "tp", None),
    check_rep=False
)
attn_output = ring_attention_sharded(xq, xk, xv, attention_bias, segment_ids)
```

Explanation of the arguments:

- `query_chunk_size` and `key_chunk_size` are the chunk sizes for the query and key. Choose as large as you can untill out of memory to speed up the computation.

- `policy` is the checkpoint policy for the attention weights, use `jax.checkpoint_policies.nothing_saveable` to enable the checkpointing.

- `causal_block_size` is the block size for block-causal attention. `causal_block_size=1` is equivalent to causal attention.

- `cache_idx` is the cache index for inference. If `cache_idx` is not `None`, the attention weights will be cached and reused in the next inference.

RingAttention is utilized in [Large World Model (LWM)](https://largeworldmodel.github.io/) for million-length vision-language training, where a complete example of using RingAttention and BlockwiseTransformers can be found: [LWM codebase](https://github.com/LargeWorldModel/LWM)


## Reference
If you find our work relevant to your research, please cite:
```bibtex
@article{liu2023blockwise,
    title={Blockwise Parallel Transformer for Large Context Models},
    author={Liu, Hao and Abbeel, Pieter},
    journal={Advances in neural information processing systems},
    year={2023}
}
```
```bibtex
@article{liu2023ring,
    title={Ring Attention with Blockwise Transformers for Near-Infinite Context},
    author={Liu, Hao and Zaharia, Matei and Abbeel, Pieter},
    journal={arXiv preprint arXiv:2310.01889},
    year={2023}
}
```
