"""This module contains RingAttention forward and backward pass, supporting both blockwise computation in Jax and fused computation in Pallas.
It features blockwise computation for feedforward networks to reduce memory cost.
For more details, refer to RingAttention ('Ring Attention with Blockwise Transformers for Near-Infinite Context') at https://arxiv.org/abs/2310.01889 and BlockwiseTransformer ('Blockwise Parallel Transformer for Large Context Models') at https://arxiv.org/abs/2305.19370.
"""

from .ringattention_inference import ring_attention_inference
from .ringattention_jax import ring_attention, blockwise_feedforward
from .ringattention_pallas_gpu import ring_flash_attention_gpu
from .ringattention_pallas_tpu import ring_flash_attention_tpu
import jax

platform = jax.lib.xla_bridge.get_backend().platform
if platform == "tpu":
    # pallas ringattention for tpu
    ringattention = ring_flash_attention_tpu
elif platform == "gpu":
    # pallas ringattention for gpu
    ringattention = ring_flash_attention_gpu
else:
    # jax ringattention
    ringattention = ring_attention

ringattention_jax = ring_attention
ringattention_inference = ring_attention_inference

__all__ = ["ringattention", "blockwise_feedforward", "ringattention_jax",
           "ringattention_inference"]
