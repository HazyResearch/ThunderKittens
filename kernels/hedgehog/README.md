This kernel is a hedgehog modified for speed. It implements:
 - Fused MLPs to produce the Q, K embeddings
 - Fused block sliding window attention
 - Block linear attention
 - Normalization across both sliding window attention and linear attention components
 - State writeouts