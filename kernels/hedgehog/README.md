This kernel is a hedgehog modified for speed. It implements:
 - Fused MLPs to produce the Q, K featurizations
 - Fused block sliding window attention
 - Block linear attention
 - Normalization across both sliding window attention and linear attention components
 - K, KV state writeouts