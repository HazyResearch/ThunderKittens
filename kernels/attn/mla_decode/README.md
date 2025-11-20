# ThunderMLA

**Updates:**
- 03/06/2025: ThunderGQA released! Check it out [here](../gqa_decode)!
- 03/05/2025: ThunderMLA released!

We've been playing with some new schedulers to deal with variable length sequences. (This is a common case in LLM inference, when you're serving requests from many different users.) Like everyone else, we were really excited about DeepSeek MLA so we decided to look into it!

We're excited to introduce ThunderMLA, our response to the performance challenges of large language model inference. ThunderMLA is a completely fused "megakernel" for decode that's 20-35% faster than DeepSeek's FlashMLA on diverse workloads. It turns out some simple scheduling tricks can get you pretty far! And although this release is focused on attention decoding, we think these techniques apply much more broadly.

On a representative workload with imbalanced inputs:
- A batch of 4 prompts, of lengths `[4641, 45118, 1730, 1696]`
- Generating 4 new tokens (e.g., for a speculator)
- 8-way tensor parallel -- that is, 16 heads per GPU for DeepSeek R1
ThunderMLA runs in **41 us**, compared to **52 us** for FlashMLA!

Benchmarks:

| Kernel | B 1, Seq 64k, Q 1 | B 64, Seq 256-1024 (random), Q 4 | B 64, Seq 512 Q 2 | B 132, Seq 4k, Q 4 |
|--------|-------------|---------------------------|--------------|--------------|
| **FlashMLA (μs, TFLOPs, GB/s)** | 55.0, 42, 1378 | 47.0, 124, 1212 | 39.0, 59, 1092 | 226.0, 333, 2839 |
| **ThunderMLA (μs, TFLOPs, GB/s)** | 44.5, 52, 1700 | 39.5, 147, 2022 | 28.6, 80, 1489 | 210.0, 358, 3055 |
| **Speedup (%)** | **23.6%** | **19.0%** | **36.3%** | **7.6%** |