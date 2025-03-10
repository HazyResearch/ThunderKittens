# ThunderGQA

**Updates:**
- 03/06/2025: ThunderGQA released! Follow the [mla_decode](../mla_decode) folder for more updates!

Now releasing ThunderGQA! The ThunderMLA kernel you know and love, adapted for GQA decoding.

On a representative workload of batch size 4, sequence lengths `[4641,45118,1730,1696]`, and 4 new query tokens, ThunderGQA runs in **22.8 us**, compared to **36.5 us** for FlashAttention 3, benchmarked at [commit `4f0640`](https://github.com/Dao-AILab/flash-attention/tree/4f0640d534888c579a448fd89c2d4e064905d798).

Below we have benchmarks for other workloads:

| Kernel | B 1, Seq 64k, Q 1 | B 64, Seq 256-1024 (random), Q 4 | B 64, Seq 512 Q 2 | B 132, Seq 4k, Q 4 |
|--------|-------------|---------------------------|--------------|--------------|
| **FA3 (μs)**        | 42.4       | 33.9      | 26.5      | 118.6 |
| **ThunderGQA (μs)** | 25.2       | 19.9      | 18.2      | 98.2 |

Enjoy!
