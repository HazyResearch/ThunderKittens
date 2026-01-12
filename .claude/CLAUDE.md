# About CUDA

## Blackwell (SM100)

- 2 CTA MMA (cta_group::2): only a leader CTA issues tcgen05.mma—the peer CTA does NOT call mma. Each CTA loads its local A tile to SMEM; each CTA loads half of B to SMEM. The mma instruction reads A from the issuing CTA's SMEM, but reads B from BOTH CTAs' SMEM via DSMEM; the hardware fetches the peer's B tile automatically. Accumulator D in TMEM is local to each CTA. To observe mma completion, tcgen05.commit is called by the leader CTA which arrives on an mbarrier--required because tcgen05.mma is async--on both CTAs.

# About ThunderKittens

# General Rules

- Do NOT modify files except those specifically mentioned in the prompt. If other changes are needed, explictly ASK first
- When modifying kernels, first benchmark and verify correctness to establish a baseline for regression testing
- When modifying kernels, always compile the kernel before testing
