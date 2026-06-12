# NVFP4 B300 GEMM performance handoff

## 2026-06-12 update: CLC work stealing landed; scheduling is now ruled out

Cluster-launch-control work stealing (CUTLASS-equivalent) is implemented and
correct. The grid covers the full problem (64x32 CTAs for the target shape);
each cluster processes its blockIdx home chunk and then steals pending cluster
launches via `clusterlaunchcontrol.try_cancel ... .multicast::cluster::all`.
The canceled chunk has the requesting cluster's shape, and each CTA covers
first_ctaid + its own in-cluster offset; tile mapping reuses the
preferred-cluster-aligned region swizzle, so any mix of preferred and fallback
chunks tiles the grid bijectively. err=0 for 2x1, 4x1, 2x2, 2x4, 4x2, 4x4
preferred shapes with 2x1 fallback.

Plumbing notes (hard-won):
- The response/full/empty ring (CLC_DEPTH=3) is consumed by the tile and scale
  producer warps, the MMA warp (pair leaders), and the consumer warpgroup;
  producer warp 1 is the scheduler.
- CRITICAL: slot reuse must be paced CLUSTER-WIDE. Only cluster rank 0 issues
  try_cancel and waits clc_empty, and every consumer on every CTA arrives at
  rank 0's clc_empty via tma::cluster::arrive. Per-CTA pacing deadlocks: the
  multicast response clobbers a peer CTA's unread slot, the pair's item
  sequences diverge, and the pair leader's tiles_arrived waits forever
  (observed as 3 straggler clusters parked in TRYWAIT spins under cuda-gdb).
- ncu --set full crashes on the CLC handshake (error 719 in the SW-counter
  SASS-patching replay); targeted hardware-counter passes profile fine.
- A minimal standalone coverage test of the CLC primitive lives in git history
  (/tmp/clc_test pattern): grid 64x32, atomicAdd coverage, PASS with mixed
  4x4/2x1 grants.

Measured (M=N=8192, K=33024, err=0, EPI 8/NDT 1):

```text
CLC 2x2 along_n SG4:  569.4-570.5us  (new best; persistent 2x2 was 586.9)
CLC 4x1 along_n SG8:  571.1us        (persistent 4x1: 573.6)
CLC 2x1:              575.4us        (persistent 2x1: 590.7)
CLC 4x4 best:         623.7us        (persistent 4x4: 660.3)
reference same day:   0.527 ms, 8.42 PFLOP/s
```

NCU verdict (reports under /tmp/ncu_clc/, hardware-counter passes):
- CLC is tensor-duty NEUTRAL: 55.61% at 2x2 vs 55.4% for the old persistent
  scheduler. Scheduling was never the gap to CUTLASS's 77.5%.
- 4x4 vs 2x2 (+9% time, same 609.3k tensor cycles/SM): 63% of the loss is
  long_scoreboard growth - multicast dedup removes L2-hit refetches (TEX
  sectors -25.5%, hits -61M) and exposes single-miss DRAM latency to all 16
  receivers at once, while cross-pair commit coupling shrinks the effective
  LOAD_PIPE_DEPTH; 37% is steal-granularity tail (SM idle 20k -> 57k cycles,
  148/16 = 9.25 clusters per steal wave). L2 (33-46% peak) and DRAM (28-31%)
  are nowhere near saturated: the dedup saved bandwidth that was not scarce
  and surfaced latency that is.
- Conclusion: 4x4 only pays off with stage recycling decoupled from the
  slowest cluster receiver or a deeper effective pipeline. With Kb locked to
  multiples of 384 (K96 x4-block scales + 64B swizzle floor) and 227KB SMEM,
  deeper staging is blocked in the current TK tile system (see 2026-06-11
  notes). That pipeline restructure - not scheduling - is what separates TK
  (55% duty, 570us) from CUTLASS (77.5% duty, 527-534us).

## 2026-06-11 update: dynamic preferred/fallback clusters landed, bottleneck isolated

The CUTLASS-style dynamic cluster launch is now implemented and correct. Work
assignment is pure blockIdx math over preferred-cluster-aligned regions, so
preferred and fallback clusters compute identical tiles; only multicast scope,
issuer election, and barrier arrive counts depend on the runtime cluster shape
(queried via %cluster_nctaid). Config template gained PREF_M/PREF_N params.
Launch uses LaunchConfig<true,true> with cudaLaunchAttributePreferredClusterDimension
plus fallback cudaLaunchAttributeClusterDimension(2,1,1); __cluster_dims__ removed.
Probe runs showed the driver grants 7x 4x4 preferred clusters + 16x 2x1 fallback
per 144-CTA launch, with err=0 under mixed shapes.

Measured (M=N=8192, K=33024, err=0 everywhere):

```text
reference preferred 4x4:        533 us  (8.32 PFLOP/s)
reference forced 2x1:           627 us
TK old static 2x1 best:         571 us
TK dynamic 2x1 control:         590.7us (matches old along_m/2 exactly; machinery is free)
TK 4x1 SG4 along_m:             573.6us (best dynamic)
TK 2x2:                         586.9us; 2x4: 627.8; 4x4: 660.3; 4x2: 683.5
```

Large preferred clusters lose to wave quantization (128 macros / 9 regions -> 15
units vs 14) plus cross-pair pipeline coupling; CUTLASS avoids this with CLC
hardware work-stealing, which TK does not have.

NCU comparison (reports under /tmp/ncu_nvfp4/): both kernels execute identical
tensor work (609,308 tensor-active cycles/SM, 128 cycles per 256x256x96 UMMA).
The whole gap is tensor-pipe idle: TK 35.5k bubble cycles per output tile vs
CUTLASS 12.8k. TK boosts to 1.91 GHz because of the idle (CUTLASS throttles to
1.55 GHz at 77% duty), which is why the wall gap is only 7%.

Hypotheses killed by direct measurement (do not redo):

```text
Raster/supergroup tuning: flat 660-676 at 4x4; flat 573-576 at 4x1.
Direct register->global epilogue stores: 587us (worse; 4B st.global L2 write
  amplification beats the SMEM bank conflicts it removes). Reverted.
EPI_PIPE_DEPTH=8/NUM_D_TILES=1 (wider TMEM loads + 64B SMEM rows): 572.6us, noise.
Scale TMEM copies (12x tcgen05.cp per stage): skipping them entirely in a
  perf probe changed nothing (573.0us). They are fully hidden.
DRAM/L2 delivery: pinning all A/B tile loads to K block 0 (L2-hot) gave only
  ~5us (568.8). Not bandwidth- or delivery-latency-bound.
```

SASS-level stall attribution (ncu --page source on the same reports, correlated
via nvdisasm; artifacts in /tmp/ncu_nvfp4/) settled the mechanism:

```text
MMA warp: 50.3% of its resident time on wait(tiles_arrived), ~10% scales path,
  outputs_finished only 34 samples (the TMEM-drain wait is NOT the bubble).
Producers: 25% on wait(inputs_finished) (ring full) at the same time.
tcgen05.mma/UTCCP issue sites: no backpressure (~47 samples on 8 UTCOMMA).
TK: 40.5% L2 sector promotion misses vs 0.00% for CUTLASS; +73% L2 read
  sectors. Cause: Kb=384 fp4 tiles have 192B rows -> TK picks 64B TMA swizzle
  -> the 5D TMA box reads 64B gmem atoms, and 192B-offset K slices straddle
  128B lines. CUTLASS reads 128B-aligned full-line K chunks.
```

The bottleneck is the refill ring round trip (tcgen05.commit at MMA-read
completion -> producer mbarrier wake -> TMA issue -> fetch -> complete_tx ->
MMA wake), which marginally exceeds the ring slack of (LOAD_PIPE_DEPTH-1) x 512
= 1536 cycles. This explains why the L2-hot probe barely helped: the sync
legs dominate even when the fetch leg is short.

Additional dead ends measured after the attribution (do not redo):

```text
Kb=768 / MMA_PER_TILE=8 / LOAD_PIPE_DEPTH=2 (fixes 128B alignment, same total
  K buffered): 618.6us; with 2x2/4x2/2x4/4x4 clusters: 616-668us. Two big
  stages have less refill slack (1024 vs 1536 cycles).
cp.async.bulk.prefetch.tensor from idle producer warp 1, paced by the
  inputs_finished ring with LOAD_PIPE_DEPTH lookahead: 1091us (catastrophic;
  prefetch contends with the real TMA stream on this part).
```

Note the K-granularity trap: K96 x4-block scale layout forces Kb % 384 == 0
(96-alignment plus the 64B-swizzle floor); 128B swizzle needs Kb=768, and three
Kb=768 stages need 288KB SMEM > 227KB. Finer-stage CUTLASS-style staging is
impossible in the current TK tile system.

Final state of this session: best config is
config<256, 4, 8, 4, 1, false, false, 4, false, 4, 1>
(Nb=256, LOAD_PIPE_DEPTH=4, EPI_PIPE_DEPTH=8, SUPERGROUP_SIZE=4, NUM_D_TILES=1,
no overlap, along_m, MMA_PER_TILE=4, no global scale, preferred cluster 4x1):

```text
TK:        572.5 us, 7.74 PFLOP/s, err mean/max 0
reference: 0.534 ms, 8.30 PFLOP/s (same day, same GPU)
gap:       7.2 percent
```

Three NVFP4_PERF_PROBE_* ifdef blocks remain in the sample (NO_SCALE_CP,
HOT_LOADS, NO_EPI); they produce WRONG results and exist only to decompose the
bubble. Floor with all probes on: 547.5us.

Remaining credible paths to close the gap, in order:

```text
1. CLC (clusterlaunchcontrol.try_cancel) work stealing as in CUTLASS: removes
   wave quantization for big clusters (so preferred 4x4 with its 4x A / 2x B
   multicast becomes viable) and is what lets CUTLASS run 77% tensor duty.
   This is the structural answer; everything smaller has been measured out.
2. Shorten the sync legs of the refill ring (commit -> wake -> issue): e.g.
   CUTLASS-style try_wait token pre-probing in the MMA warp, or issuing the
   refill TMA from a warp that spins closer to the commit. Expected gain is
   bounded (~10-20us) by the probe floor.
3. The 26us boundary cost (NO_EPI probe): needs CUTLASS's overlapping
   accumulator trick for MMA_N=256; TMEM cannot hold two 256-col accumulators
   plus scales, so this requires epilogue-tile-granular TMEM reuse.
```

The original handoff below is retained for reference; the cubin/instruction
match and validation steps still apply.


This note is for continuing the work to close the performance gap between the
ThunderKittens NVFP4 B300 K96 GEMM sample and the local CUTLASS/tcgen05
reference benchmark.

## Goal

Make `ThunderKittens/kernels/gemm/nvfp4_b300/nvfp4_b300_gemm.cu` match the
performance of the local `tcgen05_sustained` reference for the same GEMM shape
and tcgen05 instruction.

Primary target shape:

```text
M=8192, N=8192, K=33024, batch=1
CTA tile: 256x256x768
tcgen05 instruction: 256x256x96
operation: 3xNVFP4/E2M1 x 3xNVFP4/E2M1 -> F32
scale type: UE8M0, vector size 32
```

Reference command:

```bash
/home/dmoss/scratch/repos/kernels/tcgen05-benchmark/build-x86_64/tcgen05_sustained \
  --m=8192 --n=8192 --k=33024 --warmups=10 --repeats=50 \
  --cluster-m=4 --cluster-n=4 --cluster-k=1 \
  --fallback-cluster-m=2 --fallback-cluster-n=1 --fallback-cluster-k=1 \
  --swizzle=2 --raster=along_m
```

Observed local reference result on GB300/B300:

```text
avg: 0.531 ms, 8.342 PFLOP/s
```

The user also pasted a faster external/reference run:

```text
avg: 0.464 ms, 9.546 PFLOP/s
```

The local comparison point should be the measured local binary unless the next
person reruns and confirms otherwise.

## Current TK state

Branch:

```text
nvfp4-k96-tcgen05
```

Currently modified files:

```text
include/ops/thread/mma/tcgen05.cuh
include/ops/group/mma/tcgen05.cuh
kernels/gemm/nvfp4_b300/nvfp4_b300_gemm.cu
```

Current standalone config in `main()`:

```cpp
run_benchmark<nvfp4_gemm::config<256, 4, 16, 8, 2, false, true, 4, false>>(
    8192, 8192, 33024, ncu);
```

Interpreted as:

```text
Nb=256
LOAD_PIPE_DEPTH=4
EPI_PIPE_DEPTH=16
SUPERGROUP_SIZE=8
NUM_D_TILES=2
OVERLAP_EPI=false
RASTER_ALONG_N=true
MMA_PER_TILE=4
APPLY_GLOBAL_SCALE=false
```

The config now uses a static physical launch cluster of `2x1x1`:

```cpp
static constexpr int CLUSTER_SIZE = 2;
static constexpr int LAUNCH_CLUSTER_M = 2;
static constexpr int LAUNCH_CLUSTER_N = 1;
static constexpr int MMA_GROUPS_M = LAUNCH_CLUSTER_M / CLUSTER_SIZE;
static constexpr int MMA_GROUPS_N = LAUNCH_CLUSTER_N;
```

Current measured TK result for the target shape:

```text
Average kernel execution time: 571.004 us
Throughput: 7.762 PFLOP/s
Correctness: err mean/max 0
```

So the local remaining gap is roughly:

```text
571 us TK vs 531 us tcgen05_sustained, about 7.5 percent slower
```

## Current input initialization

After an attempted CUTLASS-random A/B match, the sample was reverted to the
correctness-friendly input path:

```cpp
fill<uint8_t, FillMode::CONSTANT>(reinterpret_cast<uint8_t*>(d_A[i]), M*K/2, 0x22);
fill<__nv_fp8_e8m0, FillMode::RANDOM>(d_A_sc[i], A_scale_elems, seed + i*100 + 1, 1.0f, 4.0f);
fill<__nv_fp8_e8m0, FillMode::RANDOM>(d_B_sc[i], B_scale_elems, seed + i*100 + 2, 1.0f, 4.0f);
fill<uint8_t, FillMode::CONSTANT>(reinterpret_cast<uint8_t*>(d_B[i]), N*K/2, 0x22);
fill<float, FillMode::CONSTANT>(d_A_sc_global[i], 1, 1.0f);
fill<float, FillMode::CONSTANT>(d_B_sc_global[i], 1, 1.0f);
```

The local `tcgen05_sustained.cu` source initializes differently:

```text
A:   CUTLASS random uniform [-2, 2], seed 2026
SFA: random UE8M0 [1, 4], seed 2027
SFB: random UE8M0 [1, 4], seed 2028
B:   CUTLASS random uniform [-2, 2], seed 2029
```

Do not re-add arbitrary random A/B unless the FP4 packing and scalar reference
semantics are validated first. A previous random-A/B experiment produced
nonzero reference errors and slower kernel time. Constant packed FP4 ones
(`0x22`) with random E8M0 scales is a useful correctness path and currently
gives `err=0`.

## Relevant changes already made

### K96 MMA wrappers

`include/ops/thread/mma/tcgen05.cuh` and
`include/ops/group/mma/tcgen05.cuh` now accept an optional destination CTA mask
on the semaphore-based microscaling K96 paths. This allows pair-local commits
instead of always committing to CTA mask `0b11`.

Representative signature:

```cpp
mm2_ABt_k96(..., semaphore &sem, uint16_t dst_cta_mask = 0b11)
```

The sample passes `pair_mask` into:

```cpp
mm2_ABt_k96(...)
mma2_ABt_k96(...)
tensor_commit<2>(outputs_arrived, pair_mask)
```

### Physical cluster and pair-local ownership

The sample derives:

```cpp
cta_rank = cluster_ctarank()
physical_cluster_size = cluster_nctarank()
cta_x, cta_y
cta_id = cta_x & 1
pair_leader_rank = cta_rank - cta_id
pair_mask = 0b11 << pair_leader_rank
```

For the current static `2x1` cluster this is equivalent to the original
2-CTA group. The logic was generalized enough to experiment with larger
physical clusters, but static `4x4` was much slower.

### Global scale

`APPLY_GLOBAL_SCALE` was added with default `true` so existing configs keep
behavior. The tcgen05-comparison config sets it to `false`.

This is not a meaningful difference from `tcgen05_sustained` for the current
sample because the global scales are initialized to `1.0f`.

## Validation already run

After changing shared tcgen05 wrappers, the focused SM103 group MMA tensor test
passed:

```bash
cd /home/dmoss/scratch/repos/kernels/ThunderKittens/tests
rm -f build/group/mma/tensor/mma.o unit_tests
make unit_tests ARCH=SM103 COMP_LEVEL=fast \
  NVCCFLAGS='-std=c++20 --expt-relaxed-constexpr -Itesting_commons -I../include --extended-lambda -DTEST_INTENSITY=2 -DTEST_GROUP_MMA_TENSOR_MMA -O0 --threads=0 -diag-suppress 20054 -gencode arch=compute_103a,code=sm_103a -DKITTENS_SM103 -lcuda -lcudart'
./unit_tests
```

Result:

```text
52 tests passed, 0 failed
```

The standalone sample was also built and run:

```bash
cd /home/dmoss/scratch/repos/kernels/ThunderKittens/kernels/gemm/nvfp4_b300
make
./nvfp4_b300_gemm.out
```

Result:

```text
Average kernel execution time: 571.004 us
err mean/max 0
```

## Benchmark history and dead ends

Useful measurements from this session:

```text
tcgen05 target, preferred 4x4 fallback 2x1, along_m/2:
  0.531 ms, 8.342 PFLOP/s

TK current best, physical 2x1, along_n/8:
  571.004 us, 7.762 PFLOP/s, err=0

TK physical 2x1, along_m/2:
  about 590.8 us, err=0

tcgen05 forced 2x1, along_m/2:
  0.628 ms, 7.058 PFLOP/s

TK static 4x4, along_m/2:
  about 827 us with overlaunched clusters, err=0
  about 1167 us with grid sized by physical 16-CTA clusters, err=0
```

Supergroup/raster sweep on TK physical `2x1`:

```text
along_m/2:   590.97 us
along_m/4:   578.01 us
along_m/8:   573.43 us
along_m/16:  572.36 us
along_m/32:  598.05 us
along_n/8:   570.93 us
```

Other attempted changes:

```text
LOAD_PIPE_DEPTH=3: about 688 us, worse
EPI_PIPE_DEPTH=8: failed shared-memory static assert
OVERLAP_EPI=true: regressed
Logical 4x4 scheduling over physical 2x1: correct but slower, about 615-645 us
Dynamic preferred/fallback launch experiment: faulted or deadlocked
Random A/B input experiment: correctness mismatch and slower kernel
```

## What CUTLASS/tcgen05 is doing

The local reference binary uses:

```text
KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs32Sm103
```

Instruction atom:

```text
SM103_MXF4_ULTRA_2x1SM_SS_VS
tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block32
instruction shape: 256x256x96
```

Important CUTLASS structural differences observed in source:

1. CUTLASS separates preferred physical cluster shape from fallback cluster
   shape. It launches with required fallback cluster `2x1x1` and a preferred
   cluster such as `4x4x1` through `ClusterLauncher::launch_with_fallback_cluster`.

2. The CUTLASS kernel does not use static `__cluster_dims__`; it queries the
   actual runtime cluster shape using `cute::cluster_shape()`.

3. CUTLASS builds or selects TMA descriptors and multicast behavior for the
   actual cluster shape. Preferred and fallback descriptors exist in the
   mainloop structures.

4. CUTLASS separates the physical cluster shape from the 2-CTA MMA atom
   shape. The atom remains `2x1` even when the preferred physical cluster is
   larger.

5. CUTLASS uses separate mainloop pipelines for A/B tiles and scale-factor
   tiles:

```text
MainloopABPipeline
MainloopSFPipeline
PipelineUmmaAsync for accumulators
```

The TK sample currently has simpler producer/consumer state and a single MMA
warp path that waits on scale tiles, loads scales to TMEM, waits on A/B tiles,
then issues K96 MMA.

6. CUTLASS partitions the TiledMMA slice using global block X parity:

```cpp
TiledMma{}.get_slice(blockIdx.x % size(AtomThrID{}))
```

The TK sample currently uses cluster-local `cta_x & 1`. For static `2x1` this
is equivalent. For dynamic preferred/fallback launch, this may matter because
CUTLASS keeps 2-CTA group ownership tied to global block X parity.

7. CUTLASS TMA multicast masks are built from a cluster layout divided by the
   MMA atom thread shape:

```text
cta_layout_mnk = make_layout(select_cluster_shape(ClusterShape{}, cute::cluster_shape()))
cta_layout_vmnk = tiled_divide(cta_layout_mnk, make_tile(TiledMma::AtomThrID{}))
mcast_mask_a = create_tma_multicast_mask<2>(...)
mcast_mask_b = create_tma_multicast_mask<1>(...)
```

The TK sample has hand-written multicast masks. They reduce correctly for
physical `2x1`, but they are not a complete CUTLASS-equivalent dynamic
preferred/fallback implementation.

## Likely reason for the remaining gap

The instruction itself now matches CUTLASS: both use the SM103 2-CTA NVFP4
Ultra K96 tcgen05 instruction.

The current gap is more likely from scheduling and pipeline structure:

```text
TK tensor pipe utilization is lower than tcgen05/CUTLASS
TK uses static 2x1 launch instead of CUTLASS preferred 4x4/fallback 2x1 launch
TK does not yet model CUTLASS's separate AB/SF pipelines and accumulator pipeline
TK TMEM/barrier ownership is simpler than CUTLASS's per-atom pipeline state
```

Earlier profiling notes indicated TK tensor pipe utilization around `53.7%`
versus tcgen05 around `77%`. Treat those as directional unless rerun with the
current exact binaries.

## Recommended next path

Do not keep tuning small knobs first. The likely fix is to port more of the
CUTLASS scheduling structure into TK:

1. Reproduce CUTLASS launch semantics:

```text
preferred physical cluster: 4x4x1
fallback physical cluster: 2x1x1
MMA atom CTA group: 2x1
runtime actual cluster shape queried in-kernel
```

2. Separate physical cluster shape from MMA CTA-group shape in the TK sample.
   Avoid assuming `cluster_ctarank()` alone defines the MMA pair in all launch
   modes. Verify whether `blockIdx.x % 2` should define the atom slice, as in
   CUTLASS.

3. Add or emulate CUTLASS-equivalent preferred/fallback TMA descriptor behavior.
   A prior dynamic launch attempt faulted/deadlocked, likely because the TK TMA
   descriptors, multicast masks, and ownership mapping were still effectively
   static.

4. Introduce per-2CTA-pair pipeline/barrier state instead of one cluster-global
   set of simplified barriers. The target structure is closer to CUTLASS:

```text
AB tile pipeline
scale-factor tile pipeline
UMMA/accumulator pipeline
TMEM group ownership per 2-CTA atom
```

5. Verify TMEM group ownership explicitly for both actual launch shapes:

```text
actual cluster 2x1 fallback
actual cluster 4x4 preferred, if the driver/GPU grants it
```

6. Only after the mapping is correct, repeat the small sweeps:

```text
raster along_m/2 to match tcgen05 command
raster along_n/8 because it is current TK best
LOAD_PIPE_DEPTH
EPI_PIPE_DEPTH and NUM_D_TILES within shared-memory limits
```

## Useful files to inspect

ThunderKittens:

```text
/home/dmoss/scratch/repos/kernels/ThunderKittens/kernels/gemm/nvfp4_b300/nvfp4_b300_gemm.cu
/home/dmoss/scratch/repos/kernels/ThunderKittens/include/ops/thread/mma/tcgen05.cuh
/home/dmoss/scratch/repos/kernels/ThunderKittens/include/ops/group/mma/tcgen05.cuh
```

Reference:

```text
/home/dmoss/scratch/repos/kernels/tcgen05-benchmark
```

CUTLASS source paths to search from the reference checkout:

```bash
rg -n "KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs32Sm103|SM103_MXF4_ULTRA_2x1SM_SS_VS|launch_with_fallback_cluster|MainloopABPipeline|MainloopSFPipeline|PipelineUmmaAsync|blockIdx.x % size\\(AtomThrID" \
  /home/dmoss/scratch/repos/kernels/tcgen05-benchmark
```

## Commands for a fresh continuation

Fast audit:

```bash
cd /home/dmoss/scratch/repos/kernels/ThunderKittens
git status --short --branch
which nvcc
nvcc --version
```

Run TK sample:

```bash
cd /home/dmoss/scratch/repos/kernels/ThunderKittens/kernels/gemm/nvfp4_b300
make clean
make
./nvfp4_b300_gemm.out
```

Run local reference:

```bash
/home/dmoss/scratch/repos/kernels/tcgen05-benchmark/build-x86_64/tcgen05_sustained \
  --m=8192 --n=8192 --k=33024 --warmups=10 --repeats=50 \
  --cluster-m=4 --cluster-n=4 --cluster-k=1 \
  --fallback-cluster-m=2 --fallback-cluster-n=1 --fallback-cluster-k=1 \
  --swizzle=2 --raster=along_m
```

Run forced-2x1 reference, useful for isolating launch/scheduler effects:

```bash
/home/dmoss/scratch/repos/kernels/tcgen05-benchmark/build-x86_64/tcgen05_sustained \
  --m=8192 --n=8192 --k=33024 --warmups=10 --repeats=50 \
  --cluster-m=2 --cluster-n=1 --cluster-k=1 \
  --fallback-cluster-m=2 --fallback-cluster-n=1 --fallback-cluster-k=1 \
  --swizzle=2 --raster=along_m
```

Run focused SM103 shared-MMA validation after touching wrappers/descriptors:

```bash
cd /home/dmoss/scratch/repos/kernels/ThunderKittens/tests
rm -f build/group/mma/tensor/mma.o unit_tests
make unit_tests ARCH=SM103 COMP_LEVEL=fast \
  NVCCFLAGS='-std=c++20 --expt-relaxed-constexpr -Itesting_commons -I../include --extended-lambda -DTEST_INTENSITY=2 -DTEST_GROUP_MMA_TENSOR_MMA -O0 --threads=0 -diag-suppress 20054 -gencode arch=compute_103a,code=sm_103a -DKITTENS_SM103 -lcuda -lcudart'
./unit_tests
```

## Pitfalls

Avoid these unless there is new evidence:

```text
Do not switch to one K96 stage. The current shared tile layout wants Kb=384 with MMA_PER_TILE=4.
Do not raise LOAD_PIPE_DEPTH to 5 without checking shared-memory limits.
Do not treat APPLY_GLOBAL_SCALE=false as the real performance fix.
Do not assume random A/B correctness until CUTLASS FP4 packing/reference semantics are matched.
Do not judge static 4x4 as equivalent to CUTLASS dynamic preferred/fallback launch.
Do not remove pair-local mask plumbing from K96 wrappers; it is needed for larger-cluster experiments.
```

## Definition of done

The next implementation should be considered successful only when:

```text
TK uses the same SM103 K96 2-CTA tcgen05 instruction as tcgen05_sustained.
TK correctness remains err=0 on the constant-A/B scale-random path.
Shared tcgen05 K96 tests pass on SM103 after wrapper or descriptor changes.
For M=N=8192, K=33024, TK is within measurement noise of local tcgen05_sustained.
The benchmark report includes the exact TK command, reference command, GPU, CUDA/NVCC, correctness, and timing.
```
