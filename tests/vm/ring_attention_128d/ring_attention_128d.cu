#include "kittens.cuh"
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

constexpr int SM_COUNT = 148;
constexpr int QO_BLOCK_SIZE = 128; // sequence length must be divisible by this * 2
constexpr int KV_BLOCK_SIZE = 128; // sequence length must be divisible by this
constexpr int HEAD_DIM = 128;
constexpr int COMM_CHUNK_LENGTH = 64;

using q_tile = st_bf<QO_BLOCK_SIZE, HEAD_DIM>;
using k_tile = st_bf<KV_BLOCK_SIZE/2, HEAD_DIM>; // Split for 2-CTA cooperative matmuls
using v_tile = st_bf<KV_BLOCK_SIZE, HEAD_DIM/2>; // Split for 2-CTA cooperative matmuls
using ao_tile = st_bf<QO_BLOCK_SIZE, HEAD_DIM>; // Only possible since HEAD_DIM == KV_BLOCK_SIZE
using lm_vec = col_vec<st_fl<QO_BLOCK_SIZE, HEAD_DIM>>;
using comm_tile = st_bf<COMM_CHUNK_LENGTH, HEAD_DIM>; // Full 16KB page

struct ring_attention_config
{
    // Ring-attention specific
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2; // 12 pages are used for ring attention
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;
    static constexpr int NUM_CONSUMER_WARPS = 16;
    static constexpr int CLUSTER_BLOCKS = 2; // for 2-CTA cooperative matmuls
    static constexpr int SCRATCH_BYTES = 2048; // need at least 2048
    static constexpr int CONSUMER_REGISTERS = 96;
    static constexpr int NON_CONSUMER_REGISTERS = 96;

    // Same as default
    static constexpr int INSTRUCTION_WIDTH = 32;
    using instruction_t = int[INSTRUCTION_WIDTH];
    static constexpr int TIMING_WIDTH = 128;
    using timing_t = int[TIMING_WIDTH];
    static constexpr int DYNAMIC_SEMAPHORES = 32;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS = 1; // unused
    static constexpr int MAX_SHARED_MEMORY = kittens::MAX_SHARED_MEMORY;
    static constexpr int STATIC_SHARED_MEMORY = 1024 + INSTRUCTION_PIPELINE_STAGES * (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 + DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;
    static constexpr int PAGE_SIZE = 16384;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES == 13, "NUM_PAGES must be 13");
};
using config = ring_attention_config;

struct globals {
    constexpr static int num_devices = 4;

    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using barrier_layout = gl<uint, 1, 1, 1, num_devices>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;

    using q_layout = gl<bf16, -1, -1, -1, HEAD_DIM, q_tile>; // Batch, Head, Seq, Dim (full MHA)
    using k_layout = gl<bf16, -1, -1, -1, HEAD_DIM, k_tile, comm_tile>;
    using v_layout = gl<bf16, -1, -1, -1, HEAD_DIM, v_tile, comm_tile>;
    using o_layout = gl<bf16, -1, -1, -1, HEAD_DIM, ao_tile>;
    using lm_layout = gl<float, 1, -1, -1, -1, lm_vec>; // Batch, Head, Seq

    instruction_layout instructions;
    gl_array<barrier_layout, num_devices> barriers;
    timing_layout timings;

    q_layout Q; // local Q sharded on sequence dimension
    gl_array<k_layout, num_devices> K0s;
    gl_array<k_layout, num_devices> K1s;
    gl_array<v_layout, num_devices> V0s;
    gl_array<v_layout, num_devices> V1s;
    o_layout O;
    lm_layout L;
    lm_layout M;

    dim3 grid() { return dim3(SM_COUNT); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct RingAttentionOp {
    static constexpr int opcode = 725;
    static constexpr int PIPELINE_STAGES = 2;
    static constexpr int NUM_CONSUMERS = 2;
    static constexpr int WARPS_PER_CONSUMER = 8;

    static_assert(NUM_CONSUMERS == 2);
    static_assert(WARPS_PER_CONSUMER == 8);
    static_assert(config::NUM_CONSUMER_WARPS == WARPS_PER_CONSUMER*NUM_CONSUMERS, 
                  "RingAttentionOp requires 16 consumer warpgroups.");

    struct parsed_instruction {
        int B;             // batch index              (in units of 1)
        int H;             // head index               (in units of 1)
        int QO_idx;        // Q block index      (in units of `QO_BLOCK_SIZE * 4` tokens)
        int num_kv_blocks; // # of KV blocks to handle (in units of `KV_BLOCK_SIZE` tokens)
        int ring_stage;    // current ring stage index (0, 1, ..., NUM_DEVS - 1)
        int num_comms;     // number of SMs doing comms
        int num_comps;     // number of instructions per ring stage per device
        int dev_idx;       // current device index     (0, 1, ..., NUM_DEVS - 1)
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            B = instruction[1];
            H = instruction[2];
            QO_idx = instruction[3];
            num_kv_blocks = instruction[4];
            ring_stage = instruction[5];
            num_comms = instruction[6];
            num_comps = instruction[7];
            dev_idx = instruction[8];
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };

    // Semaphores
    __device__ static inline semaphore &lm_arrived(state<config> &s, int consumer_id)  { return s.semaphores()[0 + consumer_id]; }
    __device__ static inline semaphore &lm_finished(state<config> &s, int consumer_id) { return s.semaphores()[2 + consumer_id]; }
    __device__ static inline semaphore &q_arrived(state<config> &s, int consumer_id)   { return s.semaphores()[4 + consumer_id]; }
    __device__ static inline semaphore &o_arrived(state<config> &s, int consumer_id)   { return s.semaphores()[6 + consumer_id]; }
    __device__ static inline semaphore &o_finished(state<config> &s, int consumer_id)  { return s.semaphores()[8 + consumer_id]; }
    __device__ static inline semaphore &qk_unloaded(state<config> &s, int consumer_id) { return s.semaphores()[10 + consumer_id]; }
    __device__ static inline semaphore &av_ready(state<config> &s, int consumer_id)    { return s.semaphores()[12 + consumer_id]; }
    __device__ static inline semaphore &qk_finished(state<config> &s, int consumer_id) { return s.semaphores()[14 + consumer_id]; }
    __device__ static inline semaphore &av_finished(state<config> &s, int consumer_id) { return s.semaphores()[16 + consumer_id]; }
    __device__ static inline semaphore &k_arrived(state<config> &s, int stage)         { return s.semaphores()[18 + PIPELINE_STAGES * 0 + stage]; }
    __device__ static inline semaphore &v_arrived(state<config> &s, int stage)         { return s.semaphores()[18 + PIPELINE_STAGES * 1 + stage]; }
    __device__ static inline semaphore &k_finished(state<config> &s, int stage)        { return s.semaphores()[18 + PIPELINE_STAGES * 2 + stage]; }
    __device__ static inline semaphore &v_finished(state<config> &s, int stage)        { return s.semaphores()[18 + PIPELINE_STAGES * 3 + stage]; }

    // Pages (use PIDs for contiguity)
    __device__ static inline int get_q_page(state<config> &s, int consumer_id)  { return 0 + consumer_id*2; } // 32KB each
    __device__ static inline int get_k_page(state<config> &s, int stage)        { return 4 + PIPELINE_STAGES * 0 + stage; }
    __device__ static inline int get_v_page(state<config> &s, int stage)        { return 4 + PIPELINE_STAGES * 1 + stage; }
    __device__ static inline int get_ao_page(state<config> &s, int consumer_id) { return 8 + consumer_id*2; } // 32KB each

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            int lids[config::NUM_PAGES] = {12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
            return lids[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            for (int i = 0; i < NUM_CONSUMERS; ++i) {
                init_semaphore(q_arrived(s, i), 0, 2); // 2 CTAs
                init_semaphore(qk_unloaded(s, i), 0, WARPS_PER_CONSUMER * 2); // 8 warps per consumer * 2 CTAs
                init_semaphore(av_ready(s, i), 0, WARPS_PER_CONSUMER * 2);    // 8 warps per consumer * 2 CTAs
                init_semaphore(qk_finished(s, i), 0, 1);
                init_semaphore(av_finished(s, i), 0, 1);
                init_semaphore(o_arrived(s, i), 0, 1);
                init_semaphore(lm_arrived(s, i), 0, 1);
                init_semaphore(o_finished(s, i), 0, WARPS_PER_CONSUMER);  // 8 warps per consumer
                init_semaphore(lm_finished(s, i), 0, WARPS_PER_CONSUMER); // 8 warps per consumer
            }
            for (int i = 0; i < PIPELINE_STAGES; ++i) {
                init_semaphore(k_arrived(s, i), 0, 2);  // 2 CTAs
                init_semaphore(v_arrived(s, i), 0, 2);  // 2 CTAs
                init_semaphore(k_finished(s, i), 0, NUM_CONSUMERS); // 2 consumers
                init_semaphore(v_finished(s, i), 0, NUM_CONSUMERS); // 2 consumers
            }
            return NUM_CONSUMERS*9 + PIPELINE_STAGES*4;
        }
    };

    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int ctarank = cluster_ctarank();
            int laneid = warp::laneid();

            // Wait for the previous ring stage to finish
            while (inst.ring_stage > 0 && *(volatile int *)&g.barriers[inst.dev_idx][{inst.ring_stage - 1}] < inst.num_comms + inst.num_comps)
                __nanosleep(20);

            if (laneid < 2) { // Load Q for the 2 consumers
                int consumer_id = laneid;
                int local_QO_idx = inst.QO_idx + NUM_CONSUMERS*ctarank + consumer_id; // inst.QO_idx is given by units of 512
                auto q_page = get_q_page(s, consumer_id);
                auto &q = *reinterpret_cast<q_tile *>(s.pages[q_page].data);
                s.wait_page_ready(q_page);
                s.wait_page_ready(q_page + 1);
                tma::cluster::expect(q_arrived(s, consumer_id), 0, q);
                tma::cluster::load_async(q, g.Q, {inst.B, inst.H, local_QO_idx, 0}, q_arrived(s, consumer_id), (uint16_t)(1<<ctarank), 0);
            } else if (laneid == 2) { // Load Ks
                uint32_t phasebit = 0;
                for (int i = 0; i < inst.num_kv_blocks; i++) {
                    int stage = i % PIPELINE_STAGES;
                    auto k_page = get_k_page(s, stage);
                    auto &k = *reinterpret_cast<k_tile *>(s.pages[k_page].data);
                    if (i < PIPELINE_STAGES) {
                        s.wait_page_ready(get_k_page(s, stage));
                    } else { // no need for cluster wait since this means matmul complete
                        wait(k_finished(s, stage), get_phasebit<0>(phasebit, stage));
                        update_phasebit<0>(phasebit, stage);
                    }
                    tma::cluster::expect(k_arrived(s, stage), 0, k);
                    if ((inst.ring_stage & 1) == 0) // 64 x 128
                        tma::cluster::load_async(k, g.K0s[inst.dev_idx], {inst.B, inst.H, i*2 + ctarank, 0}, k_arrived(s, stage), (uint16_t)(1<<ctarank), 0);
                    else
                        tma::cluster::load_async(k, g.K1s[inst.dev_idx], {inst.B, inst.H, i*2 + ctarank, 0}, k_arrived(s, stage), (uint16_t)(1<<ctarank), 0);
                }
                for (int i = 0; i < PIPELINE_STAGES; i++) {
                    int stage = (i + inst.num_kv_blocks) % PIPELINE_STAGES;
                    wait(k_finished(s, stage), get_phasebit<0>(phasebit, stage));
                    s.finish_page(get_k_page(s, stage), config::NUM_CONSUMER_WARPS);
                    update_phasebit<0>(phasebit, stage);
                }
            } else if (laneid == 3) { // Load Vs
                uint32_t phasebit = 0;
                for (int i = 0; i < inst.num_kv_blocks; i++) {
                    int stage = i % PIPELINE_STAGES;
                    auto v_page = get_v_page(s, stage);
                    auto &v = *reinterpret_cast<v_tile *>(s.pages[v_page].data);
                    if (i < PIPELINE_STAGES) {
                        s.wait_page_ready(get_v_page(s, stage));
                    } else {
                        wait(v_finished(s, stage), get_phasebit<0>(phasebit, stage));
                        update_phasebit<0>(phasebit, stage);
                    }
                    tma::cluster::expect(v_arrived(s, stage), 0, v);
                    if ((inst.ring_stage & 1) == 0) // 128 x 64
                        tma::cluster::load_async(v, g.V0s[inst.dev_idx], {inst.B, inst.H, i, ctarank}, v_arrived(s, stage), (uint16_t)(1<<ctarank), 0);
                    else
                        tma::cluster::load_async(v, g.V1s[inst.dev_idx], {inst.B, inst.H, i, ctarank}, v_arrived(s, stage), (uint16_t)(1<<ctarank), 0);
                }
                for (int i = 0; i < PIPELINE_STAGES; i++) {
                    int stage = (i + inst.num_kv_blocks) % PIPELINE_STAGES;
                    wait(v_finished(s, stage), get_phasebit<0>(phasebit, stage));
                    s.finish_page(get_v_page(s, stage), config::NUM_CONSUMER_WARPS);
                    update_phasebit<0>(phasebit, stage);
                }
            } else if (laneid < 6) { // Load Os for the 2 consumers
                int consumer_id = laneid - 4;
                int local_QO_idx = inst.QO_idx + NUM_CONSUMERS*ctarank + consumer_id;
                auto ao_page = get_ao_page(s, consumer_id);
                auto &out = *reinterpret_cast<ao_tile *>(s.pages[ao_page].data);
                s.wait_page_ready(ao_page);
                s.wait_page_ready(ao_page + 1);
                if (inst.ring_stage == 0) {
                    arrive(o_arrived(s, consumer_id));
                } else {
                    tma::expect(o_arrived(s, consumer_id), out);
                    tma::load_async(out, g.O, {inst.B, inst.H, local_QO_idx, 0}, o_arrived(s, consumer_id));
                }
            } else if (laneid < 8) { // Load Ls and Ms for the 2 consumers
                int consumer_id = laneid - 6;
                int local_QO_idx = inst.QO_idx + NUM_CONSUMERS*ctarank + consumer_id;
                auto &l = *(reinterpret_cast<lm_vec *>(
                    reinterpret_cast<char *>(s.scratch()) + ((2*consumer_id+0)*sizeof(lm_vec))
                )); // share 1 page between 2 consumers for Ls and Ms
                auto &m = *(reinterpret_cast<lm_vec *>(
                    reinterpret_cast<char *>(s.scratch()) + ((2*consumer_id+1)*sizeof(lm_vec))
                ));
                if (inst.ring_stage == 0) {
                    arrive(lm_arrived(s, consumer_id));
                } else {
                    tma::expect(lm_arrived(s, consumer_id), l, m);
                    tma::load_async(l, g.L, {inst.B, inst.H, local_QO_idx}, lm_arrived(s, consumer_id));
                    tma::load_async(m, g.M, {inst.B, inst.H, local_QO_idx}, lm_arrived(s, consumer_id));
                }
            } else if (12 <= laneid && laneid < config::NUM_PAGES) { // Release unused pages
                s.wait_page_ready(laneid);
                s.finish_page(laneid, config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int ctarank = cluster_ctarank();
            int laneid = warp::laneid();

            // Nothing is ready until the tensor cores are ready
            s.wait_tensor_ready();

            if (ctarank == 0) {
                if (laneid < 2) { // Launch Q @ K^T for the 2 consumers
                    int consumer_id = laneid;
                    auto q_page = get_q_page(s, consumer_id);
                    auto &q = *reinterpret_cast<q_tile *>(s.pages[q_page].data);
                    wait(q_arrived(s, consumer_id), 0);
                    
                    uint32_t phasebit = 0;
                    for (int i = 0; i < inst.num_kv_blocks; ++i) {
                        int stage = i % PIPELINE_STAGES;
                        if (i > 0) {
                            tma::cluster::wait(qk_unloaded(s, consumer_id), get_phasebit<1>(phasebit, consumer_id));
                            update_phasebit<1>(phasebit, consumer_id);
                        }
                        auto k_page = get_k_page(s, stage);
                        auto &k = *reinterpret_cast<k_tile *>(s.pages[k_page].data);
                        tma::cluster::wait(k_arrived(s, stage), get_phasebit<0>(phasebit, stage));
                        update_phasebit<0>(phasebit, stage);
                        auto qk_accumulator = s.tensor_alloc.template allocate<tt<float, QO_BLOCK_SIZE, KV_BLOCK_SIZE>>(consumer_id*KV_BLOCK_SIZE);
                        mm2_ABt(qk_accumulator, q, k, qk_finished(s, consumer_id));
                    }
                } else if (laneid < 4) { // Launch ATT @ V for the 2 consumers
                    int consumer_id = laneid-2;
                    auto ao_page = get_ao_page(s, consumer_id);
                    auto &att = *reinterpret_cast<ao_tile *>(s.pages[ao_page].data);

                    uint32_t phasebit = 0;
                    for (int i = 0; i < inst.num_kv_blocks; ++i) {
                        int stage = i % PIPELINE_STAGES;
                        auto v_page = get_v_page(s, stage);
                        auto &v = *reinterpret_cast<v_tile *>(s.pages[v_page].data);
                        tma::cluster::wait(v_arrived(s, stage), get_phasebit<0>(phasebit, stage));
                        update_phasebit<0>(phasebit, stage);
                        tma::cluster::wait(av_ready(s, consumer_id), get_phasebit<1>(phasebit, consumer_id));
                        update_phasebit<1>(phasebit, consumer_id);
                        auto av_accumulator = s.tensor_alloc.template allocate<tt<float, QO_BLOCK_SIZE, HEAD_DIM>>(2*KV_BLOCK_SIZE+(consumer_id)*HEAD_DIM);
                        mma2_AB(av_accumulator, att, v, av_finished(s, consumer_id));
                    }
                }
            }
        }
    };

    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            using all_consumers = group<NUM_CONSUMERS*WARPS_PER_CONSUMER>;
            using consumer = group<WARPS_PER_CONSUMER>;
            
            constexpr float softmax_temp = 0.08838834764831843f * 1.44269504089f; // 1 / {sqrt(HEAD_DIM=128) * ln(2)}

            parsed_instruction inst{s};
            int consumer_id = consumer::groupid();

            col_vec<rt_fl<QO_BLOCK_SIZE / WARPS_PER_CONSUMER, KV_BLOCK_SIZE>> max_vec;
            col_vec<rt_fl<QO_BLOCK_SIZE / WARPS_PER_CONSUMER, KV_BLOCK_SIZE>> scaled_max_vec;
            col_vec<rt_fl<QO_BLOCK_SIZE / WARPS_PER_CONSUMER, KV_BLOCK_SIZE>> last_scaled_max_vec;
            col_vec<rt_fl<QO_BLOCK_SIZE / WARPS_PER_CONSUMER, KV_BLOCK_SIZE>> diff_scaled_max_vec;
            col_vec<rt_fl<QO_BLOCK_SIZE / WARPS_PER_CONSUMER, HEAD_DIM>> norm_vec;

            auto ao_page = get_ao_page(s, consumer_id);
            auto &att = *reinterpret_cast<ao_tile *>(s.pages[ao_page].data);
            auto &l = *(reinterpret_cast<lm_vec *>(
                reinterpret_cast<char *>(s.scratch()) + ((2*consumer_id+0)*sizeof(lm_vec))
            )); // share 1 page between 2 consumers for Ls and Ms
            auto &m = *(reinterpret_cast<lm_vec *>(
                reinterpret_cast<char *>(s.scratch()) + ((2*consumer_id+1)*sizeof(lm_vec))
            ));

            wait(o_arrived(s, consumer_id), 0);
            wait(lm_arrived(s, consumer_id), 0);

            auto qk_accumulator = s.tensor_alloc.template allocate<tt<float, QO_BLOCK_SIZE, KV_BLOCK_SIZE>>(consumer_id*KV_BLOCK_SIZE);
            auto av_accumulator = s.tensor_alloc.template allocate<tt<float, QO_BLOCK_SIZE, HEAD_DIM>>(2*KV_BLOCK_SIZE + consumer_id*HEAD_DIM);

            if (inst.ring_stage == 0) {
                warp::neg_infty(max_vec);
                warp::zero(last_scaled_max_vec); // correct as long as not +-inf
                warp::zero(norm_vec);
            } else {
                // Continue from the previous ring stage
                consumer::load(max_vec, m);
                consumer::load(norm_vec, l); // note that l is not an LSE until the last stage
                warp::mul(last_scaled_max_vec, max_vec, softmax_temp);
                rt_fl<QO_BLOCK_SIZE / WARPS_PER_CONSUMER, KV_BLOCK_SIZE> att_fl;
                consumer::load(att_fl, att);
                consumer::store_async(av_accumulator, att_fl);
                tensor_store_wait();
                __syncwarp();
            }

            uint32_t phasebit = 0;
            for (int i = 0; i < inst.num_kv_blocks; ++i) {
                int stage = i % PIPELINE_STAGES;

                // Read in QK^T
                tma::cluster::wait(qk_finished(s, consumer_id), get_phasebit<0>(phasebit, 0)); // wait for mm to finish
                update_phasebit<0>(phasebit, 0);
                if (consumer::laneid() == 0) {
                    arrive(k_finished(s, stage));
                    if (i == inst.num_kv_blocks - 1) {
                        s.finish_page(get_q_page(s, consumer_id), config::NUM_CONSUMER_WARPS);
                        s.finish_page(get_q_page(s, consumer_id) + 1, config::NUM_CONSUMER_WARPS);
                    }
                }
                rt_fl<QO_BLOCK_SIZE / WARPS_PER_CONSUMER, KV_BLOCK_SIZE> att_fl;
                consumer::load_async(att_fl, qk_accumulator);
                tensor_load_wait();
                __syncwarp();
                if (warp::laneid() == 0) tma::cluster::arrive(qk_unloaded(s, consumer_id), 0); // must arrive per warp

                // Start softmax
                // Get maximums and scale by softmax temp
                warp::row_max(max_vec, att_fl, max_vec);
                warp::mul(att_fl, att_fl, softmax_temp);
                warp::mul(scaled_max_vec, max_vec, softmax_temp);

                // Compute softmax numerator
                warp::sub_row(att_fl, att_fl, scaled_max_vec);
                warp::exp2(att_fl, att_fl);

                // Compute normalizer
                warp::sub(diff_scaled_max_vec, last_scaled_max_vec, scaled_max_vec);
                warp::exp2(diff_scaled_max_vec, diff_scaled_max_vec);
                warp::copy(last_scaled_max_vec, scaled_max_vec); // save for next iteration

                // Normalize and accumulate softmax denominator
                warp::mul(norm_vec, norm_vec, diff_scaled_max_vec);
                warp::row_sum(norm_vec, att_fl, norm_vec);

                // Prepare for AV
                if (i == 0) {
                    consumer::store(att, att_fl);
                    if (inst.ring_stage == 0) {
                        warp::zero(att_fl);
                    } else {
                        consumer::load_async(att_fl, av_accumulator);
                        tensor_load_wait();
                        __syncwarp();
                    }
                } else { // must accumulate on the previous AV matmul
                    tma::cluster::wait(av_finished(s, consumer_id), get_phasebit<1>(phasebit, 0));
                    update_phasebit<1>(phasebit, 0);
                    int prev_stage = (i + PIPELINE_STAGES - 1) % PIPELINE_STAGES;
                    if (consumer::laneid() == 0) arrive(v_finished(s, prev_stage));
                    consumer::store(att, att_fl);
                    consumer::load_async(att_fl, av_accumulator);
                    tensor_load_wait(); // TODO: is this needed?
                    __syncwarp();       // TODO: is this needed?
                }
                warp::mul_row(att_fl, att_fl, diff_scaled_max_vec); // normalize previous outputs
                consumer::store_async(av_accumulator, att_fl);
                tensor_store_wait();
                __syncwarp();
                if (warp::laneid() == 0) tma::cluster::arrive(av_ready(s, consumer_id), 0); // must arrive per warp
                consumer::sync(consumer_id); // TODO: is this needed?
            }

            // Wait for the last AV matmul to finish and load the final output
            tma::cluster::wait(av_finished(s, consumer_id), get_phasebit<1>(phasebit, 0));
            if (consumer::laneid() == 0) arrive(v_finished(s, (inst.num_kv_blocks - 1) % PIPELINE_STAGES));
            rt_fl<QO_BLOCK_SIZE / WARPS_PER_CONSUMER, KV_BLOCK_SIZE> att_fl;
            consumer::load_async(att_fl, av_accumulator);
            tensor_load_wait(); // TODO: is this needed?
            __syncwarp();       // TODO: is this needed?
            warp::arrive(s.tensor_finished);

            // Finish softmax if last ring stage
            if (inst.ring_stage == globals::num_devices - 1) {
                warp::div_row(att_fl, att_fl, norm_vec);
                // Make LSE
                warp::log2(norm_vec, norm_vec);
                warp::add(norm_vec, norm_vec, last_scaled_max_vec);
            }

            // Store the outputs and signal the storer
            consumer::store(att, att_fl);
            warp::arrive(o_finished(s, consumer_id));
            consumer::store(l, norm_vec);
            consumer::store(m, max_vec);
            warp::arrive(lm_finished(s, consumer_id));
        }
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int ctarank = cluster_ctarank();
            int laneid = warp::laneid();

            if (laneid < 2) { // store Os
                int consumer_id = laneid;
                int local_QO_idx = inst.QO_idx + NUM_CONSUMERS*ctarank + consumer_id;
                int ao_page = get_ao_page(s, consumer_id);
                auto &out = *reinterpret_cast<ao_tile *>(s.pages[ao_page].data);
                wait(o_finished(s, consumer_id), 0);
                tma::store_async(g.O, out, {inst.B, inst.H, local_QO_idx, 0});
                tma::store_async_read_wait();
                s.finish_page(ao_page, config::NUM_CONSUMER_WARPS);
                s.finish_page(ao_page + 1, config::NUM_CONSUMER_WARPS);
            } else if (laneid < 4) { // store Ls and Ms
                int consumer_id = laneid-2;
                int local_QO_idx = inst.QO_idx + NUM_CONSUMERS*ctarank + consumer_id;
                auto &l = *(reinterpret_cast<lm_vec *>(
                    reinterpret_cast<char *>(s.scratch()) + ((2*consumer_id+0)*sizeof(lm_vec))
                )); // share 1 page between 2 consumers for Ls and Ms
                auto &m = *(reinterpret_cast<lm_vec *>(
                    reinterpret_cast<char *>(s.scratch()) + ((2*consumer_id+1)*sizeof(lm_vec))
                ));
                wait(lm_finished(s, consumer_id), 0);
                tma::store_async(g.L, l, {inst.B, inst.H, local_QO_idx});
                tma::store_async(g.M, m, {inst.B, inst.H, local_QO_idx});
            }
            __syncwarp();
            if (laneid == 0) {
                tma::store_async_wait();
                atomicAdd_system(&g.barriers[inst.dev_idx][{inst.ring_stage}], 1); // signal the next ring stage comms
            }
        }
    };
};

template<typename config=config> struct CommOp {
    static constexpr int opcode = 97;

    struct parsed_instruction {
        int k_or_v;     // 0 for K, 1 for V
        int num_chunks; // number of chunks per SM
        int comm_idx;
        int num_comms;
        int num_comps;
        int num_chunks_N; // number of chunks in sequence dimension
        int num_chunks_H; // number of chunks in head dimension
        int dev_idx;
        int prev_dev_idx;
        int next_dev_idx;
        int ring_stage;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            k_or_v = instruction[1];
            num_chunks = instruction[2];
            comm_idx = instruction[3];
            num_comms = instruction[4]; // total number of SMs doing comms (includes all K & V comms)
            num_comps = instruction[5];
            num_chunks_N = instruction[6];
            num_chunks_H = instruction[7];
            dev_idx = instruction[8];
            prev_dev_idx = instruction[9];
            next_dev_idx = instruction[10];
            ring_stage = instruction[11];
        }
        __device__ inline parsed_instruction(state<config> &s): parsed_instruction(s.instruction()) {}
    };

    __device__ static inline semaphore &data_arrived(state<config> &s, int idx) { return s.semaphores()[idx]; }
    __device__ static inline semaphore &data_finished(state<config> &s, int idx) { return s.semaphores()[idx + config::NUM_PAGES]; }

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            int release_order[config::NUM_PAGES] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
            return release_order[query];
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s) {
            for(int i = 0; i < config::NUM_PAGES; i++) {
                init_semaphore(data_arrived(s, i), 1);
                init_semaphore(data_finished(s, i), 1);
                arrive(data_finished(s, i)); // arrive first
            }
            return 2 * config::NUM_PAGES;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) { }
    };
    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) { 
            s.wait_tensor_ready();
            if (laneid() == 0) arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
        }
    };
    struct consumer { // use consumer warps for more registers
        static __device__ void run(const globals &g, state<config> &s) {
            if (group<config::NUM_CONSUMER_WARPS>::warpid() != 0) return;
            parsed_instruction inst{s};
            int laneid = warp::laneid();
            constexpr uint32_t membermask = 0xFFFFFFFF >> (32 - config::NUM_PAGES);
            if (laneid < config::NUM_PAGES) {
                int page = s.pid(laneid);
                s.wait_page_ready(page);
                auto &data = reinterpret_cast<comm_tile &>(s.pages[page]);
                int phasebit = 0;
                int iters = (inst.num_chunks + config::NUM_PAGES - 1) / config::NUM_PAGES;

                // Are we ready to move on? == previous device's store done + current device's compute done + next device's load done
                // TODO: technically, I can put the latter two conditions before the first store
                while (inst.ring_stage > 0 && 
                        (*(volatile int *)&g.barriers[inst.prev_dev_idx][{inst.ring_stage - 1}] < inst.num_comms + inst.num_comps ||
                        *(volatile int *)&g.barriers[inst.dev_idx     ][{inst.ring_stage - 1}] < inst.num_comms + inst.num_comps ||
                        *(volatile int *)&g.barriers[inst.next_dev_idx][{inst.ring_stage - 1}] < inst.num_comms + inst.num_comps))
                    __nanosleep(20);

                for (int i = 0; i < iters; ++i) {
                    int local_index = i * config::NUM_PAGES + laneid;
                    if (local_index < inst.num_chunks) {
                        int index = inst.comm_idx * inst.num_chunks + local_index;
                        int B_idx = (index / inst.num_chunks_N / inst.num_chunks_H);
                        int H_idx = (index / inst.num_chunks_N) % inst.num_chunks_H;
                        int N_idx = index % inst.num_chunks_N;
                        kittens::tma::expect(data_arrived(s, laneid), data);
                        if (inst.k_or_v == 0) {
                            if (inst.ring_stage % 2 == 0)
                                kittens::tma::load_async(data, g.K0s[inst.prev_dev_idx], {B_idx, H_idx, N_idx, 0}, data_arrived(s, laneid));
                            else
                                kittens::tma::load_async(data, g.K1s[inst.prev_dev_idx], {B_idx, H_idx, N_idx, 0}, data_arrived(s, laneid));
                        } else {
                            if (inst.ring_stage % 2 == 0)
                                kittens::tma::load_async(data, g.V0s[inst.prev_dev_idx], {B_idx, H_idx, N_idx, 0}, data_arrived(s, laneid));
                            else
                                kittens::tma::load_async(data, g.V1s[inst.prev_dev_idx], {B_idx, H_idx, N_idx, 0}, data_arrived(s, laneid));
                        }
                        wait(data_arrived(s, laneid), phasebit);
                        phasebit ^= 1;
                        if (inst.k_or_v == 0) {
                            if (inst.ring_stage % 2 == 0)
                                kittens::tma::store_async(g.K1s[inst.dev_idx], data, {B_idx, H_idx, N_idx, 0});
                            else
                                kittens::tma::store_async(g.K0s[inst.dev_idx], data, {B_idx, H_idx, N_idx, 0});
                        } else {
                            if (inst.ring_stage % 2 == 0)
                                kittens::tma::store_async(g.V1s[inst.dev_idx], data, {B_idx, H_idx, N_idx, 0});
                            else
                                kittens::tma::store_async(g.V0s[inst.dev_idx], data, {B_idx, H_idx, N_idx, 0});
                        }
                    }
                    asm volatile("{bar.warp.sync %0;}" ::"n"(membermask));
                    if (laneid == 0) asm volatile("{cp.async.bulk.wait_group 0;}");
                    asm volatile("{bar.warp.sync %0;}" ::"n"(membermask));
                }
                if (laneid == 0) {
                    asm volatile("{fence.acq_rel.sys;}");
                    atomicAdd_system(&g.barriers[inst.dev_idx][{inst.ring_stage}], 1); // mark store finished
                }

                s.finish_page(page, config::NUM_CONSUMER_WARPS); 
            }
        }
    };
    struct storer {
        static __device__ void run(const globals &g, state<config> &s) { }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(ring_attention, m) {
    m.doc() = "ring attention python module";
    kittens::py::bind_kernel<kvm<config, globals,
        RingAttentionOp<config>,
        CommOp<config>
    >>(m, "ring_attention",
        &globals::instructions,
        &globals::barriers,
        &globals::timings,
        &globals::Q,
        &globals::K0s,
        &globals::K1s,
        &globals::V0s,
        &globals::V1s,
        &globals::O,
        &globals::L,
        &globals::M
    );
}
