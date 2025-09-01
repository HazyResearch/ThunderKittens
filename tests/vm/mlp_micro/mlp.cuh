#pragma once

#include "kittens.cuh"
#include "vm/vm.cuh"
#include "vm/util.cuh"
#include <iostream>

#define OPCODE_Up 1
#define OPCODE_Down 2

namespace kittens::prototype::vm {

template<int _pipeline_stages>
struct mlp_config {
    // Instruction pipeline
    static constexpr int INSTRUCTION_PIPELINE_STAGES = _pipeline_stages;

    // num bits required to represent num pipeline stages
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;

    static constexpr int INSTRUCTION_WIDTH = 32; // 128 bytes per instruction.
    using instruction_t = int[INSTRUCTION_WIDTH];

    // Timing info
    static constexpr int TIMING_WIDTH = 128;
    using timing_t = int[TIMING_WIDTH];

    // How many semaphores are available for dynamic use?
    static constexpr int DYNAMIC_SEMAPHORES = 32;

    // One controller warp, one load warp, one store warp, and one mma warp.
    static constexpr int NUM_CONSUMER_WARPS = 16;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int CLUSTER_BLOCKS = 1;
    static constexpr int MAX_SHARED_MEMORY = kittens::MAX_SHARED_MEMORY;

    // Shared memory declared statically
    static constexpr int SCRATCH_BYTES = 1024;
    static constexpr int STATIC_SHARED_MEMORY = 512 + INSTRUCTION_PIPELINE_STAGES * (SCRATCH_BYTES + (INSTRUCTION_WIDTH + TIMING_WIDTH) * 4 + DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    // Shared memory declared dynamically
    static constexpr int PAGE_SIZE = 16384;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES == 13, "NUM_PAGES must be 13");

    static constexpr bool TIMING_RECORD_ENABLED = true;

    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;

    static constexpr int CONSUMER_REGISTERS = 104;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
};

template <int _hidden_dim, int _intermediate_dim, int _sm_count=132>
struct globals_t {

    constexpr static int matvec_block_size = 16;

    constexpr static int hidden_dim = _hidden_dim;
    constexpr static int intermediate_dim = _intermediate_dim;
    constexpr static int sm_count = _sm_count;

    using instruction_layout = ::kittens::prototype::vm::instruction_layout<mlp_config<1>>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<mlp_config<1>>;

    using up_weights_t = gl<bf16, 1, -1, intermediate_dim, hidden_dim, st_bf<matvec_block_size, 512>>;                 // assumed to be N by 2048 (X@W.T).
    using down_weights_t = gl<bf16, 1, -1, hidden_dim, intermediate_dim, st_bf<matvec_block_size, 512>>; // assumed to be N by 2048 (X@W.T).

    using input_activations_t = gl<float, 1, 1, 1, hidden_dim, sv_fl<hidden_dim>>;
    using intermediate_activations_t = gl<float, 1, 1, 1, intermediate_dim, sv_fl<hidden_dim>, sv_fl<matvec_block_size>>;
    using output_activations_t = gl<float, 1, 1, 1, hidden_dim, sv_fl<matvec_block_size>>;

    using barriers = gl<uint, 1, 1, 1, -1>;

    // vm stuff
    barriers Bar;
    instruction_layout instructions;
    timing_layout timings;

    // model weights
    up_weights_t up_weights;
    down_weights_t down_weights;

    // activation buffers
    input_activations_t inputs;
    intermediate_activations_t intermediates;
    output_activations_t outputs;

    dim3 grid() { return dim3(sm_count); }
    dim3 block() { return dim3(mlp_config<1>::NUM_THREADS); }
    int dynamic_shared_memory() { return mlp_config<1>::DYNAMIC_SHARED_MEMORY; }
};

typedef globals_t<
    2048,
    4096,
    132>
    mlp_micro_globals;

// template <typename mlp_config = mlp_config, typename globals = mlp_micro_globals>
// struct UpOp;

// template <typename mlp_config = mlp_config, typename globals = mlp_micro_globals>
// struct DownOp;

}