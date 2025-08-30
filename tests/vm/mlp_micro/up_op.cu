#include "matvec_pipeline.cuh"
#include "mlp.cuh"
using namespace kittens;
using namespace kittens::prototype;

namespace kittens::prototype::vm {

    template <typename Config, typename Globals>
    struct UpOp {
        static constexpr int opcode = 1; // Op index within the layer -- controls which barrier to listen to.

        struct parsed_instruction {
            int iters;
            int * block_idxs;
            __device__ inline parsed_instruction(typename Config::instruction_t &instruction) {
                iters = instruction[1];
                block_idxs = instruction + 2;
            }
            __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}
        };

        struct pipeline_specifics {
            static __device__ inline void gmem_wait(const Globals &g, state<Config> &s) {}

            static __device__ inline void load_iter(state<Config> &s, const Globals &g, parsed_instruction &inst, int iter, int col_idx, st_bf<16, 512> &weight_chunk, semaphore &sem) {
                auto block_idx = inst.block_idxs[iter];
                if(iter == 0 && laneid() == 0) s.record(80);
                tma::load_async<dim::ROW, cache_policy::EVICT_FIRST>(weight_chunk, g.up_weights, {block_idx, col_idx}, sem);

            }

            static __device__ inline void store(state<Config> &s, const Globals &g, parsed_instruction &inst, int output_idx, int output_stage) {

                int block_idx = inst.block_idxs[output_idx];

                sv_fl<16> &output_smem = *reinterpret_cast<sv_fl<16> *>((float *)s.scratch() + (32 * output_stage));

                rv_fl<16> output_rv;
                warp::load(output_rv, output_smem);
                warp::apply(output_rv, output_rv, []__device__(int i, float x) { return x < 0 ? 0 : x; }); // do relu
                warp::store(output_smem, output_rv);
                warp::sync();

                if (warp::laneid() == 0) {
                    if(output_idx == inst.iters-1) s.record(82);
                    auto &OutputActivations = g.intermediates; // object in global memory
                    tma::store_async<cache_policy::EVICT_LAST>(OutputActivations, output_smem, {block_idx});
                    tma::store_async_wait();
                    parsed_instruction inst{s};
                    atomicAdd(&g.Bar[{block_idx * 16 / Globals::hidden_dim}], 1);
                }

                warp::sync();
                warp::zero(output_smem);
                warp::sync();
            }
        };

        using pipeline = matvec_pipeline<Config, Globals, parsed_instruction, pipeline_specifics>;
        static_assert(pipeline::OUTPUT_PIPELINE_STAGES == 3);

        struct controller {
            static __device__ int release_lid(const Globals &g, typename Config::instruction_t &instruction, int &query) {
                return pipeline::release_lid(g, instruction, query);
            }
            static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
                return pipeline::init_semaphores(s);
            }
        };

        struct loader {
            static __device__ void run(const Globals &g, state<Config> &s) {
                s.template zero_scratch<1024>();

                parsed_instruction inst{s};
                pipeline::loader_loop(s, g);
            }
        };
        struct launcher {
            static __device__ void run(const Globals &g, state<Config> &s) {
                if (laneid() == 0) {
#ifdef KITTENS_BLACKWELL
                    s.wait_tensor_ready();
                    arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
                }
            }
        };
        struct consumer {
            static __device__ void run(const Globals &g, state<Config> &s) {

                using sv_t = sv_fl<pipeline::REDUCTION_DIM_PER_WARP>;
                using rv_t = rv_fl<pipeline::REDUCTION_DIM_PER_WARP>;
                parsed_instruction inst{s};

                sv_t &activations_smem = reinterpret_cast<sv_t *>(&pipeline::get_activations(s))[warpid()];
                if(laneid() == 0) s.record(81);

                warp::load(activations_smem, g.inputs, coord<>{warpid() * pipeline::REDUCTION_DIM_PER_WARP});
                warp::sync();

                rv_t activations_vec;
                warp::load(activations_vec, activations_smem);
                warp::sync();

                s.warp_finish_page(pipeline::get_activation_page(s), 1);

                pipeline::consumer_loop(s, g, activations_vec);
            }
        };

        struct storer {

            static __device__ void run(const Globals &g, state<Config> &s) {
                pipeline::storer_loop(s, g);
            }
        };
    };
}
