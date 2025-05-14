#include "matmul_pipeline.cuh"

#include "pyutils/pyutils.cuh"

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

struct matmul_globals {
    instruction_layout<llama_config> instructions;
    timing_layout<llama_config> timings;
    gl<bf16, 1, 1, -1, -1, st_bf<128, 64>> A;
    gl<bf16, 1, 1, -1, -1, st_bf<256, 64>> B;
    gl<bf16, 1, 1, -1, -1, st_bf<16, 256>> D;

    dim3 grid() { return dim3(148); }
    dim3 block() { return dim3(llama_config::NUM_THREADS); }
    int dynamic_shared_memory() { return llama_config::DYNAMIC_SHARED_MEMORY; }
};

struct matmul_op {
    using config = llama_config;
    using globals = matmul_globals;

    static constexpr int opcode = 1;

    struct parsed_instruction {
        int row;
        int col;
        int iters;
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            row = instruction[1];
            col = instruction[2];
            iters = instruction[3];
        }
        __device__ inline parsed_instruction(state<config> &s) : parsed_instruction(s.instruction()) {}
    };

    using matmul_pipeline = matmul_pipeline<config, globals, parsed_instruction, &globals::A, &globals::B>;
    static constexpr int MATMUL_SEMAPHORES = matmul_pipeline::SEM_COUNT;

    struct controller {
        static __device__ int release_lid(const globals &g, typename config::instruction_t &instruction, int &query) {
            return matmul_pipeline::release_lid(g, instruction, query);
        }
        static __device__ int init_semaphores(const globals &g, state<config> &s);
            return matmul_pipeline::init_semaphores(s);
        }
    };
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            matmul_pipeline::loader_loop(s, g);
        }
    };
    struct launcher {
        static __device__ void run(const globals &g, state<config> &s) {
            matmul_pipeline::launcher_loop(s, g);
        }
    };
    using half_consumer = group<config::NUM_CONSUMER_WARPS/2>;
    using constorer = group<config::NUM_CONSUMER_WARPS + 1>;
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            wait(matmul_pipeline::outputs_arrived(s), 0);
            rt_bf<16, 256> out;
            auto dt = s.tensor_alloc.template allocate<tt<float, 128, 256>>(half_consumer::groupid() * 256);
            half_consumer::load_async(out, dt);
            tensor_load_wait();
            __syncwarp();
            warp::arrive(s.tensor_finished);
            int store_bar = 10 + s.instruction_index%2;
            auto &smem = *reinterpret_cast<st_bf<16, 256>*>(s.scratch());
            for(int i = 0; i < config::NUM_CONSUMER_WARPS; i++) {
                if(warpid() == i) warp::store(smem, out);
                constorer::sync(store_bar); // arrive for storer
                constorer::sync(store_bar); // await release from storer
            }
        }
    };

    struct storer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst{s};
            int store_bar = 10 + s.instruction_index%2;
            auto &smem = *reinterpret_cast<st_bf<16, 256>*>(s.scratch());
            for(int i = 0; i < config::NUM_CONSUMER_WARPS; i++) {
                constorer::sync(store_bar); // await arrive from consumer
                warp::tma::store_async(g.D, smem, {16*inst.row + 8*(i>=8) + 2*(i%4) + ((i%8)/4), inst.col});
                tma::store_async_read_wait();
                constorer::sync(store_bar); // release back to consumer
            }
        }
    };
};

PYBIND11_MODULE(kvm_matmul, m)
{
    m.doc() = "kvm_matmul";
    kittens::py::bind_kernel<kvm<llama_config, 
        matmul_globals, 
        matmul_op
    >>(m, "kvm_matmul",
        &matmul_globals::instructions,
        &matmul_globals::timings,
        &matmul_globals::A,
        &matmul_globals::B,
        &matmul_globals::D
    );
}
