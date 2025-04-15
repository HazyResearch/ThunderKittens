#define RED_TEXT "\033[31m"
#define GREEN_TEXT "\033[32m"
#define YELLOW_TEXT "\033[33m"
#define BLUE_TEXT "\033[34m"
#define MAGENTA_TEXT "\033[35m"
#define CYAN_TEXT "\033[36m"
#define WHITE_TEXT "\033[37m"
#define RESET_TEXT "\033[0m"

#include "kittens.cuh"
// #define KVM_DEBUG
#include "vm/vm.cuh"
#include <iostream>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::vm;

/*
Instruction format:
[0] = opcode
[1] = Row offset of C, in units of 128
[2] = Col offset of C, in units of 128
[3] = K reduction dimension, in units of 128
*/

using config = default_config;
struct globals {
    using instruction_layout = ::kittens::prototype::vm::instruction_layout<config>;
    using timing_layout = ::kittens::prototype::vm::timing_layout<config>;
    using fp8_matrix = gl<fp8e4m3, 1, 1, -1, -1, st_fp8e4m3<128, 128>>;
    instruction_layout instructions;
    timing_layout timings;
    fp8_matrix A, B, C;
    dim3 grid() { return dim3(148); }
    dim3 block() { return dim3(config::NUM_THREADS); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct MatmulOp {
    static constexpr int opcode = 1;
    struct parsed_instruction {
        int row, col, iters;
    };
    static __device__ inline parsed_instruction parse_instruction(const globals &g, state<config> &s) {
        return parsed_instruction{s.instruction()[1], s.instruction()[2], s.instruction()[3]};
    }
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst = parse_instruction(g, s);
            s.advance_mini_page(1);
            // for(int i = 0; i < inst.iters; i++) {
            //     #pragma unroll
            //     for(int j = 0; j < 2; j++) {
            //         int a_page = s.get_page();
            //         st_fp8e4m3<128, 128> &a = s.pages[a_page].template as_st<fp8e4m3>();
            //         warp::tma::expect(s.page_arrived[a_page], a);
            //         warp::tma::load_async(a, g.A, {inst.row+j, i}, s.page_arrived[a_page]);
            //         warp::arrive(s.page_arrived[a_page], config::NUM_CONSUMER_WARPS-1);
            //         // if(laneid() == 0) printf(RED_TEXT "A PAGE %d received %d arrivals + expect from loader\n" RESET_TEXT, a_page, config::NUM_CONSUMER_WARPS-1);
            //     }
            //     #pragma unroll
            //     for(int j = 0; j < 2; j++) {
            //         int b_page = s.get_page();
            //         st_fp8e4m3<128, 128> &b = s.pages[b_page].template as_st<fp8e4m3>();
            //         warp::tma::expect(s.page_arrived[b_page], b);
            //         warp::tma::load_async(b, g.B, {inst.col+j, i}, s.page_arrived[b_page]);
            //         // warp::tma::load_async(b, g.B, {i, inst.col+j}, s.page_arrived[b_page]);
            //         warp::arrive(s.page_arrived[b_page], config::NUM_CONSUMER_WARPS-1);
            //         // if(laneid() == 0) printf(RED_TEXT "B PAGE %d received %d arrivals + expect from loader\n" RESET_TEXT, b_page, config::NUM_CONSUMER_WARPS-1);
            //     }
            // }
            if(laneid() < 4) {
                s.advance_page(laneid());
                for(int i = 0; i < inst.iters; i++) {
                    // printf(BLUE_TEXT "Loader %d requesting page for iter %d\n" RESET_TEXT, laneid(), i);
                    int load_page = s.get_page();
                    st_fp8e4m3<128, 128> &load_buffer = s.pages[load_page].template as_st<fp8e4m3>();
                    tma::expect(s.page_arrived[load_page], load_buffer);
                    if(laneid() < 2) { // load a
                        // printf(RED_TEXT "Loader %d loading A page %d for iter %d\n" RESET_TEXT, laneid(), load_page, i);
                        tma::load_async(load_buffer, g.A, {inst.row+laneid(), i}, s.page_arrived[load_page]);
                    }
                    else { // load b
                        // printf(RED_TEXT "Loader %d loading B page %d for iter %d\n" RESET_TEXT, laneid(), load_page, i);
                        tma::load_async(load_buffer, g.B, {inst.col+laneid()-2, i}, s.page_arrived[load_page]);
                    }
                    arrive(s.page_arrived[load_page], config::NUM_CONSUMER_WARPS-1);
                    s.advance_page(4-1); // -1 for the successful get_page
                }
                s.advance_page(4-laneid()); // Advance 4 pages for the stores, minus how many over the end we are.
            }
            else s.advance_page(4*inst.iters+4);
            warp::sync();
        }
    };
    struct launcher { // launches mma's
        // Uses one minipage, and 4*iters full pages.
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst = parse_instruction(g, s);
            int semaphore_page = s.get_mini_page();
            semaphore *mma_sems = reinterpret_cast<semaphore*>(&s.mini_pages[semaphore_page]);
            init_semaphore(mma_sems[laneid()], 1); // create 32 semaphores for tracking mma lanes, fully cycling every 8 iters.
            __syncwarp();
            auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(laneid()*128);
            int base_mma_lane = laneid()%4;
            if(laneid() < 8) for(int i = 0; i < inst.iters; i++) {
                int p[4];
                int a_page, b_page;
                #pragma unroll
                for(int j = 0; j < 2; j++) {
                    p[j] = s.get_page();
                    if(base_mma_lane/2 == j) a_page = p[j];
                }
                #pragma unroll
                for(int j = 0; j < 2; j++) {
                    p[j+2] = s.get_page();
                    if(base_mma_lane%2 == j) b_page = p[j+2];
                }
                // printf("launcher lane %d, iter %d, pages received: %d, %d, %d, %d\n", laneid(), i, p[0], p[1], p[2], p[3]);
                int active_mma_lane = (base_mma_lane + 4*i)%32;
                if(laneid() < 4) {
                    // printf(YELLOW_TEXT "mma LAUNCHER lane %d, iter %d, waiting for semaphore %d\n" RESET_TEXT, base_mma_lane, i, active_mma_lane);
                    wait(mma_sems[active_mma_lane], 1);
                    // printf(YELLOW_TEXT "mma LAUNCHER lane %d, iter %d, waiting for A page %d\n" RESET_TEXT, base_mma_lane, i, a_page);
                    s.wait_page_arrived(a_page);
                    // printf(YELLOW_TEXT "mma LAUNCHER lane %d, iter %d, waiting for B page %d\n" RESET_TEXT, base_mma_lane, i, b_page);
                    s.wait_page_arrived(b_page);
                    st_fp8e4m3<128, 128> &a = s.pages[a_page].template as_st<fp8e4m3>();
                    st_fp8e4m3<128, 128> &b = s.pages[b_page].template as_st<fp8e4m3>();
                    // printf(BLUE_TEXT "mma LAUNCHER lane %d, iter %d, launching matmul into lane %d\n" RESET_TEXT, base_mma_lane, i, active_mma_lane);
                    if(i == 0) mm<transpose::N, transpose::T>(accumulator, a, b, mma_sems[active_mma_lane]);
                    else mma<transpose::N, transpose::T>(accumulator, a, b, mma_sems[active_mma_lane]);
                }
                else if(laneid() < 8) {
                    // printf(GREEN_TEXT "mma RECEIVER lane %d, iter %d, waiting for semaphore %d\n" RESET_TEXT, base_mma_lane, i, active_mma_lane);
                    wait(mma_sems[active_mma_lane], 0);
                    // printf(MAGENTA_TEXT "mma RECEIVER lane %d, iter %d, active lane %d, page barriers %d, %d\n" RESET_TEXT, base_mma_lane, i, active_mma_lane, a_page, b_page);
                    arrive(mma_sems[active_mma_lane]);
                    arrive(s.page_finished[a_page], config::NUM_CONSUMER_WARPS/2);
                    // printf(MAGENTA_TEXT "PAGE %d received %d arrivals from lane %d\n" RESET_TEXT, a_page, config::NUM_CONSUMER_WARPS/2, base_mma_lane);
                    arrive(s.page_finished[b_page], config::NUM_CONSUMER_WARPS/2);
                    // printf(MAGENTA_TEXT "PAGE %d received %d arrivals from lane %d\n" RESET_TEXT, b_page, config::NUM_CONSUMER_WARPS/2, base_mma_lane);
                }
            }
            else s.advance_page(4*inst.iters);
            __syncwarp();
            invalidate_semaphore(mma_sems[laneid()]); // Clean up minipage
            __syncwarp();
            // Mark minipage as arrived.
            warp::arrive(s.mini_page_arrived[semaphore_page], config::NUM_CONSUMER_WARPS);

            s.advance_page(4); // advance 4 pages for the stores.
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst = parse_instruction(g, s);
            int minipage = s.get_mini_page();
            s.advance_page(inst.iters*4);
            s.wait_mini_page_arrived(minipage);
            int cons_id = warpgroup::groupid();
            int store_page;
            for(int i = 0; i < 4; i++) {
                int p = s.get_page();
                if(cons_id == i) store_page = p;
            }
            st_fp8e4m3<128, 128> &store_buffer = s.pages[store_page].template as_st<fp8e4m3>();
            warpgroup::sync(warpgroup::groupid());
            warpgroup::arrive(s.mini_page_finished[minipage]);
            // Great, now we can start the store.
            auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(cons_id*128);
            rt_fl<32, 128> acc_rt;
            rt_fp8e4m3<32, 128> acc_fp8;
            warpgroup::load_async(acc_rt, accumulator);
            warp::copy(acc_fp8, acc_rt);
            warpgroup::store(store_buffer, acc_fp8);
            __syncwarp();
            warp::arrive(s.page_arrived[store_page], config::NUM_CONSUMER_WARPS/4);
        }
    };
    struct storer {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {
            parsed_instruction inst = parse_instruction(g, s);
            int minipage = s.get_mini_page();
            s.advance_page(inst.iters*4);
            s.wait_mini_page_arrived(minipage);
            warp::arrive(s.mini_page_finished[minipage], config::NUM_CONSUMER_WARPS*3/4);
            int output_pages[4];
            for(int i = 0; i < 4; i++) {
                output_pages[i] = s.get_page();
            }
            for(int r = 0; r < 2; r++) {
                for(int c = 0; c < 2; c++) {
                    st_fp8e4m3<128, 128> &output = s.pages[output_pages[2*r+c]].template as_st<fp8e4m3>();
                    s.wait_page_arrived(output_pages[2*r+c]);
                    warp::tma::store_async(g.C, output, {inst.row+r, inst.col+c});
                    warp::arrive(s.page_finished[output_pages[2*r+c]], config::NUM_CONSUMER_WARPS);
                }
            }
            tma::store_async_read_wait();
            for(int i = 0; i < 4; i++) {
                warp::arrive(s.page_finished[output_pages[i]], config::NUM_CONSUMER_WARPS);
            }
        }
    };
};

#include "pyutils/pyutils.cuh"

PYBIND11_MODULE(matmul, m) {
    m.doc() = "matmul python module";
    kittens::py::bind_kernel<kernel<config, globals, MatmulOp<config>>>(m, "matmul",
        &globals::instructions,
        &globals::timings,
        &globals::A,
        &globals::B,
        &globals::C
    );
}