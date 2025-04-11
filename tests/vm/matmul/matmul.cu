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
    dim3 grid() { return dim3(1, 1, 1); }
    dim3 block() { return dim3(config::NUM_THREADS, 1, 1); }
    int dynamic_shared_memory() { return config::DYNAMIC_SHARED_MEMORY; }
};

template<typename config=config> struct TestOp {
    static constexpr int opcode = 1;
    struct parsed_instruction {
        int row, col, iters;
    };
    static __device__ inline parsed_instruction parse_instruction(const globals &g, state<config> &s) {
        return parsed_instruction{s.instruction()[1], s.instruction()[2], s.instruction()[3]};
    }
    struct loader {
        static __device__ void run(const globals &g, state<config> &s) {
            if(laneid() == 0) printf("Loader entering\n"); __nanosleep(1000); __syncwarp();
            parsed_instruction inst = parse_instruction(g, s);
            s.advance_mini_page(1);
            for(int i = 0; i < inst.iters; i++) {
                #pragma unroll
                for(int j = 0; j < 2; j++) {
                    int a_page = s.get_page();
                    if(laneid() == 0) printf("Loading A page %d\n", a_page); __nanosleep(1000); __syncwarp();
                    st_fp8e4m3<128, 128> &a = s.pages[a_page].template as_st<fp8e4m3>();
                    warp::tma::expect(s.page_arrived[a_page], a);
                    warp::tma::load_async(a, g.A, {inst.row+j, i}, s.page_arrived[a_page]);
                    warp::arrive(s.page_arrived[a_page], config::NUM_CONSUMER_WARPS-1);
                }
                #pragma unroll
                for(int j = 0; j < 2; j++) {
                    int b_page = s.get_page();
                    if(laneid() == 0) printf("Loading B page %d\n", b_page); __nanosleep(1000); __syncwarp();
                    st_fp8e4m3<128, 128> &b = s.pages[b_page].template as_st<fp8e4m3>();
                    warp::tma::expect(s.page_arrived[b_page], b);
                    warp::tma::load_async(b, g.B, {i, inst.col+j}, s.page_arrived[b_page]);
                    warp::arrive(s.page_arrived[b_page], config::NUM_CONSUMER_WARPS-1);
                }
            }
            s.advance_page(4); // Advance 4 pages for the stores.
            if(laneid() == 0) printf("Loader EXITING\n"); __nanosleep(1000); __syncwarp();
        }
    };
    struct launcher {
        // Uses one minipage, and 4*iters full pages.
        static __device__ void run(const globals &g, state<config> &s) {
            if(laneid() < 4) printf("Launcher entering\n"); __nanosleep(1000); __syncwarp();
            parsed_instruction inst = parse_instruction(g, s);
            int semaphore_page = s.get_mini_page();
            semaphore *mma_sems = reinterpret_cast<semaphore*>(&s.mini_pages[semaphore_page]);
            init_semaphore(mma_sems[laneid()], 1); // create 32 semaphores for tracking mma lanes, fully cycling every 8 iters.
            __syncwarp();
            if(laneid() < 4) printf("Launcher initialized semaphores\n"); __nanosleep(1000); __syncwarp();
            auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(laneid()*128);
            int base_mma_lane = laneid()%4;
            if(laneid() < 8) for(int i = 0; i < inst.iters; i++) {
                int a_page, b_page;
                #pragma unroll
                for(int j = 0; j < 2; j++) {
                    int p = s.get_page();
                    if(base_mma_lane/2 == j) a_page = p;
                }
                #pragma unroll
                for(int j = 0; j < 2; j++) {
                    int p = s.get_page();
                    if(base_mma_lane%2 == j) b_page = p;
                }
                int active_mma_lane = (base_mma_lane + 4*i)%32;
                if(laneid() < 4) {
                    wait(mma_sems[active_mma_lane], 1);
                    s.wait_page_arrived(a_page);
                    s.wait_page_arrived(b_page);
                    st_fp8e4m3<128, 128> &a = s.pages[a_page].template as_st<fp8e4m3>();
                    st_fp8e4m3<128, 128> &b = s.pages[b_page].template as_st<fp8e4m3>();
                    printf("Launching mma on A, B pages %d, %d, into semaphore %d\n", a_page, b_page, active_mma_lane); __nanosleep(1000); __syncwarp(0x0000000F);
                    mma_AB(accumulator, a, b, mma_sems[active_mma_lane]);
                }
                else if(laneid() < 8) {
                    printf("Page finisher waiting on mma semaphore %d\n", active_mma_lane); __nanosleep(1000); __syncwarp(0x000000F0);
                    wait(mma_sems[active_mma_lane], 0);
                    printf("Page finisher, mma semaphore %d\n", active_mma_lane); __nanosleep(1000); __syncwarp(0x000000F0);
                    arrive(mma_sems[active_mma_lane]);
                    // Arrive on the relevant page barriers
                    arrive(s.page_finished[a_page], config::NUM_CONSUMER_WARPS/2);
                    arrive(s.page_finished[b_page], config::NUM_CONSUMER_WARPS/2);
                }
            }
            else s.advance_page(4*inst.iters);
            __syncwarp();
            invalidate_semaphore(mma_sems[laneid()]); // Clean up minipage
            // Mark minipage as arrived.
            if(laneid() == 0) printf("Mini page arriving\n"); __nanosleep(1000); __syncwarp();
            warp::arrive(s.mini_page_arrived[semaphore_page], config::NUM_CONSUMER_WARPS);

            s.advance_page(4); // advance 4 pages for the stores.
            printf("Launcher %d EXITING running\n", laneid()); __nanosleep(1000); __syncwarp();
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, state<config> &s) {
            if(laneid() == 0) printf("Consumer entering\n"); __nanosleep(1000); __syncwarp();
            parsed_instruction inst = parse_instruction(g, s);
            int minipage = s.get_mini_page();
            s.advance_page(inst.iters*4);
            int cons_id = warpgroup::groupid();
            int store_page;
            for(int i = 0; i < 4; i++) {
                int p = s.get_page();
                if(cons_id == i) store_page = p;
            }
            st_fp8e4m3<128, 128> &store_buffer = s.pages[store_page].template as_st<fp8e4m3>();
            s.wait_mini_page_arrived(minipage);
            __syncwarp();
            warp::arrive(s.mini_page_finished[minipage]);
            if(laneid() == 0) printf("Consumer %d going to store into page %d\n", cons_id, store_page); __nanosleep(1000); __syncwarp();
            // Great, now we can start the store.
            auto accumulator = s.tensor_alloc.template allocate<tt<float, 128, 128>>(cons_id*128);
            rt_fl<32, 128> acc_rt;
            rt_fp8e4m3<32, 128> acc_fp8;
            warpgroup::load_async(acc_rt, accumulator);
            warp::copy(acc_fp8, acc_rt);
            warpgroup::store(store_buffer, acc_fp8);
            __syncwarp();
            warp::arrive(s.page_arrived[store_page], 4);
            if(laneid() == 0) printf("Consumer EXITING\n"); __nanosleep(1000); __syncwarp();
        }
    };
    struct storer {
        // Uses 4 full pages for outputs.
        static __device__ void run(const globals &g, state<config> &s) {
            if(laneid() == 0) printf("Storer entering\n"); __nanosleep(1000); __syncwarp();
            parsed_instruction inst = parse_instruction(g, s);
            s.advance_mini_page(1);
            s.advance_page(inst.iters*4);
            int output_pages[4];
            for(int i = 0; i < 4; i++) {
                output_pages[i] = s.get_page();
            }
            for(int r = 0; r < 2; r++) {
                for(int c = 0; c < 2; c++) {
                    st_fp8e4m3<128, 128> &output = s.pages[output_pages[2*r+c]].template as_st<fp8e4m3>();
                    if(laneid() == 0) printf("Awaiting C page %d: %d\n", 2*r+c, output_pages[2*r+c]); __nanosleep(1000); __syncwarp();
                    s.wait_page_arrived(output_pages[2*r+c]);
                    if(laneid() == 0) printf("Storing C page %d: %d\n", 2*r+c, output_pages[2*r+c]); __nanosleep(1000); __syncwarp();
                    warp::tma::store_async(g.C, output, {inst.row+r, inst.col+c});
                    warp::arrive(s.page_finished[output_pages[2*r+c]], config::NUM_CONSUMER_WARPS);
                }
            }
            tma::store_async_read_wait();
            for(int i = 0; i < 4; i++) {
                warp::arrive(s.page_finished[output_pages[i]], config::NUM_CONSUMER_WARPS);
            }
            if(laneid() == 0) printf("Storer EXITING\n"); __nanosleep(1000); __syncwarp();
        }
    };
};

PYBIND11_MODULE(matmul, m) {
    m.doc() = "matmul python module";
    kittens::py::bind_kernel<interpreter::kernel<config<16>, partial_template<16>, reduction_template<16>>>(m, "matmul",
        &config<16>::globals::instructions,
        &config<16>::globals::Q_rot,
        &config<16>::globals::Q,
        &config<16>::globals::K_rot_cache,
        &config<16>::globals::KV_cache,
        &config<16>::globals::Table,
        &config<16>::globals::O,
        &config<16>::globals::O_scratch,
        &config<16>::globals::Lvec_scratch,
        &config<16>::globals::semaphore,
        &config<16>::globals::Softmax_scale,
        &config<16>::globals::tic
#ifdef KITTENS_TIMINGS
        , &config<16>::globals::timings
#endif
    );
}

// int main() {

//     vm::print_config<config>();

//     // Initialize a vector of a basic instruction.
//     int instruction[config::INSTRUCTION_WIDTH] = {1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

//     // Create a device array for the instruction
//     int *d_instruction;
//     cudaMalloc(&d_instruction, config::INSTRUCTION_WIDTH * sizeof(int));
//     cudaMemcpy(d_instruction, instruction, config::INSTRUCTION_WIDTH * sizeof(int), cudaMemcpyHostToDevice);

//     // Create a device array for timing data
//     int *d_timing;
//     cudaMalloc(&d_timing, config::TIMING_WIDTH * sizeof(int));
//     cudaMemset(d_timing, 0, config::TIMING_WIDTH * sizeof(int));

//     fp8e4m3 *d_A, *d_B, *d_C;
//     cudaMalloc(&d_A, 128*128*sizeof(fp8e4m3));
//     cudaMalloc(&d_B, 128*128*sizeof(fp8e4m3));
//     cudaMalloc(&d_C, 128*128*sizeof(fp8e4m3));
    
//     // Use the device array
//     typename globals::instruction_layout instructions{d_instruction, nullptr, 1, 1, nullptr};
//     typename globals::timing_layout timings{d_timing, nullptr, 1, 1, nullptr};
//     typename globals::fp8_matrix A{d_A, nullptr, nullptr, 128, 128};
//     typename globals::fp8_matrix B{d_B, nullptr, nullptr, 128, 128};
//     typename globals::fp8_matrix C{d_C, nullptr, nullptr, 128, 128};
//     globals g{instructions, timings, A, B, C};
//     cudaFuncSetAttribute(kernel<config, globals, TestOp<config>>, cudaFuncAttributeMaxDynamicSharedMemorySize, config::DYNAMIC_SHARED_MEMORY);
//     ::kittens::prototype::vm::kernel<config, globals, TestOp<config>><<<1, config::NUM_THREADS, config::DYNAMIC_SHARED_MEMORY>>>(g);
    
//     // Check for CUDA errors
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
//         return 1;
//     }
    
//     // Synchronize device to ensure all operations are complete
//     error = cudaDeviceSynchronize();
//     if (error != cudaSuccess) {
//         std::cerr << "CUDA synchronize error: " << cudaGetErrorString(error) << std::endl;
//         return 1;
//     }

//     // Copy timing data back to host
//     int h_timing[config::TIMING_WIDTH];
//     cudaMemcpy(h_timing, d_timing, config::TIMING_WIDTH * sizeof(int), cudaMemcpyDeviceToHost);
    
//     // Print all timing data
//     std::cout << "Timing data:" << std::endl;
//     for (int i = 0; i < config::TIMING_WIDTH; i++) {
//         std::cout << "timing[" << i << "] = " << h_timing[i] << std::endl;
//     }

//     // Clean up
//     cudaFree(d_instruction);
//     cudaFree(d_timing);

//     std::cout << "Test passed!" << std::endl;

//     return 0;
// }