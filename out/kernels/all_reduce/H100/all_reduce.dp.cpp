#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "kittens.dp.hpp"

#include <random>
#include <sycl/ext/intel/math.hpp>

#include <cmath>

constexpr int NUM_DEVICES = 8;
constexpr size_t N = 4096;

constexpr int WARPSIZE = 32;
constexpr int ITER_PER_THREAD = 32;
constexpr int MAX_VEC_SIZE = 16;

using namespace kittens;

using global_layout   =  gl<bf16, 1, 1, -1, -1>;
using pgl_m  =  pgl_manager<gl<bf16, 1, 1, -1, -1>, true>;

void all_reduce_bf16(kittens::pgl<global_layout> p_o) {
    kittens::all_reduce_add(p_o);
}

int main() {

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Setup
    int nelem = N * N;
    size_t size = nelem * sizeof(bf16);

    // Allocate and initialize host memory
    float **host_mats = new float*[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        host_mats[dev_idx] =  new float[nelem];
        for (int i = 0; i < nelem; ++i) host_mats[dev_idx][i] = dis(gen);
    }
    bf16 **host_mats_bf16 = new bf16*[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        host_mats_bf16[dev_idx] = new bf16[nelem];
        for (int i = 0; i < nelem; ++i)
            host_mats_bf16[dev_idx][i] =
                sycl::ext::intel::math::float2bfloat16(host_mats[dev_idx][i]);
    }
    float *expected = new float[nelem];
    for (int i = 0; i < nelem; ++i) {
        expected[i] = 0.0f;
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx)
            expected[i] += host_mats[dev_idx][i];
    }

    // Print data
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        std::cout << "Device " << dev_idx << ": ";
        for (int i = 0; i < std::min(nelem, 10); ++i) {
            std::cout << host_mats[dev_idx][i] << " ";
        }
        std::cout << "... (" << nelem << " elements)" << std::endl;
    }
    std::cout << "Expected: ";
    for (int i = 0; i < std::min(nelem, 10); ++i) {
        std::cout << expected[i] << " ";
    }
    std::cout << "... (" << nelem << " elements)" << std::endl;

    int device_ids[NUM_DEVICES];
    for (int i = 0; i < NUM_DEVICES; ++i) device_ids[i] = i;

    // Allocate and copy data to device
    bf16 **dev_mats = new bf16*[NUM_DEVICES];
    dpct::experimental::physical_mem_ptr *dev_handles =
        new dpct::experimental::physical_mem_ptr[NUM_DEVICES];
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        /*
        DPCT1093:610: The "dev_idx" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        dpct::select_device(dev_idx);
        pglCudaMalloc(NUM_DEVICES, device_ids, dev_idx, &dev_mats[dev_idx], &dev_handles[dev_idx], size);
        dpct::get_in_order_queue()
            .memcpy(dev_mats[dev_idx], host_mats_bf16[dev_idx], size)
            .wait();
    }

    // Initialize parallel global layout
    pgl_m dev_mat_pgl{device_ids, NUM_DEVICES, dev_mats, nullptr, nullptr, N, N};

    // Perform the reduction
    KittensClub club(device_ids, NUM_DEVICES);

    int nelem_per_dev = nelem / NUM_DEVICES;
    constexpr int nelem_per_block =
        256 * ITER_PER_THREAD *
        (MAX_VEC_SIZE / sizeof(sycl::ext::oneapi::bfloat16));

    dpct::dim3 grid((nelem_per_dev + nelem_per_block - 1) / nelem_per_block);
    dpct::dim3 block(256);

    club.execute([&](int worker_id) {
        /*
        DPCT1049:427: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        {
            auto exp_props = sycl::ext::oneapi::experimental::properties{
                sycl::ext::oneapi::experimental::use_root_sync};

            dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
                auto dev_mat_pgl_get_pgl_obj_worker_id_ct0 =
                    dev_mat_pgl.get_pgl_obj(worker_id);

                cgh.depends_on(dpct::get_current_device()
                                   .get_in_order_queues_last_events());

                cgh.parallel_for(sycl::nd_range<3>(grid * block, block),
                                 exp_props, [=](sycl::nd_item<3> item_ct1) {
                                     all_reduce_bf16(
                                         dev_mat_pgl_get_pgl_obj_worker_id_ct0);
                                 });
            });
        }
        CHECK_CUDA_ERROR(DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw()));
    });

    // Bring back data
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        /*
        DPCT1093:611: The "dev_idx" device may be not the one intended for use.
        Adjust the selected device if needed.
        */
        dpct::select_device(dev_idx);
        dpct::get_in_order_queue()
            .memcpy(host_mats_bf16[dev_idx], dev_mats[dev_idx], size)
            .wait();
    }

    // Convert back to float
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        for (int i = 0; i < nelem; ++i)
            host_mats[dev_idx][i] = sycl::ext::intel::math::bfloat162float(
                host_mats_bf16[dev_idx][i]);
    }

    // Print results
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        std::cout << "Device " << dev_idx << ": ";
        for (int i = 0; i < std::min(nelem, 10); ++i) {
            std::cout << host_mats[dev_idx][i] << " ";
        }
        std::cout << "... (" << nelem << " elements)" << std::endl;
    }

    // Verify the results
    float TOL = 1e-1; // large due to fp16 <-> bf16 conversion
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        for (int i = 0; i < nelem; ++i) {
            if (fabs(expected[i] - host_mats[dev_idx][i]) > TOL) {
                std::cerr << "Mismatch at device " << dev_idx << 
                             ", index " << i << 
                             ": expected " << expected[i] << 
                             ", got " << host_mats[dev_idx][i] << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }

    // Cleanup and exit
    for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
        delete[] host_mats[dev_idx];
        delete[] host_mats_bf16[dev_idx];
        pglCudaFree(dev_idx, dev_mats[dev_idx], dev_handles[dev_idx], size);
    }
    delete[] host_mats;
    delete[] host_mats_bf16;
    delete[] expected;
    delete[] dev_mats;
    delete[] dev_handles;

    std::cout << "Done!" << std::endl;
    return 0;
}