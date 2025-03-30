#include "gang.cuh"

#ifdef TEST_GANG

template<typename Ker, kittens::ducks::pgl::all PGL>
static __global__ void global_wrapper(const __grid_constant__ PGL input, const __grid_constant__ PGL output, const __grid_constant__ int dev_idx) {
    Ker::device_func(input, output, dev_idx);
}

template<int NUM_DEVICES>
struct gang_test {
    using dtype = float;
    using PGL = kittens::pgl<kittens::gl<dtype, 1, 1, 1, kittens::WARP_THREADS * 2>, NUM_DEVICES, true>;

    static void run(test_data& results) {
        test_info this_result;
        this_result.label = "gang_sync";

        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count < NUM_DEVICES) {
            std::cerr << "Warning: Not enough GPU devices to run test " << this_result.label << std::endl;
            return;
        }

        // initialize
        constexpr int SIZE = kittens::WARP_THREADS * 2;
        dtype *d_i_arr[NUM_DEVICES];
        dtype *d_buf1_arr[NUM_DEVICES];
        dtype *d_buf2_arr[NUM_DEVICES];
        dtype *d_o_arr[NUM_DEVICES];
        std::vector<std::vector<float>> i_ref(NUM_DEVICES, std::vector<float>(SIZE));
        std::vector<std::vector<float>> dummy1(NUM_DEVICES, std::vector<float>(SIZE));
        std::vector<std::vector<float>> dummy2(NUM_DEVICES, std::vector<float>(SIZE));
        std::vector<std::vector<float>> o_ref(NUM_DEVICES, std::vector<float>(SIZE));
        int device_ids[NUM_DEVICES];
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) device_ids[dev_idx] = dev_idx;

        // This is hacky, but I didn't want to rewrite the initialize function just for this
        initialize<NUM_DEVICES>(device_ids, d_i_arr, d_buf1_arr, i_ref, dummy1);
        initialize<NUM_DEVICES>(device_ids, d_o_arr, d_buf2_arr, o_ref, dummy2); // values written to o_ref is discarded
        PGL input(device_ids, d_i_arr, nullptr, nullptr, nullptr, nullptr);
        PGL buffer1(device_ids, d_buf1_arr, nullptr, nullptr, nullptr, nullptr);
        PGL buffer2(device_ids, d_buf2_arr, nullptr, nullptr, nullptr, nullptr);
        PGL output(device_ids, d_o_arr, nullptr, nullptr, nullptr, nullptr);

        // run kernel
        for (int dev_idx = 0; dev_idx < NUM_DEVICES; ++dev_idx) {
            cudaSetDevice(dev_idx);
            global_wrapper<gang_test><<<1, kittens::WARP_THREADS>>>(input, output, dev_idx);
        }

        // fill in correct results on cpu
        host_func(i_ref, o_ref);
        // check and cleanup
        this_result.result = validate<NUM_DEVICES, PGL, dtype>(input, output, i_ref, o_ref, this_result.label);
        results.push_back(this_result);
    }

    __host__ static void host_func(const std::vector<std::vector<float>> &i_ref, std::vector<std::vector<float>> &o_ref) {
        // each vector represents a GPU device holding the data
        for (int dev_idx = 0; dev_idx < i_ref.size(); ++dev_idx) {
            for (int i = 0; i < i_ref[dev_idx].size(); ++i) {
                o_ref[dev_idx][i] = 0;
                for (int other_dev_idx = 0; other_dev_idx < i_ref.size(); ++other_dev_idx)
                    o_ref[dev_idx][i] += i_ref[other_dev_idx][i];
            }
        }
    }

    __device__ static void device_func(const PGL &input, const PGL &output, const int dev_idx) {
    }
};

void gang::tests(test_data &results) {
    std::cout << "\n ------------------------------     Starting ops/gang tests!     ------------------------------\n" << std::endl;

    if (check_multi_gpus()) {
        gang_test<NUM_GPUS>::run(results);
    }
}

#endif
