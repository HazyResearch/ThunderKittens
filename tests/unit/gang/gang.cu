#include "gang.cuh"

#ifdef TEST_GANG

template<typename Ker, kittens::ducks::pgl::all PGL, kittens::ducks::sync_manager::all SyncManager>
static __global__ void global_wrapper(
    const __grid_constant__ PGL input,
    const __grid_constant__ PGL buffer1,
    const __grid_constant__ PGL buffer2,
    const __grid_constant__ PGL output,
    const __grid_constant__ SyncManager sm, 
    const __grid_constant__ int sync_id,
    const __grid_constant__ int dev_idx
) {
    Ker::device_func(input, buffer1, buffer2, output, sm, sync_id, dev_idx);
}

template<int NUM_DEVICES>
struct gang_test {
    static constexpr int NUM_BLOCKS = 256;
    static constexpr int NUM_THREADS = 256;
    static constexpr int SIZE = NUM_BLOCKS * NUM_THREADS * 2;

    using dtype = float;
    using PGL = kittens::pgl<kittens::gl<dtype, 1, 1, 1, SIZE>, NUM_DEVICES, true>;
    using SyncManager = kittens::sync_manager<NUM_DEVICES, NUM_BLOCKS, 1>;

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
        KittensClub club(device_ids, NUM_DEVICES);
        SyncManager sm = SyncManager::create(device_ids);

        // run kernel
        club.execute([&](int dev_idx) {
            global_wrapper<gang_test><<<NUM_BLOCKS, NUM_THREADS>>>(input, buffer1, buffer2, output, sm, 0, dev_idx);
            cudaDeviceSynchronize(); // Important to sync here for the tests to run as intended
        });

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

    __device__ static void device_func(
        const PGL &input, 
        const PGL &buffer1, 
        const PGL &buffer2, 
        const PGL &output,
        const SyncManager &sm,
        const int sync_id, 
        const int dev_idx
    ) {
        // This should be run on a single warp
        int index = (blockIdx.x * blockDim.x + threadIdx.x) * 2;

        // pgl is purely used for convenient multicast/unicast generation up to this point
        buffer1[dev_idx].raw_ptr[index] = input[dev_idx].raw_ptr[index];
        buffer1[dev_idx].raw_ptr[index + 1] = input[dev_idx].raw_ptr[index + 1];

        // Stall odd threads && even blocks && odd devices for half a second
        // With this, it should be extremely unlikely for the test to pass without sync'ing
        stall(500 * (dev_idx % 2) * ((blockIdx.x + 1) % 2) * (threadIdx.x % 2));

        buffer2[dev_idx].raw_ptr[index] = buffer1[dev_idx].raw_ptr[index];
        buffer2[dev_idx].raw_ptr[index + 1] = buffer1[dev_idx].raw_ptr[index + 1];

        using gang = kittens::gang<NUM_DEVICES>;
        gang::sync(sm, sync_id, dev_idx);

        kittens::multimem_ld_reduce_op<float2, kittens::ReduceOp::ADD>::apply(
            reinterpret_cast<float2*>(&output[dev_idx].raw_ptr[index]), 
            reinterpret_cast<float2*>(&buffer2.mc_vas[dev_idx][index])
        );
    }

    __device__ static void stall(unsigned int ms) {
        for (int i = 0; i < ms; ++i) __nanosleep(1000000U); // nanosleep has an undocumented upper bound
    }
};

void gang::tests(test_data &results) {
    std::cout << "\n ------------------------------     Starting ops/gang tests!     ------------------------------\n" << std::endl;

    if (check_multi_gpus()) {
        gang_test<NUM_GPUS>::run(results);
    }
}

#endif
