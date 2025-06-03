#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for automatic Python list -> std::vector conversion

#include <random>
#include "kittens.cuh"

using namespace kittens;

__global__ void test_kernel(int *in, int *out) {
    // 32 threads, single block, array of size 32
    int tid = threadIdx.x;
    out[tid] = in[tid];
}

void test(pybind11::list ins, pybind11::list outs, std::shared_ptr<KittensClub> club) {

    if (!pybind11::hasattr(ins[0], "__class__") && 
            ins[0].attr("__class__").attr("__name__").cast<std::string>() == "Tensor")
        throw std::runtime_error("Expected a list of torch.Tensor");
    if (!ins[0].attr("is_contiguous")().cast<bool>())
        throw std::runtime_error("Tensor must be contiguous");
    if (ins[0].attr("device").attr("type").cast<std::string>() == "cpu")
        throw std::runtime_error("Tensor must be on CUDA device");

    int device_ids[4];
    for (int i = 0; i < 4; i++) device_ids[i] = i;

    // Assume all tensors are of shape (32,)
    // Assume num_devices = 4
    int *in_ds[4];
    int *out_ds[4];

    for (int i = 0; i < 4; i++) {
        in_ds[i] = reinterpret_cast<int*>(ins[i].attr("data_ptr")().cast<uint64_t>());
        out_ds[i] = reinterpret_cast<int*>(outs[i].attr("data_ptr")().cast<uint64_t>());
    }

    pgl<gl<int, 1, 1, 1, 32>, 4> in_pgl{device_ids, in_ds, nullptr, nullptr, nullptr, nullptr};
    pgl<gl<int, 1, 1, 1, 32>, 4> out_pgl{device_ids, out_ds, nullptr, nullptr, nullptr, nullptr};

    club->execute([&](int worker_id) {
        if (worker_id == 0) {
            test_kernel<<<1, 32>>>(in_pgl.mc_vas[0], out_pgl.mc_vas[0]);
            CUDACHECK(cudaDeviceSynchronize());
        }
    });
}

void run_club(std::shared_ptr<KittensClub> club) {
    club->execute([](int worker_id) {
        printf("Hello, from device %d!\n", worker_id);
    });
}

PYBIND11_MODULE(example_pgl, m) {
    m.doc() = "example_pgl python module";

    pybind11::class_<KittensClub, std::shared_ptr<KittensClub>>(m, "KittensClub")
        .def(pybind11::init([](const std::vector<int>& device_ids) {
            int device_count;
            CUDACHECK(cudaGetDeviceCount(&device_count));
            if (device_count < device_ids.size())
                throw std::runtime_error("Not enough CUDA devices available");
            auto club = std::make_shared<KittensClub>(device_ids.data(), device_ids.size());
            club->execute([&](int worker_idx) {
                // Warmup + enable peer access
                for (int idx = 0; idx < device_ids.size(); idx++) {
                    if (idx == worker_idx) continue;
                    int can_access = 0;
                    CUDACHECK(cudaDeviceCanAccessPeer(&can_access, device_ids[worker_idx], device_ids[idx]));
                    if (!can_access)
                        throw std::runtime_error("Device " + std::to_string(device_ids[worker_idx]) + " cannot access device " + std::to_string(device_ids[idx]));
                    CUDACHECK(cudaDeviceEnablePeerAccess(device_ids[idx], 0));
                }
            }); 
            return club;
        }), pybind11::arg("device_ids"));

    // pybind11::class_<pgl, std::shared_ptr<pgl>>(m, "pgl")
    //     .def(pybind11::init([](const std::vector<int>& device_ids) {
    //         auto pgl = std::make_shared<pgl>(device_ids.data(), device_ids.size());
    //         pgl->execute([](int worker_id) {}); // warmup
    //         return pgl;
    //     }), pybind11::arg("device_ids"));

    m.def("run_club", &run_club, "Run a club");
    m.def("test", &test, "Test");
}
