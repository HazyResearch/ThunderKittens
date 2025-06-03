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

void test(pybind11::object in, pybind11::object out, std::shared_ptr<KittensClub> club) {
    int *in_d = reinterpret_cast<int*>(in.attr("data_ptr")().cast<uint64_t>());
    int *out_d = reinterpret_cast<int*>(out.attr("data_ptr")().cast<uint64_t>());

    club->execute([&](int worker_id) {
        test_kernel<<<1, 32>>>(in_d, out_d);
        CUDACHECK(cudaDeviceSynchronize());
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
            auto club = std::make_shared<KittensClub>(device_ids.data(), device_ids.size());
            club->execute([](int worker_id) {}); // warmup
            return club;
        }), pybind11::arg("device_ids"));

    m.def("run_club", &run_club, "Run a club");
    m.def("test", &test, "Test");
}
