#pragma once

#include "util.cuh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // for automatic Python list -> std::vector conversion

namespace kittens {
namespace py {

template<typename T> struct from_object {
    static T make(pybind11::object obj) {
        return obj.cast<T>();
    }
};
template<ducks::gl::all GL> struct from_object<GL> {
    static GL make(pybind11::object obj) {
        // Check if argument is a torch.Tensor
        if (pybind11::hasattr(obj, "__class__") && 
            obj.attr("__class__").attr("__name__").cast<std::string>() == "Tensor") {
        
            // Check if tensor is contiguous
            if (!obj.attr("is_contiguous")().cast<bool>()) {
                throw std::runtime_error("Tensor must be contiguous");
            }
            if (obj.attr("device").attr("type").cast<std::string>() == "cpu") {
                throw std::runtime_error("Tensor must be on CUDA device");
            }
            
            // Get shape, pad with 1s if needed
            std::array<int, 4> shape = {1, 1, 1, 1};
            auto py_shape = obj.attr("shape").cast<pybind11::tuple>();
            size_t dims = py_shape.size();
            if (dims > 4) {
                throw std::runtime_error("Expected Tensor.ndim <= 4");
            }
            for (size_t i = 0; i < dims; ++i) {
                shape[4 - dims + i] = pybind11::cast<int>(py_shape[i]);
            }
            
            // Get data pointer using data_ptr()
            uint64_t data_ptr = obj.attr("data_ptr")().cast<uint64_t>();
            
            // Create GL object using make_gl
            return make_gl<GL>(data_ptr, shape[0], shape[1], shape[2], shape[3]);
        }
        throw std::runtime_error("Expected a torch.Tensor");
    }
};
template<ducks::pgl::all PGL> struct from_object<PGL> {
    static PGL make(pybind11::object obj) {
        if (!pybind11::isinstance<pybind11::list>(obj))
            throw std::runtime_error("Expected a Python list.");
        pybind11::list tensors = pybind11::cast<pybind11::list>(obj);
        if (tensors.size() != PGL::num_devices)
            throw std::runtime_error("Expected a list of " + std::to_string(PGL::num_devices) + " tensors");
        std::array<int, 4> shape = {1, 1, 1, 1};
        uint64_t data_ptrs[PGL::num_devices];
        for (int i = 0; i < PGL::num_devices; i++) {
            auto tensor = tensors[i];
            if (!pybind11::hasattr(tensor, "__class__") || 
                tensor.attr("__class__").attr("__name__").cast<std::string>() != "Tensor")
                throw std::runtime_error("Expected a list of torch.Tensor");
            if (!tensor.attr("is_contiguous")().cast<bool>())
                throw std::runtime_error("Tensor must be contiguous");
            if (tensor.attr("device").attr("type").cast<std::string>() == "cpu")
                throw std::runtime_error("Tensor must be on CUDA device");
            auto py_shape = tensor.attr("shape").cast<pybind11::tuple>();
            size_t dims = py_shape.size();
            if (dims > 4)
                throw std::runtime_error("Expected Tensor.ndim <= 4");
            for (size_t j = 0; j < dims; ++j) {
                if (i == 0)
                    shape[4 - dims + j] = pybind11::cast<int>(py_shape[j]);
                else if (shape[4 - dims + j] != pybind11::cast<int>(py_shape[j]))
                    throw std::runtime_error("All tensors must have the same shape");
            }
            data_ptrs[i] = tensor.attr("data_ptr")().cast<uint64_t>();
        }
        int device_ids[PGL::num_devices];
        for (int i = 0; i < PGL::num_devices; i++)
            device_ids[i] = tensors[i].attr("device").attr("index").cast<int>();
        size_t mem_granularity;
        CUmemAllocationProp mem_prop = {};
        CUCHECK(cuMemGetAllocationGranularity(&mem_granularity, &mem_prop, MEM_GRAN_TYPE));
        if constexpr (PGL::_INIT_MC) {
            if (sizeof(typename PGL::dtype) * shape[0] * shape[1] * shape[2] * shape[3] < mem_granularity)
                throw std::runtime_error("PGL tensor size must be at least " + std::to_string(mem_granularity) + " bytes");
        }
        return make_pgl<PGL>(device_ids, data_ptrs, shape[0], shape[1], shape[2], shape[3]);
    }
};

template<typename T> concept has_dynamic_shared_memory = requires(T t) { { t.dynamic_shared_memory() } -> std::convertible_to<int>; };
template<typename T> concept is_multigpu_globals = requires { 
    { T::num_devices } -> std::convertible_to<std::size_t>;
    { T::dev_idx } -> std::convertible_to<std::size_t>;
} && T::num_devices >= 1;

template<typename> struct trait;
template<typename MT, typename T> struct trait<MT T::*> { using member_type = MT; using type = T; };
template<typename> using object = pybind11::object;
template<auto kernel, typename TGlobal> static void bind_kernel(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def(name, [](object<decltype(member_ptrs)>... args, pybind11::kwargs kwargs) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        cudaStream_t raw_stream = nullptr;
        if (kwargs.contains("stream")) {
            // Extract stream pointer
            uintptr_t stream_ptr = kwargs["stream"].attr("cuda_stream").cast<uintptr_t>();
            raw_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
        }
        if constexpr (has_dynamic_shared_memory<TGlobal>) {
            int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
            kernel<<<__g__.grid(), __g__.block(), __dynamic_shared_memory__, raw_stream>>>(__g__);
        } else {
            kernel<<<__g__.grid(), __g__.block(), 0, raw_stream>>>(__g__);
        }
    });
}
template<auto function, typename TGlobal> static void bind_function(auto m, auto name, auto TGlobal::*... member_ptrs) {
    m.def(name, [](object<decltype(member_ptrs)>... args) {
        TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...};
        function(__g__);
    });
}
template<auto kernel, typename TGlobal> static void bind_multigpu_kernel(auto m, auto name, auto TGlobal::*... member_ptrs) {
    static_assert(is_multigpu_globals<TGlobal>, "Multigpu globals must have a member num_devices >= 1 and dev_idx");
    m.def("enable_all_p2p_access", [](const std::vector<int>& device_ids) {
        int device_count;
        CUDACHECK(cudaGetDeviceCount(&device_count));
        if (device_count < device_ids.size())
            throw std::runtime_error("Not enough CUDA devices available");
        for (int i = 0; i < device_ids.size(); i++) {
            CUDACHECK(cudaSetDevice(device_ids[i]));
            for (int j = 0; j < device_ids.size(); j++) {
                if (i == j) continue;
                int can_access = 0;
                CUDACHECK(cudaDeviceCanAccessPeer(&can_access, device_ids[i], device_ids[j]));
                if (!can_access)
                    throw std::runtime_error("Device " + std::to_string(device_ids[i]) + " cannot access device " + std::to_string(device_ids[j]));
                CUDACHECK(cudaDeviceEnablePeerAccess(device_ids[j], 0));
            }
        }
    });
    pybind11::class_<KittensClub, std::shared_ptr<KittensClub>>(m, "KittensClub")
        .def(pybind11::init([](const std::vector<int>& device_ids) {
            int device_count;
            CUDACHECK(cudaGetDeviceCount(&device_count));
            if (device_count < device_ids.size())
                throw std::runtime_error("Not enough CUDA devices available");
            auto club = std::make_shared<KittensClub>(device_ids.data(), device_ids.size());
            club->execute([&](int dev_idx) {}); // warmup
            return club;
        }), pybind11::arg("device_ids"));
    pybind11::class_<TGlobal, std::shared_ptr<TGlobal>>(m, "Globals")
        .def(pybind11::init([](object<decltype(member_ptrs)>... args) {
            return std::make_shared<TGlobal>(from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args)...);
        }));
    m.def(name, [](std::shared_ptr<TGlobal> globals, std::shared_ptr<KittensClub> club) {
        if constexpr (has_dynamic_shared_memory<TGlobal>) {
            club->execute([&](int dev_idx) {
                TGlobal __g__ = *globals;
                __g__.dev_idx = dev_idx;
                int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
                kernel<<<__g__.grid(), __g__.block(), __dynamic_shared_memory__>>>(__g__);
            });
        } else {
            club->execute([&](int dev_idx) {
                TGlobal __g__ = *globals;
                __g__.dev_idx = dev_idx;
                kernel<<<__g__.grid(), __g__.block(), 0>>>(__g__);
            });
        }
    });
    // TODO: PGL destructor binding
}

} // namespace py
} // namespace kittens
