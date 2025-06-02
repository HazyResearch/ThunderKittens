#pragma once

#include "util.cuh"
#include <pybind11/pybind11.h>

namespace kittens {
namespace py {

template<typename T> struct from_object {
    static T make(pybind11::object obj) {
        return obj.cast<T>();
    }
    static T make(pybind11::object obj, int dev_idx) {
        if (!pybind11::isinstance<pybind11::list>(obj))
            throw std::runtime_error("Expected a Python list.");
        pybind11::list lst = pybind11::cast<pybind11::list>(obj);
        if (lst.size() <= dev_idx)
            throw std::runtime_error("Invalid list index " + std::to_string(dev_idx));
        return make(lst[dev_idx]);
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
    static GL make(pybind11::object obj, int dev_idx) {
        if (!pybind11::isinstance<pybind11::list>(obj))
            throw std::runtime_error("Expected a Python list.");
        pybind11::list lst = pybind11::cast<pybind11::list>(obj);
        if (lst.size() <= dev_idx)
            throw std::runtime_error("Invalid list index " + std::to_string(dev_idx));
        return make(lst[dev_idx]);
    }
};
template<ducks::gl_array::all GL_ARRAY> struct from_object<GL_ARRAY> {
    static GL_ARRAY make(pybind11::object obj) {
        if (!pybind11::isinstance<pybind11::list>(obj))
            throw std::runtime_error("Expected a Python list.");
        pybind11::list lst = pybind11::cast<pybind11::list>(obj);
        if (lst.size() != GL_ARRAY::N)
            throw std::runtime_error("Expected a Python list of size " + std::to_string(GL_ARRAY::N));
        uint64_t data_ptrs[GL_ARRAY::N];

        // Use the first element exclusively for extracting shape
        if (!pybind11::hasattr(lst[0], "__class__") && 
                lst[0].attr("__class__").attr("__name__").cast<std::string>() == "Tensor")
            throw std::runtime_error("Expected a list of torch.Tensor");

        // Check if tensor is contiguous & on gpu
        if (!lst[0].attr("is_contiguous")().cast<bool>())
            throw std::runtime_error("Tensor must be contiguous");
        if (lst[0].attr("device").attr("type").cast<std::string>() == "cpu")
            throw std::runtime_error("Tensor must be on CUDA device");

        std::array<int, 4> shape_ref = {1, 1, 1, 1};
        auto py_shape = lst[0].attr("shape").cast<pybind11::tuple>();
        size_t dims = py_shape.size();
        if (dims > 4)
            throw std::runtime_error("Expected Tensor.ndim <= 4");
        for (size_t i = 0; i < dims; ++i)
            shape_ref[4 - dims + i] = pybind11::cast<int>(py_shape[i]);

        data_ptrs[0] = lst[0].attr("data_ptr")().cast<uint64_t>();

        // Now check rest of the list
        for (int i = 1; i < GL_ARRAY::N; ++i) {
            if (!pybind11::hasattr(lst[i], "__class__") && 
                    lst[i].attr("__class__").attr("__name__").cast<std::string>() == "Tensor")
                throw std::runtime_error("Expected a list of torch.Tensor");

            // Check if tensor is contiguous & is on gpu
            if (!lst[i].attr("is_contiguous")().cast<bool>())
                throw std::runtime_error("Tensor must be contiguous");
            if (lst[i].attr("device").attr("type").cast<std::string>() == "cpu")
                throw std::runtime_error("Tensor must be on CUDA device");

            auto py_shape = lst[i].attr("shape").cast<pybind11::tuple>();
            size_t dims = py_shape.size();
            if (dims > 4)
                throw std::runtime_error("Expected Tensor.ndim <= 4");
            for (size_t j = 0; j < dims; ++j)
                if (pybind11::cast<int>(py_shape[j]) != shape_ref[4 - dims + j])
                    throw std::runtime_error("Tensor shapes do not match");

            data_ptrs[i] = lst[i].attr("data_ptr")().cast<uint64_t>();
        }

        // Create gl_array object
        return GL_ARRAY(data_ptrs, shape_ref[0], shape_ref[1], shape_ref[2], shape_ref[3]);
    }
    static inline GL_ARRAY make(pybind11::object obj, int dev_idx) {
        return make(obj);
    }
};

// Multi-GPU support
bool can_access_peer(int device, int peer_device) {
    int can_access = 0;
    CUDACHECK(cudaDeviceCanAccessPeer(&can_access, device, peer_device));
    return can_access != 0;
}
void enable_p2p_access(int device, int peer_device) {
    int current_device;
    CUDACHECK(cudaGetDevice(&current_device));
    CUDACHECK(cudaSetDevice(device));
    CUDACHECK(cudaDeviceEnablePeerAccess(peer_device, 0));
    CUDACHECK(cudaSetDevice(current_device));
}

template<typename T> concept has_dynamic_shared_memory = requires(T t) { { t.dynamic_shared_memory() } -> std::convertible_to<int>; };
template<typename T> concept is_multi_gpu = requires { 
    { T::num_devices } -> std::convertible_to<std::size_t>;
} && T::num_devices >= 1;

template<typename> struct trait;
template<typename MT, typename T> struct trait<MT T::*> { using member_type = MT; using type = T; };
template<typename> using object = pybind11::object;
template<auto kernel, typename TGlobal, size_t... I>
static inline void bind_multigpu_kernel(std::index_sequence<I...>, auto m, auto name, auto TGlobal::*... member_ptrs) {
    static_assert(is_multi_gpu<TGlobal>, "Globals must be a multi-GPU-compatible type");
    m.def(name, [](object<decltype(member_ptrs)>... args, pybind11::kwargs kwargs) {
        for (int dev_idx = 0; dev_idx < TGlobal::num_devices; ++dev_idx) {
            cudaSetDevice(dev_idx);
            // Just use the default cuda stream for now
            TGlobal __g__ {from_object<typename trait<decltype(member_ptrs)>::member_type>::make(args, dev_idx)...};
            if constexpr (has_dynamic_shared_memory<TGlobal>) {
                int __dynamic_shared_memory__ = (int)__g__.dynamic_shared_memory();
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, __dynamic_shared_memory__);
                kernel<<<__g__.grid(), __g__.block(), __dynamic_shared_memory__>>>(__g__);
            } else {
                kernel<<<__g__.grid(), __g__.block(), 0>>>(__g__);
            }
        }
    });
    // Additional helpers for multi-GPU support
    m.def("can_access_peer", can_access_peer);
    m.def("enable_p2p_access", enable_p2p_access);
}
template<auto kernel, typename TGlobal> static void bind_kernel(auto m, auto name, auto TGlobal::*... member_ptrs) {
    if constexpr (is_multi_gpu<TGlobal>)
        bind_multigpu_kernel<kernel, TGlobal>(std::make_index_sequence<TGlobal::num_devices>{}, m, name, member_ptrs...);
    else
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

} // namespace py
} // namespace kittens
