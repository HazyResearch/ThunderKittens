#pragma once

#include "util.cuh"
#include <pybind11/pybind11.h>

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
            for (size_t i = 0; i < dims && i < 4; ++i) {
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

#define EMPTY()
#define DEFER(m) m EMPTY EMPTY()()
#define EVAL(...) EVAL1024(__VA_ARGS__)
#define EVAL1024(...) EVAL512(EVAL512(__VA_ARGS__))
#define EVAL512(...) EVAL256(EVAL256(__VA_ARGS__))
#define EVAL256(...) EVAL128(EVAL128(__VA_ARGS__))
#define EVAL128(...) EVAL64(EVAL64(__VA_ARGS__))
#define EVAL64(...) EVAL32(EVAL32(__VA_ARGS__))
#define EVAL32(...) EVAL16(EVAL16(__VA_ARGS__))
#define EVAL16(...) EVAL8(EVAL8(__VA_ARGS__))
#define EVAL8(...) EVAL4(EVAL4(__VA_ARGS__))
#define EVAL4(...) EVAL2(EVAL2(__VA_ARGS__))
#define EVAL2(...) EVAL1(EVAL1(__VA_ARGS__))
#define EVAL1(...) __VA_ARGS__
#define MAP(m, struct_type, idx, first, ...)           \
  m(struct_type, idx, first)                           \
  __VA_OPT__(                        \
    , DEFER(_MAP)()(m, struct_type, (idx+1), __VA_ARGS__)    \
  )
#define _MAP() MAP
#define ITER(m, struct_type, ...) EVAL(MAP(m, struct_type, 0, __VA_ARGS__))

#define EXPAND_MEMBER_ACCESS(struct_type, idx, x) \
    kittens::py::from_object<decltype(std::declval<struct_type>().x)>::make(args[idx])

#define GENERATE_CONSTRUCTOR(struct_type, ...) \
    [](pybind11::args args) { \
        return struct_type{ \
            ITER(EXPAND_MEMBER_ACCESS, struct_type, __VA_ARGS__) \
        }; \
    }

#define BIND_KERNEL(module, name, kernel, globals_struct, ...) \
    module.def(name, [](pybind11::args args) { \
        globals_struct __g__{ ITER(EXPAND_MEMBER_ACCESS, globals_struct, __VA_ARGS__) }; \
        kernel<<<__g__.grid(), __g__.block()>>>(__g__); \
    });
#define BIND_FUNCTION(module, name, function, globals_struct, ...) \
    module.def(name, [](pybind11::args args) { \
        globals_struct __g__{ ITER(EXPAND_MEMBER_ACCESS, globals_struct, __VA_ARGS__) }; \
        function(__g__); \
    });

} // namespace py
} // namespace kittens