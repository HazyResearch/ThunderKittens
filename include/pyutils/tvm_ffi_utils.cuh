#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/extra/cuda/device_guard.h>
#include <tvm/ffi/extra/dtype.h>
#include <tvm/ffi/tvm_ffi.h>

#include "kittens.cuh"

namespace kittens {
namespace tvm_ffi {
namespace detail {

__host__ inline void check_cuda_device(DLDevice device) {
  TVM_FFI_CHECK(device.device_type == kDLCUDA, ValueError)
      << "Expected a CUDA tensor";
  TVM_FFI_CHECK(device.device_id >= 0, ValueError)
      << "CUDA device id must be non-negative";
}

__host__ inline std::uintptr_t tensor_data_address(
    const ::tvm::ffi::TensorView &tensor
) {
  TVM_FFI_CHECK(tensor.data_ptr() != nullptr, ValueError)
      << "Tensor data pointer must not be null";
  const std::uintptr_t base =
      reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
  TVM_FFI_CHECK(tensor.byte_offset() <=
                    std::numeric_limits<std::uintptr_t>::max() - base,
                ValueError)
      << "Tensor byte offset overflows the data pointer";
  return base + tensor.byte_offset();
}

template <typename DType>
__host__ inline std::uintptr_t validate_tensor(
    const ::tvm::ffi::TensorView &tensor
) {
  check_cuda_device(tensor.device());
  TVM_FFI_CHECK(tensor.ndim() >= 0 && tensor.ndim() <= 4, ValueError)
      << "Expected a tensor with 0 to 4 dimensions, got " << tensor.ndim();

  static_assert(std::is_same_v<DType, ::kittens::half> ||
                    std::is_same_v<DType, ::kittens::bf16> ||
                    std::is_same_v<DType, float> ||
                    std::is_same_v<DType, signed char> ||
                    std::is_same_v<DType, int>,
                "Unsupported TVM-FFI tensor dtype");
  constexpr DLDataType expected = ::tvm_ffi::dtype_trait<DType>::value;
  TVM_FFI_CHECK(tensor.dtype() == expected, TypeError)
      << "Tensor dtype mismatch: expected " << expected << ", got "
      << tensor.dtype();

  int64_t element_count = 1;
  for (int32_t i = 0; i < tensor.ndim(); ++i) {
    const int64_t dim = tensor.size(i);
    TVM_FFI_CHECK(dim > 0, ValueError)
        << "Tensor dimensions must be positive, got " << dim << " at axis "
        << i;
    TVM_FFI_CHECK(dim <= std::numeric_limits<int>::max(), ValueError)
        << "Tensor dimension exceeds int range at axis " << i << ": " << dim;
    TVM_FFI_CHECK(
        element_count <= std::numeric_limits<int64_t>::max() / dim,
        ValueError
    ) << "Tensor element count exceeds int64 range";
    element_count *= dim;
  }

  TVM_FFI_CHECK(tensor.IsContiguous(), ValueError)
      << "Tensor must be contiguous";

  const std::uintptr_t data = tensor_data_address(tensor);
  TVM_FFI_CHECK(data % alignof(DType) == 0, ValueError)
      << "Tensor data pointer is not naturally aligned for its dtype";

  return data;
}

} // namespace detail

__host__ inline void check_data_alignment(
    const ::tvm::ffi::TensorView &tensor,
    std::size_t alignment
) {
  detail::check_cuda_device(tensor.device());
  TVM_FFI_CHECK(
      alignment != 0 && (alignment & (alignment - 1)) == 0,
      ValueError
  ) << "Tensor alignment must be a nonzero power of two";
  const std::uintptr_t data = detail::tensor_data_address(tensor);
  TVM_FFI_CHECK(data % alignment == 0, ValueError)
      << "Tensor data pointer must be aligned to " << alignment << " bytes";
}

template <typename DType>
__host__ inline DType *tensor_data_ptr(
    const ::tvm::ffi::TensorView &tensor
) {
  const std::uintptr_t data =
      detail::validate_tensor<DType>(tensor);
  return reinterpret_cast<DType *>(data);
}

template <typename... TensorViews>
__host__ inline void check_same_device(const ::tvm::ffi::TensorView &first,
                                       const TensorViews &...rest) {
  detail::check_cuda_device(first.device());
  const DLDevice expected = first.device();
  const auto check_one = [expected](const ::tvm::ffi::TensorView &tensor) {
    detail::check_cuda_device(tensor.device());
    const DLDevice actual = tensor.device();
    TVM_FFI_CHECK(actual.device_type == expected.device_type &&
                      actual.device_id == expected.device_id,
                  ValueError)
        << "All tensors must be on the same CUDA device";
  };
  (check_one(rest), ...);
}

__host__ inline cudaStream_t get_cuda_stream(DLDevice device) {
  detail::check_cuda_device(device);
  return static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));
}

} // namespace tvm_ffi
} // namespace kittens
