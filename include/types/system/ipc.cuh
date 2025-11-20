#pragma once

#include <concepts>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "../../common/common.cuh"
#include "vmm.cuh"

namespace kittens {
namespace ducks {
namespace ipc {
namespace handle {

struct identifier {};

template<typename T> concept all = requires {
    typename T::identifier;
} && std::is_same_v<typename T::identifier, identifier>;

} // namespace handle
} // namespace ipc
} // namespace ducks

namespace detail {
namespace ipc {

enum flavor {
    LEGACY = 0,
    VMM = 1
};

template<flavor _flavor>
struct handle;

template<> 
struct handle<flavor::LEGACY> {
    using identifier = ducks::ipc::handle::identifier;
    static constexpr flavor flavor_ = flavor::LEGACY;
    cudaIpcMemHandle_t handle_ {};
};

template<>
struct handle<flavor::VMM> {
    using identifier = ducks::ipc::handle::identifier;
    static constexpr flavor flavor_ = flavor::VMM;
    int handle_;
};

__host__ inline static void check_support(const int device_id) {
    CUdevice device;
    CUCHECK(cuDeviceGet(&device, device_id));

    int ipc_supported = 0;
    CUDACHECK(cudaDeviceGetAttribute(&ipc_supported, cudaDevAttrIpcEventSupport, device_id));
    int ipc_handle_supported = 0;
    CUCHECK(cuDeviceGetAttribute(&ipc_handle_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED, device));

    if (!ipc_supported || !ipc_handle_supported)
        throw std::runtime_error("CUDA IPC is not supported on this device");
}

template<ducks::ipc::handle::all IPC_HANDLE>
__host__ inline static void export_handle(
    IPC_HANDLE *ipc_handle,
    void *ptr
) {
    if constexpr (IPC_HANDLE::flavor_ == flavor::LEGACY) {
        CUDACHECK(cudaIpcGetMemHandle(&ipc_handle->handle_, ptr));
    } else if constexpr (IPC_HANDLE::flavor_ == flavor::VMM) {
        CUmemGenericAllocationHandle memory_handle;
        detail::vmm::vm_retrieve_handle(&memory_handle, ptr);
        // ** Important: this handle (FD) must be manually closed by the user **
        CUCHECK(cuMemExportToShareableHandle(&ipc_handle->handle_, memory_handle, detail::vmm::HANDLE_TYPE, 0));
        detail::vmm::vm_free(memory_handle);
    } else {
        throw std::runtime_error("Invalid IPC handle type");
    }
}

template<ducks::ipc::handle::all IPC_HANDLE>
__host__ inline static void export_handle(
    IPC_HANDLE *ipc_handle,
    CUmemGenericAllocationHandle &memory_handle
) {
    if constexpr (IPC_HANDLE::flavor_ == flavor::VMM) {
        CUCHECK(cuMemExportToShareableHandle(&ipc_handle->handle_, memory_handle, detail::vmm::HANDLE_TYPE, 0));
    } else {
        throw std::runtime_error("Invalid IPC handle type");
    }
}

template<ducks::ipc::handle::all IPC_HANDLE>
__host__ inline static void import_handle (
    void **ptr,
    IPC_HANDLE &ipc_handle,
    const size_t size,
    int local_world_size
) {
    if constexpr (IPC_HANDLE::flavor_ == flavor::LEGACY) {
        CUDACHECK(cudaIpcOpenMemHandle(ptr, ipc_handle.handle_, cudaIpcMemLazyEnablePeerAccess)); // this is the only flag supported
    } else if constexpr (IPC_HANDLE::flavor_ == flavor::VMM) {
        CUmemGenericAllocationHandle memory_handle;
        CUCHECK(cuMemImportFromShareableHandle(&memory_handle, reinterpret_cast<void *>(static_cast<uintptr_t>(ipc_handle.handle_)), detail::vmm::HANDLE_TYPE));
        detail::vmm::vm_map(ptr, memory_handle, size);
        detail::vmm::vm_set_access(*ptr, size, local_world_size);
        detail::vmm::vm_free(memory_handle);
        close(ipc_handle.handle_); // close fd immediately
        ipc_handle.handle_ = -1;
    } else {
        throw std::runtime_error("Invalid IPC handle type");
    }
}

template<ducks::ipc::handle::all IPC_HANDLE>
__host__ inline static void import_handle (
    CUmemGenericAllocationHandle *memory_handle,
    IPC_HANDLE &ipc_handle,
    const size_t size,
    int local_world_size
) {
    if constexpr (IPC_HANDLE::flavor_ == flavor::VMM) {
        CUCHECK(cuMemImportFromShareableHandle(memory_handle, reinterpret_cast<void *>(static_cast<uintptr_t>(ipc_handle.handle_)), detail::vmm::HANDLE_TYPE));
        close(ipc_handle.handle_); // close fd immediately
        ipc_handle.handle_ = -1;
    } else {
        throw std::runtime_error("Invalid IPC handle type");
    }
}

template<flavor _flavor>
__host__ inline static void free_handle(
    void *ptr,
    const size_t size
) {
    if constexpr (_flavor == flavor::LEGACY) {
        CUDACHECK(cudaIpcCloseMemHandle(ptr));
    } else if constexpr (_flavor == flavor::VMM) {
        detail::vmm::vm_unmap(ptr, size);
    } else {
        throw std::runtime_error("Invalid IPC handle type");
    }
}

} // namespace ipc
} // namespace detail
} // namespace kittens
