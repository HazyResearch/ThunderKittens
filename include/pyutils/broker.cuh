/**
    @file
    @brief KittensBroker utilities for multiprocess tensor exchange.

    This file specifically targets workloads run with `torchrun`, but should work
    for any single-node, multi-gpu, multi-process runs.

    Note that the code relies on POSIX shared memory/semaphores for inter-process
    communication and synchronization.

    Export to Python (pybind11):

        PYBIND11_MODULE(_C, m){
            pybind11::class_<KittensBroker>(m, "KittensBroker")
                .def(pybind11::init<int,int>())
                .def("gather_ipc_ptrs",
                    pybind11::overload_cast<const at::Tensor&>(&KittensBroker::gather_ipc_ptrs),
                    pybind11::call_guard<pybind11::gil_scoped_release>(),
                    pybind11::return_value_policy::move);
            pybind11::class_<KittensIPCPointerSet>(m, "KittensIPCPointerSet")
                .def(pybind11::init<>());
        }

    Example PyTorch usage:

        import torch
        from _C import KittensBroker

        ... (initialize torch distributed)

        broker = KittensBroker(local_rank, local_world_size)
        ipc_ptrs = broker.gather_ipc_ptrs(tensor)

        ... (now pass `broker` and `ipc_ptrs` to your kernels!)
 */

#pragma once

#include <cerrno>
#include <fcntl.h>
#include <iostream>
#include <semaphore.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <ATen/core/Tensor.h>
#include <cuda_runtime.h>

namespace kittens {

struct KittensIPCPointerSet {
    std::vector<bool> is_imported_ {};
    std::vector<void *> raw_ptrs_ {}; // if imported, MUST be freed with cudaIpcCloseMemHandle (not cudaFree), which is why we have a custom struct

    __host__ inline void append_ptr(void *raw_ptr, bool is_imported = false) {
        is_imported_.push_back(is_imported);
        raw_ptrs_.push_back(raw_ptr);
    }

    __host__ inline void append_from_handle(const cudaIpcMemHandle_t &import_handle) {
        void *raw_ptr;
        // Import the IPC handle. This implicitly & lazily does cudaDeviceEnablePeerAccess
        CUDACHECK(cudaIpcOpenMemHandle(&raw_ptr, import_handle, cudaIpcMemLazyEnablePeerAccess)); // this is the only flag supported
        append_ptr(raw_ptr, true);
    }

    __host__ inline void free_imported_ptrs(bool warn = false) {
        if (is_imported_.size() != raw_ptrs_.size())
            throw std::runtime_error("KittensIPCPointerSet: Size mismatch");

        // Close the imported pointers (this MUST be done first before cudaFree on the source)
        for (size_t i = 0; i < raw_ptrs_.size(); i++) {
            if (is_imported_[i]) {
                if (warn) std::cerr << "WARNING: Freeing an imported pointer." << std::endl;
                CUDACHECK(cudaIpcCloseMemHandle(raw_ptrs_[i]));
            }
        }
    }

    __host__ inline KittensIPCPointerSet() : is_imported_(), raw_ptrs_() {}

    KittensIPCPointerSet(const KittensIPCPointerSet&) = delete;
    KittensIPCPointerSet& operator=(const KittensIPCPointerSet&) = delete;

    __host__ inline KittensIPCPointerSet(KittensIPCPointerSet&& other) noexcept 
        : is_imported_(std::move(other.is_imported_)),
          raw_ptrs_(std::move(other.raw_ptrs_)) {
        other.is_imported_.clear();
        other.raw_ptrs_.clear();
    }
    __host__ inline KittensIPCPointerSet& operator=(KittensIPCPointerSet&& other) noexcept {
        if (this != &other) {
            free_imported_ptrs(true);
            is_imported_ = std::move(other.is_imported_);
            raw_ptrs_ = std::move(other.raw_ptrs_);
            other.is_imported_.clear();
            other.raw_ptrs_.clear();
        }
        return *this;
    }

    __host__ inline ~KittensIPCPointerSet() {
        // It is recommended to call KittensBroker::sync() after this
        free_imported_ptrs();
    }
};

struct KittensBroker {
    __host__ inline KittensBroker(int local_rank, int local_world_size)
        : local_rank_(local_rank), 
          local_world_size_(local_world_size) {
        if (local_rank_ >= local_world_size_)
            throw std::runtime_error("Local rank is greater than local world size");
        if (local_world_size_ > MAX_LOCAL_WORLD_SIZE_)
            throw std::runtime_error("Local world size is greater than MAX_LOCAL_WORLD_SIZE");
        if (!is_device_id_valid())
            throw std::runtime_error("Invalid device ID. Must be equal to local rank.");
        if (!is_cuda_ipc_supported())
            throw std::runtime_error("CUDA IPC is not supported on this device");

        // Create or open an existing shared memory
        int shm_id = shm_open(SHM_KEY_, O_CREAT | O_RDWR, 0600);
        if (shm_id == -1)
            throw std::runtime_error("Failed to create shared memory");

        // Create named semaphores
        sem_counter_ = sem_open(SEM_COUNTER_KEY_, O_CREAT, 0600, 1);
        sem_enter_ = sem_open(SEM_ENTER_KEY_, O_CREAT, 0600, 0);
        sem_exit_ = sem_open(SEM_EXIT_KEY_, O_CREAT, 0600, 0);
        sem_ready_ = sem_open(SEM_READY_KEY_, O_CREAT, 0600, 0);
        if (sem_counter_ == SEM_FAILED || sem_enter_ == SEM_FAILED || sem_exit_ == SEM_FAILED || sem_ready_ == SEM_FAILED)
            throw std::runtime_error("Failed to create semaphores");

        if (local_rank_ == 0) {
            // Allocate a page-aligned block
            if (ftruncate(shm_id, SHM_SIZE_) == -1)
                throw std::runtime_error("Failed to allocate shared memory");

            // Map shared memory
            void *p = mmap(
                nullptr, SHM_SIZE_, PROT_READ | PROT_WRITE, 
                MAP_SHARED, shm_id, 0
            );
            if (p == MAP_FAILED)
                throw std::runtime_error("Failed to map shared memory");
            close(shm_id);
            shm_ = (KittensVault *)p;
        
            // Initialize shared memory
            shm_->counter = 0;

            // Wake up other processes
            for (int i = 0; i < local_world_size_ - 1; i++)
                xsem_post(sem_ready_);
        } else {
            // Wait until initialized
            xsem_wait(sem_ready_);

            // Map shared memory
            void* p = mmap(
                nullptr, SHM_SIZE_, PROT_READ | PROT_WRITE, 
                MAP_SHARED, shm_id, 0
            );
            if (p == MAP_FAILED)
                throw std::runtime_error("Failed to map shared memory");
            close(shm_id);
            shm_ = (KittensVault *)p;
        }

        // Ensure all processes reach here
        sync();

        // Unlink immediately
        if (local_rank_ == 0) {
            shm_unlink(SHM_KEY_);
            sem_unlink(SEM_COUNTER_KEY_);
            sem_unlink(SEM_ENTER_KEY_);
            sem_unlink(SEM_EXIT_KEY_);
            sem_unlink(SEM_READY_KEY_);
        }

        // Clean up
        sem_close(sem_ready_);
        sem_ready_ = nullptr;

        // Ensure all processes reach here
        sync();
    }

    KittensBroker(const KittensBroker&) = delete;
    KittensBroker& operator=(const KittensBroker&) = delete;

    __host__ inline KittensBroker(KittensBroker&& other) noexcept
        : local_rank_(other.local_rank_),
          local_world_size_(other.local_world_size_),
          sem_counter_(other.sem_counter_),
          sem_enter_(other.sem_enter_),
          sem_exit_(other.sem_exit_),
          sem_ready_(other.sem_ready_),
          shm_(other.shm_) {
        other.sem_counter_ = nullptr;
        other.sem_enter_ = nullptr;
        other.sem_exit_ = nullptr;
        other.sem_ready_ = nullptr;
        other.shm_ = nullptr;
    }

    __host__ inline KittensBroker& operator=(KittensBroker&& other) noexcept {
        if (this != &other) {
            if (sem_counter_) sem_close(sem_counter_);
            if (sem_enter_) sem_close(sem_enter_);
            if (sem_exit_) sem_close(sem_exit_);
            if (sem_ready_) sem_close(sem_ready_);
            if (shm_) munmap(shm_, SHM_SIZE_);
            local_rank_ = other.local_rank_;
            local_world_size_ = other.local_world_size_;
            sem_counter_ = other.sem_counter_;
            sem_enter_ = other.sem_enter_;
            sem_exit_ = other.sem_exit_;
            sem_ready_ = other.sem_ready_;
            shm_ = other.shm_;
            other.sem_counter_ = nullptr;
            other.sem_enter_ = nullptr;
            other.sem_exit_ = nullptr;
            other.sem_ready_ = nullptr;
            other.shm_ = nullptr;
        }
        return *this;
    }

    __host__ inline ~KittensBroker() {
        if (sem_counter_) sem_close(sem_counter_);
        if (sem_enter_) sem_close(sem_enter_);
        if (sem_exit_) sem_close(sem_exit_);
        if (shm_) munmap(shm_, SHM_SIZE_);
    }

    __host__ inline void sync(int num_ranks = -1) {
        if (num_ranks == -1)
            num_ranks = local_world_size_;
        else if (num_ranks < 0 || num_ranks > local_world_size_)
            throw std::runtime_error("Invalid number of ranks");
    
        // Phase 1: arrive
        xsem_wait(sem_counter_);
        if (++shm_->counter == num_ranks) {
            for (int i = 0; i < num_ranks; i++)
                xsem_post(sem_enter_);
        }
        xsem_post(sem_counter_);
        xsem_wait(sem_enter_);
      
        // Phase 2: depart
        xsem_wait(sem_counter_);
        if (--shm_->counter == 0) {
            for (int i = 0; i < num_ranks; i++)
                xsem_post(sem_exit_);
        }
        xsem_post(sem_counter_);
        xsem_wait(sem_exit_);
    }

    __host__ inline KittensIPCPointerSet gather_ipc_ptrs(void *src_ptr) {
        // Export IPC handle
        cudaIpcMemHandle_t export_handle;
        CUDACHECK(cudaIpcGetMemHandle(&export_handle, src_ptr));

        // Share IPC handle
        sync(); // ensure all processes are ready
        shm_->handle[local_rank_] = export_handle;
        sync();
    
        // Import IPC handle
        KittensIPCPointerSet ptrs;
        for (int i = 0; i < local_world_size_; i++) {
            if (i == local_rank_) {
                ptrs.append_ptr(src_ptr, false);
            } else {
                ptrs.append_from_handle(shm_->handle[i]);
            }
        }

        return ptrs;
    }

    __host__ inline KittensIPCPointerSet gather_ipc_ptrs(uint64_t src_ptr) {
        return gather_ipc_ptrs(reinterpret_cast<void *>(src_ptr));
    }

    __host__ inline KittensIPCPointerSet gather_ipc_ptrs(const at::Tensor &t) {
        return gather_ipc_ptrs(reinterpret_cast<void *>(t.data_ptr()));
    }

    __host__ static inline void xsem_wait(sem_t* s) {
        while (sem_wait(s) == -1) {
            if (errno == EINTR)
                continue;
            throw std::runtime_error("Failed to wait on semaphore");
        }
    }
    
    __host__ static inline void xsem_post(sem_t* s) {
        if (sem_post(s) == -1)
            throw std::runtime_error("Failed to post on semaphore");
    }

    __host__ inline bool is_device_id_valid() const {
        int device_id;
        CUDACHECK(cudaGetDevice(&device_id));
        return device_id >= 0 && device_id < local_world_size_ && device_id == local_rank_;
    }

    __host__ inline bool is_cuda_ipc_supported() const {
        // Check if IPC is supported
        int ipc_supported;
        CUDACHECK(cudaDeviceGetAttribute(&ipc_supported, cudaDevAttrIpcEventSupport, local_rank_));
        return ipc_supported;
    }

    static inline constexpr int MAX_LOCAL_WORLD_SIZE_ = 72;
    static inline constexpr int PAGE_SIZE_ = 4096;

    // TODO: make these unique per process group
    static inline constexpr const char* SHM_KEY_ = "/kittens_broker_shm";
    static inline constexpr const char* SEM_COUNTER_KEY_ = "/kittens_broker_sem_counter";
    static inline constexpr const char* SEM_ENTER_KEY_ = "/kittens_broker_sem_enter";
    static inline constexpr const char* SEM_EXIT_KEY_ = "/kittens_broker_sem_exit";
    static inline constexpr const char* SEM_READY_KEY_ = "/kittens_broker_sem_ready";

    struct KittensVault {
        int counter;
        cudaIpcMemHandle_t handle[MAX_LOCAL_WORLD_SIZE_];
        __host__ inline KittensVault() : counter(0) {}
    };
    static inline constexpr int SHM_SIZE_ = (sizeof(KittensVault) + PAGE_SIZE_ - 1) / PAGE_SIZE_ * PAGE_SIZE_;

    int local_rank_;
    int local_world_size_;
    KittensVault *shm_;
    sem_t *sem_counter_;
    sem_t *sem_enter_;
    sem_t *sem_exit_;
    sem_t *sem_ready_;
};

} // namespace kittens
