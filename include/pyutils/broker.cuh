#pragma once

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <semaphore.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace kittens {

/**
    @brief KittensBroker utility for multiprocess data exchange.

    Note that the code relies on POSIX shared memory/semaphores for inter-process
    communication and synchronization.

    The only functions meant to be used by the user are `exchange` and `sync`:

        KittensBroker broker(local_rank, local_world_size);
        broker.exchange(dst, src, size); // exchange data between all processes
        broker.sync(); // wait until all processes reach here
 */
struct KittensBroker {
    __host__ inline KittensBroker(int local_rank, int local_world_size)
        : local_rank_(local_rank), 
          local_world_size_(local_world_size) {
        if (local_rank_ >= local_world_size_)
            throw std::runtime_error("Local rank is greater than local world size");
        if (local_world_size_ > MAX_LOCAL_WORLD_SIZE_)
            throw std::runtime_error("Local world size is greater than MAX_LOCAL_WORLD_SIZE");

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

    __host__ inline void exchange(void *dst_, const void *src_, size_t size) {
        if (size > VAULT_SIZE_PER_RANK_)
            throw std::runtime_error("Size is greater than VAULT_SIZE_PER_RANK_");

        uint8_t *dst = reinterpret_cast<uint8_t *>(dst_);
        const uint8_t *src = reinterpret_cast<const uint8_t *>(src_);

        // Exchange data
        sync(); // ensure all processes are ready
        memcpy(shm_->data + local_rank_ * VAULT_SIZE_PER_RANK_, src, size);
        sync();
    
        // Pack and copy back to destination
        for (int i = 0; i < local_world_size_; i++)
            memcpy(dst + i * size, shm_->data + i * VAULT_SIZE_PER_RANK_, size);
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

    static inline constexpr int MAX_LOCAL_WORLD_SIZE_ = 72;
    static inline constexpr int VAULT_SIZE_PER_RANK_ = 64; // sizeof(cudaIpcMemHandle_t)
    static inline constexpr int PAGE_SIZE_ = 4096;

    // TODO: make these unique per process group
    static inline constexpr const char* SHM_KEY_ = "/kittens_broker_shm";
    static inline constexpr const char* SEM_COUNTER_KEY_ = "/kittens_broker_sem_counter";
    static inline constexpr const char* SEM_ENTER_KEY_ = "/kittens_broker_sem_enter";
    static inline constexpr const char* SEM_EXIT_KEY_ = "/kittens_broker_sem_exit";
    static inline constexpr const char* SEM_READY_KEY_ = "/kittens_broker_sem_ready";

    struct KittensVault {
        int counter;
        uint8_t data[MAX_LOCAL_WORLD_SIZE_ * VAULT_SIZE_PER_RANK_];
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
