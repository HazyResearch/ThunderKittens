// #include "kittens.cuh"

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <iostream>

static constexpr int CLUSTER_SIZE = 2;
static constexpr int NUM_CONSUMERS = 1;
static constexpr int NUM_PRODUCERS = 1;
static constexpr int NUM_WORKERS = (NUM_CONSUMERS + NUM_PRODUCERS) * 4;
static constexpr int NUM_THREADS = NUM_WORKERS * 32;
static constexpr int DYNAMIC_SHARED_MEMORY = 227*1024 - 1024;






















struct semaphore {
    private:
        uint64_t value;
    }; 











  using bf16 = __nv_bfloat16;





template<typename _T, int _rows, int _cols, bool _swizzle=true, int _swizzle_bytes=0>
struct __align__(128) st {
    using T = _T;
    using dtype = T;
    
    static constexpr bool swizzle = _swizzle;

    // define underlying data as same as that projected, to make clear that this is *not* a subtile.
    static constexpr int underlying_rows          = _rows;
    static constexpr int underlying_cols          = _cols;
    static constexpr int underlying_num_elements  = underlying_rows * underlying_cols;

    static constexpr int rows                = _rows; ///< Total number of rows in the tile.
    static constexpr int cols                = _cols; ///< Total number of cols in the tile.
    static constexpr int num_elements        = rows * cols; ///< Total number of elements in the tile.

    static_assert((swizzle && (rows % 16 == 0)) || (!swizzle && (rows % 16 == 0)), "Rows must be divisible by the tile dimension");
    static_assert((swizzle && (cols % 16 == 0)) || (!swizzle && (cols % 16 == 0)), "Cols must be divisible by the tile dimension");

    // If a user specifies a swizzle bytes value, the column byte size must be a multiple of the swizzle bytes.
    static_assert(_swizzle_bytes == 0 || _swizzle_bytes == 32 || _swizzle_bytes == 64 || _swizzle_bytes == 128);
    static constexpr int swizzle_bytes = _swizzle_bytes > 0 ? _swizzle_bytes : (
        sizeof(dtype) == 1 ? (  // Add FP8 case
            (cols/16)%4 == 0 ? 128 :
            (cols/16)%2 == 0 ?  64 : 32
        ) :
        sizeof(dtype) == 2 ? (
            (cols/16)%4 == 0 ? 128 :
            (cols/16)%2 == 0 ?  64 : 32
        ) :
        sizeof(dtype) == 4 ? (
            (cols/16)%2 == 0 ? 128 : 64
        ) : -1
    );

    dtype data[rows*cols]; ///< Raw data storage for the tile.
};

template<int _height, int _width, bool _swizzle=true, int _swizzle_bytes=0> 
using st_bf = st<bf16,  _height, _width, _swizzle, _swizzle_bytes>;


template<int default_alignment=1024> 
struct shared_allocator {
    int *ptr;

private:
    template<typename A, size_t... dims>
    struct variadic_array;
    template<typename A, size_t first_dim, size_t... rest_dims>
    struct variadic_array<A, first_dim, rest_dims...> {
        using type = typename variadic_array<A, rest_dims...>::type[first_dim];
    };
    template<typename A>
    struct variadic_array<A> {
        using type = A;
    };
    template<typename A, size_t... dims> 
    using variadic_array_t = typename variadic_array<A, dims...>::type;

    template<int alignment>
    __device__ inline void align_ptr() {
        if constexpr (alignment > 0) {
            uint64_t p = reinterpret_cast<uint64_t>(ptr);
            if(p % alignment != 0) {
                ptr = (int*)(p + (alignment-(p%alignment)));
            }
        }
    }

public:
    /**
    * @brief Construct a new shared allocator using a pointer to extern shared memory.
    * @param[in] _ptr Pointer to the start of the extern shared memory.
    */
    __device__ shared_allocator(int *_ptr): ptr(_ptr) {}
    /**
    * @brief Allocate shared memory for a single instance or N-dimensional array of type A.
    * @tparam A The type of the object to allocate.
    * @tparam dims... A list of dimensions for the N-dimensional array.
    * @return Reference to the allocated object.
    */
    template<typename A, size_t... dims> 
    __device__ inline variadic_array_t<A, dims...>& allocate() {
        // static_assert(sizeof(A) % default_alignment == 0, "Type is not aligned properly for array allocation");
        align_ptr<default_alignment>();
        using at = variadic_array_t<A, dims...>;
        at*p = reinterpret_cast<at*>(ptr);
        ptr += sizeof(at)/sizeof(int);
        return *p;
    }
    /**
    * @brief Allocate shared memory for a single instance or N-dimensional array of type A.
    * @tparam alignment An alignment to enforce for this particular object.
    * @tparam A The type of the object to allocate.
    * @tparam dims... A list of dimensions for the N-dimensional array.
    * @return Reference to the allocated object.
    */
    template<int alignment, typename A, size_t... dims> 
    __device__ inline variadic_array_t<A, dims...>& allocate() {
        // static_assert(sizeof(A) % alignment == 0, "Type is not aligned properly for array allocation");
        align_ptr<alignment>();
        using at = variadic_array_t<A, dims...>;
        at*p = reinterpret_cast<at*>(ptr);
        ptr += sizeof(at)/sizeof(int);
        return *p;
    }
};

using tma_allocator = shared_allocator<1024>;
using tma_swizzle_allocator = tma_allocator; // swizzled TMA modes require up to 1024 byte alignments :/



struct clc {

    struct handle {
        uint4 internal_value;
    }; // note that this is an opaque type, so the value should not be accessed directly.
    
    struct result {
        uint32_t success;
        uint32_t x;
        uint32_t y;
        uint32_t z;
    };
    
    /**
     * @brief Schedules a new threadblock. Must be called by a single thread in the entire CTA cluster.
     *        The caller must wait on the semaphore with tma::cluster::expect_bytes followed by tma::cluster::wait.
     *        The handle is multicasted to all CTAs in the cluster and signals the semaphore of all CTAs in the cluster.
     * @param h The CLC handle.
     * @param sem The semaphore that the caller will wait on.
     */
    __device__ static inline void schedule(handle &h, semaphore &sem) {
        asm volatile("{clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];}"
            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&h.internal_value))), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&sem)))
            : "memory"
        );
    }
    
    /**
     * @brief Queries the result of a schedule operation. Calling this again after failure is undefined behavior.
     * @param h The CLC handle.
     */
    __device__ static inline result query(handle &h) {
        result r;
        asm volatile(
            "{\n"
            ".reg .pred SUCCESS;\n"
            ".reg .b128 CLC_HANDLE;\n"
            "ld.shared.b128 CLC_HANDLE, [%4];\n"
            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 SUCCESS, CLC_HANDLE;\n"
            "selp.u32 %0, 1, 0, SUCCESS;\n"
            "@!SUCCESS bra.uni DONE;\n"
            "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%1, %2, %3, _}, CLC_HANDLE;\n"
            "fence.proxy.async.shared::cta;\n"
            "DONE:\n"
            "}"
            : "=r"(r.success), "=r"(r.x), "=r"(r.y), "=r"(r.z)
            : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&h.internal_value)))
            : "memory"
        );
        return r;
    }
    
}; // namespace clc





    __device__ static inline bool elect_warp_leader() {
        uint32_t elected = 0;
        asm volatile(
            "{.reg .pred P;\n"
            " elect.sync _|P, %1;\n"
            " selp.u32 %0, 1, 0, P;}\n"
            : "+r"(elected)
            : "r"(0xFFFFFFFF)
        );
        return static_cast<bool>(elected);
    }


    __device__ static inline void init_semaphore(semaphore& bar, int thread_count, int transaction_count=0) {
        void const* const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile (
            "mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"(bar_ptr), "r"(thread_count+transaction_count)
        );
    }




    template<int _nblocks_per_sm, int _ncta> struct tensor_allocator {    
        static constexpr int nblocks_per_sm = _nblocks_per_sm;
        static constexpr int cols =((512/nblocks_per_sm) / 32) * 32;
        static constexpr int ncta = _ncta;
    
        uint32_t addr;
    
        __device__ inline tensor_allocator() {
            __shared__ uint32_t shared_addr;
            static_assert(cols>0 && cols%32==0, "cols must be a multiple of 32");
            if constexpr (ncta == 1) {
                if(threadIdx.x / 32 == 0) {
                    asm volatile(
                        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32  [%0], %1;\n"
                    ::  "l"((uint64_t)&shared_addr), "n"(cols)
                    );
                    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
                }
            }
            else {
                if(threadIdx.x / 32 == 0) {
                    asm volatile(
                        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32  [%0], %1;\n"
                    ::  "l"((uint64_t)&shared_addr), "n"(cols)
                    );
                    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n");
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;\n");
            asm volatile("bar.sync 0;\n");
            asm volatile("tcgen05.fence::after_thread_sync;\n");
            addr = shared_addr;
        }
    
        __device__ inline uint32_t get_addr(int superlane, int col_offset) const { 
            return addr + ((superlane*16) << 16) + col_offset; 
        }
    
        __device__ inline uint32_t get_addr(int col_offset) const { 
            return addr + col_offset; 
        }
    
        __device__ inline ~tensor_allocator() {
            // if constexpr (ncta == 1) {
            //     if(warpid() == 0) {
            //         asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32  %0, %1;\n"
            //         ::  "r"(addr), "n"(cols)
            //         );
            //     }
            // } else {
            //     if(warpid() == 0) {
            //         asm volatile ("barrier.cluster.arrive.release.aligned;\n");
            //         asm volatile ("barrier.cluster.wait.acquire.aligned;\n");
            //         asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32  %0, %1;\n"
            //         ::  "r"(addr), "n"(cols)
            //         );
            //     }
            // }
        }
    };









    












 template<int _GROUP_WARPS>
 struct group {
 static constexpr int GROUP_WARPS = _GROUP_WARPS; // This alias produces nice parallelism.
 static constexpr int GROUP_THREADS = GROUP_WARPS * 32; // This alias produces nice parallelism.
 __device__ static inline int laneid() { return threadIdx.x % GROUP_THREADS; }
 __device__ static inline int warpid() { return laneid() / 32; }
 __device__ static inline int groupid() { return threadIdx.x / GROUP_THREADS; }
 
 };
 
 struct everyone {
 
 // Block-level synchronization
 __device__ static inline void sync(int id) {
     asm volatile("bar.sync %0;\n" :: "r"(id));
 }
 
 // Cluster-level synchronization functions
 struct tma {
 struct cluster {
 __device__ static inline void arrive_aligned() { // All threads in the cluster must call this
     asm volatile ("barrier.cluster.arrive.release.aligned;\n");
 }
 __device__ static inline void wait_aligned() {
     asm volatile ("barrier.cluster.wait.acquire.aligned;\n");
 }
 __device__ static inline void sync() {
     arrive_aligned();
     wait_aligned();
 }
 };
 };
 
 };
 
 using warp = group<1>;      // scope used by most pre-Hopper GPUs and most register operations.
 using warpgroup = group<4>; // special scope used by Hopper.
 
 


/**
 * @brief A namespace for all of ThunderKittens' TMA functionality.
*/
struct tma {

    /* ----------   Cluster-scope operations  ---------- */
    
    struct cluster {
    
    /**
    * @brief Waits for the requested semaphore phase, at cluster scope
    *
    * @param semaphore Reference to the semaphore variable.
    * @param kPhaseBit The phase bit used for the semaphore.
    */
    __device__ static inline void wait(semaphore& bar, int kPhaseBit) {
        void const* const ptr = &bar;
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
    
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(mbar_ptr),
            "r"(kPhaseBit)
        );
    }
    
    __device__ static inline bool try_wait(semaphore &bar, int kPhaseBit) {
        void const* const ptr = &bar;
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
        uint32_t success;
    
        asm volatile(
            "{\n"
            ".reg .pred P1; \n"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%1], %2; \n"
            "selp.b32 %0, 1, 0, P1; \n"
            "}\n"
            : "=r"(success)
            : "r"(mbar_ptr), "r"(kPhaseBit)
            : "memory"
        );
    
        return static_cast<bool>(success);
    }
    
    __device__ static inline void careful_wait(semaphore& bar, int kPhaseBit) {
        void const* const ptr = &bar;
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
    
        asm volatile (
            "{\n"
            ".reg .b64                 start_clock, current_clock;\n"
            "mov.b64                   start_clock, %clock64;\n"
            ".reg .pred                P_CLOCK;\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "mov.b64                   current_clock, %clock64;\n"
            "sub.u64                   current_clock, current_clock, start_clock;\n"
            "setp.ge.u64               P_CLOCK, current_clock, 1000000;\n"
            "@P_CLOCK                  trap;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(mbar_ptr),
            "r"(kPhaseBit)
        );
    }
    
    /**
    * @brief Sets the number of bytes expected at the semaphore, assuming a multicast instruction.
    *
    * This function sets the number of bytes expected at the semaphore for the first thread in the warp.
    * It converts the semaphore pointer to a generic shared memory pointer and uses an inline assembly
    * instruction to set the expected number of bytes.
    * 
    * It's worth being aware that this function is particularly necessary for multicast loads, and
    * distributed shared memory can actually be done with a normal tma::expect followed by wait. See
    * the unit tests of dsmem for an example.
    *
    * @param semaphore Reference to the semaphore variable.
    * @param bytes The number of bytes expected at the semaphore.
    */
    __device__ static inline void expect_bytes(semaphore& bar, uint32_t bytes) {
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
        asm volatile ("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
            :: "r"(mbar_addr), "r"(bytes));
    }
    /**
    * @brief Sets the number of bytes expected at the semaphore.
    *
    * This function sets the number of bytes expected at the semaphore for the first thread in the warp.
    * It converts the semaphore pointer to a generic shared memory pointer and uses an inline assembly
    * instruction to set the expected number of bytes.
    *
    * @tparam T The type of the data to be stored at the semaphore.
    * @param semaphore Reference to the semaphore variable.
    */
    /**
    * @brief Sets the number of bytes expected at the semaphore.
    *
    * This function sets the number of bytes expected at the mbarrier before the transaction arrives.
    */
    
    /**
    * @brief Arrives at a semaphore in cluster scope.
    *
    * Marks a thread arrival at an mbarrier
    *
    * @param semaphore Reference to the semaphore variable.
    * @param kPhaseBit The phase bit used for the semaphore.
    */
    __device__ static inline void arrive(semaphore& bar, int dst_cta, uint32_t count=1) {
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar)); 
        uint32_t neighbor_mbar_addr;
        asm volatile (
            "mapa.shared::cluster.u32  %0, %1, %2;\n"
            : "=r"(neighbor_mbar_addr)
            : "r"(mbar_addr), "r"(dst_cta)
        );
        asm volatile (
            "mbarrier.arrive.shared::cluster.b64 _, [%0], %1;\n"
            :
            : "r"(neighbor_mbar_addr), "r" (count)
            : "memory"
        );
    }
    
    // Generic transfer
    __device__ static inline void store_async(void *dst, void *src, int dst_cta, uint32_t size_bytes, semaphore& bar) {
        void const* const ptr = &bar;
        uint32_t mbarrier_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
    
        // **************************************************
        // load from src to dst in different threadblocks
        uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(src));
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    
        // mapa instr = https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa 
        // find dst addr in neighbor's cta
        uint32_t neighbor_addr_dst;
        asm volatile (
            "mapa.shared::cluster.u32  %0, %1, %2;\n"
            : "=r"(neighbor_addr_dst)
            : "r"(dst_ptr), "r"(dst_cta)
        );
        
        uint32_t neighbor_addr_mbarrier = mbarrier_ptr;
        asm volatile (
            "mapa.shared::cluster.u32  %0, %1, %2;\n"
            : "=r"(neighbor_addr_mbarrier)
            : "r"(mbarrier_ptr), "r"(dst_cta)
        );
        
        // cp.async instr = https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk 
        // copy src into dst in neighbor's cta
        asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");
        asm volatile (
            "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
            :
            : "r"(neighbor_addr_dst), "r"(src_ptr), "r"(size_bytes), "r"(neighbor_addr_mbarrier)
            : "memory"
        );
    }
    
    
    }; // namespace cluster
    }; // namespace tma



    __device__ static inline void wait(semaphore& sem, int kPhaseBit) {
        void const* const ptr = &sem;
        uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
    
        asm volatile (
            "{\n"
            ".reg .pred                P1;\n"
            "LAB_WAIT:\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
            "@P1                       bra.uni DONE;\n"
            "bra.uni                   LAB_WAIT;\n"
            "DONE:\n"
            "}\n"
            :: "r"(mbar_ptr),
            "r"(kPhaseBit)
        );
    }
    

    template<typename D, typename AB, int M, int N, bool trans_a, bool trans_b, bool neg=false>
    __device__ static inline constexpr uint32_t instruction_descriptor() {
        uint32_t desc = 0;
        if constexpr (sizeof(AB) == 2) { // kind::f16
            // either accumulate to float, or the input is half and the output is half
            static_assert(std::is_same_v<D, float> || std::is_same_v<AB, half>);
            desc |= 0b00      << 0;  // sparsity bits unneeded
            desc |= 0b0       << 2;  // dense
            desc |= 0b0       << 3;  // no saturate on fp types
            if constexpr (std::is_same_v<D, float>) {
                desc |= 0b01  << 4; // D matrix is FP32
            }
            else {
                desc |= 0b00  << 4; // D matrix is FP16
            }
            desc |= 0b0       << 6;  // reserved
                desc |= 0b001 << 7;  // 16-bit A input type as BF16
                desc |= 0b001 << 10; // 16-bit B input type as BF16
            if constexpr (neg) {
                desc |= 0b1   << 13; // Do negate A matrix
            }
            else {
                desc |= 0b0   << 13; // Don't negate A matrix
            }
            desc |= 0b0       << 14; // Don't negate B matrix (in all cases)
            if constexpr (trans_a) {
                desc |= 0b1   << 15; // Transpose A matrix
            }
            else {
                desc |= 0b0   << 15; // Don't transpose A matrix
            }
            if constexpr (trans_b) {
                desc |= 0b1  << 16; // Transpose B matrix
            }
            else {
                desc |= 0b0  << 16; // Don't transpose B matrix
            }
            desc |= (N >> 3) << 17; // B matrix has dimension N, encoded
            desc |= 0b0      << 23; // reserved
            desc |= (M >> 4) << 24; // A matrix has dimension M, encoded
            desc |= 0b0      << 29; // reserved
            desc |= 0b00     << 30; // no shift for B-matrix reuse
        } else {
            static_assert(sizeof(AB) == 999, "Invalid AB type size; not implemented yet.");
        }
        return desc;
    };

    
    struct detail {
    
    // See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
    __device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }
    
    template <typename T>
    __device__ static inline uint64_t matrix_descriptor_raw(
        T *addr,
        uint32_t leading_dim_offset,
        uint32_t stride_dim_offset,
        uint32_t swizzle_mode
    ) {
        // see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-shared-memory-descriptor
        return matrix_descriptor_encode(reinterpret_cast<uint64_t>(addr)) | 
               (1llu << 46) | // needed for blackwell shared memory descriptors
               matrix_descriptor_encode((uint64_t)leading_dim_offset) << 16 |
               matrix_descriptor_encode((uint64_t)stride_dim_offset) << 32 |
               (uint64_t)swizzle_mode << 62;
    }
    
    }; // namespace detail
    
    template<typename _ST, int MN_major>
    struct st_descriptor {
        using ST = _ST;
        using T = typename ST::T;
        static constexpr int rows = ST::rows;
        static constexpr int cols = ST::cols;
        static constexpr bool swizzle = ST::swizzle;
        uint64_t base_desc;
        __device__ inline st_descriptor(const ST &tile) {
            // See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-leading-dimension-byte-offset
            if constexpr (MN_major) { // MN major mode (i.e., K x M for A matrix, K x N for B matrix)
                if constexpr (ST::swizzle_bytes == 128) // 128B swizzle mode
                    base_desc = detail::matrix_descriptor_raw(&tile.data[0], 2048*ST::rows/16, 1024, 1);
                else if constexpr (ST::swizzle_bytes == 64) // 64B swizzle mode
                    base_desc = detail::matrix_descriptor_raw(&tile.data[0], 1024*ST::rows/16, 512, 2);
                else // 32B swizzle mode
                    base_desc = detail::matrix_descriptor_raw(&tile.data[0], 512*ST::rows/16, 256, 3);
            }
            else { // K major mode (i.e., M x K for A matrix, N x K for B matrix)
                if constexpr (ST::swizzle_bytes == 128) // 128B swizzle mode
                    base_desc = detail::matrix_descriptor_raw(&tile.data[0], 16 /* does not matter */, 1024, 1);
                else if constexpr (ST::swizzle_bytes == 64) // 64B swizzle mode
                    base_desc = detail::matrix_descriptor_raw(&tile.data[0], 16 /* does not matter */, 512, 2);
                else // 32B swizzle mode
                    base_desc = detail::matrix_descriptor_raw(&tile.data[0], 16 /* does not matter */, 256, 3);
            }
        }
        __device__ inline st_descriptor(const st_descriptor<ST, MN_major> &other) : base_desc(other.base_desc) {} // copy constructor
        __device__ inline uint64_t chunk_descriptor(int chunk_idx) {
            // Return the n-th chunk along the K dimension.
            // In MMA instructions, K per tensor core call is always 32 bytes
            //   ex. Hopper: K=32 for FP8, K=16 for BF16/FP16, K=8 for TF32)
            //   ex. Blackwell: K=64 for FP4, K=32 for FP8, K=16 for BF16/FP16, K=8 for TF32 (*For FP4, K=96 also possible, but we don't support yet)
            // So for MN-major, this is same as asking "how to forward 32 bytes worth of elements (=K elements) in the stride dimension?"
            // And for K-major, "how to forward K elements in the leading dimension?"
            if constexpr (MN_major) { // MN major mode (i.e., K x M for A matrix, K x N for B matrix)
                if constexpr (ST::swizzle_bytes == 128) { // 128B swizzle: 
                    return base_desc + detail::matrix_descriptor_encode(chunk_idx*2048);
                }
                else if constexpr (ST::swizzle_bytes == 64) {
                    return base_desc + detail::matrix_descriptor_encode(chunk_idx*1024);
                }
                else {
                    return base_desc + detail::matrix_descriptor_encode(chunk_idx*512);
                }
            }
            else { // K major mode (i.e., M x K for A matrix, N x K for B matrix)
                if constexpr (ST::swizzle_bytes == 128) {
                    // 128B swizzle: 4 chunks fit within swizzle bytes; move on to next every 4 chunks (rows * 128B swizzle bytes)
                    return base_desc + detail::matrix_descriptor_encode((chunk_idx%4)*32 + (chunk_idx/4)*(ST::rows/16)*2048);
                }
                else if constexpr (ST::swizzle_bytes == 64) {
                    // 64B swizzle: 2 chunks fit within swizzle bytes; move on to next every 2 chunks (rows * 64B swizzle bytes)
                    return base_desc + detail::matrix_descriptor_encode((chunk_idx%2)*32 + (chunk_idx/2)*(ST::rows/16)*1024);
                }
                else {
                    // 32B swizzle: Entire chunk fits within swizzle bytes; move on to next on every chunk (rows * 32B swizzle bytes)
                    return base_desc + detail::matrix_descriptor_encode(chunk_idx*(ST::rows/16)*512);
                }
            }
        }
    };








    template<int half>
    __device__ static inline int get_phasebit(uint32_t bitfield, int ring_id) {
        if constexpr (half == 0)
            return (bitfield >> (ring_id)) & 0b1;
        else if constexpr (half == 1)
            return (bitfield >> (ring_id + 16)) & 0b1;
        else
            asm volatile ("brkpt;\n");
        return -1;
    }
    
    template<int half> 
    __device__ static inline void update_phasebit(uint32_t &bitfield, int ring_id) {
        if constexpr (half == 0)
            bitfield ^= (1 << (ring_id));
        else if constexpr (half == 1)
            bitfield ^= (1 << (ring_id + 16));
        else
            asm volatile ("brkpt;\n");
    }













__device__ static inline int cluster_ctarank() {
    uint32_t ctarank;
    asm volatile("mov.u32 %0, %cluster_ctarank;\n" : "=r"(ctarank));
    return ctarank;
}













template <int _SUPERGROUP_SIZE, int _Mb, int _Nb, int _Kb, int _SMEM_PIPE_DEPTH, int _MMA_PIPE_DEPTH, int _TMEM_PIPE_DEPTH>
struct globals {
    static constexpr int SUPERGROUP_SIZE = _SUPERGROUP_SIZE;

    static constexpr int Mb = _Mb;
    static constexpr int Nb = _Nb;
    static constexpr int Kb = _Kb;
    
    static constexpr int CLUSTER_M = 2*Mb;
    static constexpr int CLUSTER_N = Nb;

    static constexpr int SMEM_PIPE_DEPTH = _SMEM_PIPE_DEPTH;
    static constexpr int MMA_PIPE_DEPTH = _MMA_PIPE_DEPTH;
    static constexpr int TMEM_PIPE_DEPTH = _TMEM_PIPE_DEPTH;
    static constexpr int CLC_PIPE_DEPTH = 1;

    static constexpr int NUM_D_TILES = TMEM_PIPE_DEPTH > 1 ? 2 : 1;

    using a_tile = st_bf<Mb, Kb>;
    using b_tile = st_bf<Nb/2, Kb>;
    using d_tile = st_bf<Mb, Nb/TMEM_PIPE_DEPTH>;

    __host__ __inline__ dim3 grid() { return dim3(8192/Mb*8192/Nb); }
    __host__ __inline__ dim3 block() { return dim3(NUM_THREADS); }
    __host__ __inline__ int dynamic_shared_memory() { return DYNAMIC_SHARED_MEMORY; }
};



















template <typename G>
__cluster_dims__(CLUSTER_SIZE, 1, 1) __launch_bounds__(NUM_THREADS, 1)
__global__ void kernel(
    const __grid_constant__ CUtensorMap a_tmap,
    const __grid_constant__ CUtensorMap b_tmap
) {
    const int cta_rank = cluster_ctarank();
    const int iters_per_task = 8192 / G::Kb;
    const int rblks = 8192 / G::CLUSTER_M;
    const int cblks = 8192 / G::CLUSTER_N;
    const int supergroup_cblks = (cblks/G::SUPERGROUP_SIZE)*G::SUPERGROUP_SIZE;
    const int supergroup_numel = G::SUPERGROUP_SIZE*rblks;
    const int finalgroup_cblks = cblks-supergroup_cblks;
    constexpr int bytes = (sizeof(G::a_tile) + sizeof(G::b_tile))*1;

    auto get_tile_idx = [&](int block_idx) -> int2 {
        const int cluster_idx = block_idx / CLUSTER_SIZE;
        if (cluster_idx < rblks*supergroup_cblks) {
            const int supergroup_idx = cluster_idx/supergroup_numel;
            const int rblk_idx = (cluster_idx%supergroup_numel)/G::SUPERGROUP_SIZE;
            return { (supergroup_idx&1) ? rblks-rblk_idx-1 : rblk_idx, G::SUPERGROUP_SIZE*supergroup_idx + cluster_idx%G::SUPERGROUP_SIZE };
        } else {
            const int supergroup_idx = cluster_idx/supergroup_numel;
            const int remainder_task_id = cluster_idx - supergroup_cblks*cblks;
            const int rblk_idx = remainder_task_id/finalgroup_cblks;
            return { (supergroup_idx&1) ? rblks-rblk_idx-1 : rblk_idx, supergroup_cblks + remainder_task_id%finalgroup_cblks };
        }
    };

    extern __shared__ int __shm[]; 
    tma_swizzle_allocator al((int*)&__shm[0]);

    typename G::a_tile (&a_smem)[G::SMEM_PIPE_DEPTH] = al.template allocate<G::a_tile, G::SMEM_PIPE_DEPTH>();
    typename G::b_tile (&b_smem)[G::SMEM_PIPE_DEPTH] = al.template allocate<G::b_tile, G::SMEM_PIPE_DEPTH>();

    if (threadIdx.x < 32) {
        __shared__ uint32_t shared_addr;
        asm volatile(
            "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32  [%0], %1;\n"
        ::  "l"((uint64_t)&shared_addr), "n"(512)
        );
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;\n");
    }

    __shared__ typename clc::handle clc_handle[G::CLC_PIPE_DEPTH];
    __shared__ semaphore schedule_arrived[G::CLC_PIPE_DEPTH], schedule_finished[G::CLC_PIPE_DEPTH];
    __shared__ semaphore inputs_arrived[G::SMEM_PIPE_DEPTH], inputs_finished[G::SMEM_PIPE_DEPTH], outputs_arrived, outputs_finished[G::MMA_PIPE_DEPTH];
    uint32_t bitfield = 0xFFFF0000; // ***_finished phase bits start as 1s, ***_arrived phase bits start as 0s

    if (threadIdx.x == 0) { 
        #pragma unroll
        for (int i = 0; i < G::CLC_PIPE_DEPTH; i++) {
            init_semaphore(schedule_arrived[i], 0, 1);
            init_semaphore(schedule_finished[i], 0, (2*CLUSTER_SIZE+1)*32);
        }
        #pragma unroll
        for (int i = 0; i < G::SMEM_PIPE_DEPTH; i++) {
            init_semaphore(inputs_arrived[i], 0, 2);
            init_semaphore(inputs_finished[i], 0, 1);
        }
        init_semaphore(outputs_arrived, 0, 1);
        #pragma unroll
        for (int i = 0; i < G::MMA_PIPE_DEPTH; i++) {
            init_semaphore(outputs_finished[i], 0, CLUSTER_SIZE);
        }
    }
    everyone::tma::cluster::sync();

    if (warpgroup::groupid() == 1) {
        // warpgroup::increase_registers<256>();
        if (warpgroup::warpid() == 2) {
            int input_ring = 0;
            int2 tile_coord = get_tile_idx(blockIdx.x);
            #pragma unroll 1
            for (int task_iter = 0; true; task_iter++) {

                int skip = 0;
                asm volatile(
                    "{\n"
                    ".reg .pred P1; \n"
                    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n"
                    "selp.b32 %0, 1, 0, P1; \n"
                    "}\n"
                    : "=r"(skip)
                    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_finished[input_ring]))), 
                      "r"(get_phasebit<1>(bitfield, input_ring))
                    : "memory"
                );

                #pragma unroll 1
                for (int idx = 0; idx < iters_per_task; idx++) {
                    if (!skip)
                        asm volatile (
                            "{\n"
                            ".reg .pred                P1;\n"
                            "LAB_WAIT:\n"
                            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
                            "@P1                       bra.uni DONE;\n"
                            "bra.uni                   LAB_WAIT;\n"
                            "DONE:\n"
                            "}\n"
                            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_finished[input_ring]))),
                            "r"(get_phasebit<1>(bitfield, input_ring)), "r"(0x989680)
                            : "memory"
                        );
                    update_phasebit<1>(bitfield, input_ring);

                    asm volatile(
                        "{\n"
                        ".reg .pred P1; \n"
                        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n"
                        "selp.b32 %0, 1, 0, P1; \n"
                        "}\n"
                        : "=r"(skip)
                        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_finished[(input_ring + 1) % G::SMEM_PIPE_DEPTH]))), 
                          "r"(get_phasebit<1>(bitfield, (input_ring + 1) % G::SMEM_PIPE_DEPTH))
                        : "memory"
                    );

                    if (elect_warp_leader()) {
                        asm volatile ("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;\n"
                            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_arrived[input_ring])) & 0xFEFFFFFF), 
                            "r"(bytes) : "memory");
                        asm volatile (
                            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%3, %4, %5}], [%2];"
                            :
                            : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&a_smem[input_ring]))), 
                            "l"(reinterpret_cast<uint64_t>(&a_tmap)), 
                            "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_arrived[input_ring])) & 0xFEFFFFFF),
                            "r"(idx*64), "r"((tile_coord.x*2+cta_rank)*128), "n"(0)
                            : "memory"
                        );
                        asm volatile (
                            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.cta_group::2"
                            " [%0], [%1, {%3, %4, %5}], [%2];"
                            :
                            : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&b_smem[input_ring]))), 
                            "l"(reinterpret_cast<uint64_t>(&b_tmap)), 
                            "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_arrived[input_ring])) & 0xFEFFFFFF),
                            "r"(idx*64), "r"((tile_coord.y*2+cta_rank)*128), "n"(0)
                            : "memory"
                        );
                    }
                    input_ring = (input_ring + 1) % G::SMEM_PIPE_DEPTH;
                }
                wait(schedule_arrived[task_iter%G::CLC_PIPE_DEPTH], (task_iter/G::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%G::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%G::CLC_PIPE_DEPTH], 0);
                if (schedule.success) tile_coord = get_tile_idx(schedule.x);
                else break;
            }

            #pragma unroll 1
            for (int idx = 0; idx < G::SMEM_PIPE_DEPTH; idx++) {
                asm volatile (
                    "{\n"
                    ".reg .pred                P1;\n"
                    "LAB_WAIT:\n"
                    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
                    "@P1                       bra.uni DONE;\n"
                    "bra.uni                   LAB_WAIT;\n"
                    "DONE:\n"
                    "}\n"
                    :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_finished[input_ring]))),
                    "r"(get_phasebit<1>(bitfield, input_ring)), "r"(0x989680)
                    : "memory"
                );
                update_phasebit<1>(bitfield, input_ring);
                input_ring = (input_ring + 1) % G::SMEM_PIPE_DEPTH;
            }
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32  %0, %1;\n" :: "r"(0), "n"(512));
        } else if (warpgroup::warpid() == 1) {
            #pragma unroll 1
            for (int task_iter = 0; true; task_iter++) {
                if (cta_rank == 0 && elect_warp_leader()) {
                    wait(schedule_finished[task_iter%G::CLC_PIPE_DEPTH], ((task_iter+G::CLC_PIPE_DEPTH)/G::CLC_PIPE_DEPTH)%2);
                    clc::schedule(clc_handle[task_iter%G::CLC_PIPE_DEPTH], schedule_arrived[task_iter%G::CLC_PIPE_DEPTH]);
                }
                if (elect_warp_leader()) tma::cluster::expect_bytes(schedule_arrived[task_iter%G::CLC_PIPE_DEPTH], sizeof(clc_handle[task_iter%G::CLC_PIPE_DEPTH]));
                wait(schedule_arrived[task_iter%G::CLC_PIPE_DEPTH], (task_iter/G::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%G::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%G::CLC_PIPE_DEPTH], 0);
                if (!schedule.success) break;
            }
        } else if (cta_rank == 0 && warpgroup::warpid() == 0) {
            constexpr uint32_t idesc = instruction_descriptor<float, bf16, 256, 256, 0, 0, false>();
            int input_ring = 0;
            #pragma unroll 1
            for (int task_iter = 0; true; task_iter++) {
                wait(schedule_arrived[task_iter%G::CLC_PIPE_DEPTH], (task_iter/G::CLC_PIPE_DEPTH)%2);
                auto schedule = clc::query(clc_handle[task_iter%G::CLC_PIPE_DEPTH]);
                tma::cluster::arrive(schedule_finished[task_iter%G::CLC_PIPE_DEPTH], 0);

                int skip = 0;
                asm volatile(
                    "{\n"
                    ".reg .pred P1; \n"
                    "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n"
                    "selp.b32 %0, 1, 0, P1; \n"
                    "}\n"
                    : "=r"(skip)
                    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_arrived[input_ring]))), 
                      "r"(get_phasebit<0>(bitfield, input_ring))
                    : "memory"
                );

                const uint32_t d_addr = task_iter%2 == 0 ? 0 : 256;
                // wait(outputs_finished[task_iter%G::MMA_PIPE_DEPTH], ((task_iter+G::MMA_PIPE_DEPTH)/G::MMA_PIPE_DEPTH)%2);
                #pragma unroll 1
                for(int idx = 0; idx < iters_per_task; idx++) {
                    if (!skip)
                        asm volatile (
                            "{\n"
                            ".reg .pred                P1;\n"
                            "LAB_WAIT:\n"
                            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
                            "@P1                       bra.uni DONE;\n"
                            "bra.uni                   LAB_WAIT;\n"
                            "DONE:\n"
                            "}\n"
                            :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_arrived[input_ring]))),
                            "r"(get_phasebit<0>(bitfield, input_ring)), "r"(0x989680)
                            : "memory"
                        );
                    update_phasebit<0>(bitfield, input_ring);

                    asm volatile(
                        "{\n"
                        ".reg .pred P1; \n"
                        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n"
                        "selp.b32 %0, 1, 0, P1; \n"
                        "}\n"
                        : "=r"(skip)
                        : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(&inputs_arrived[(input_ring + 1) % G::SMEM_PIPE_DEPTH]))), 
                          "r"(get_phasebit<0>(bitfield, (input_ring + 1) % G::SMEM_PIPE_DEPTH))
                        : "memory"
                    );

                    st_descriptor<typename G::a_tile, 0> a_desc(a_smem[input_ring]);
                    st_descriptor<typename G::b_tile, 0> b_desc(b_smem[input_ring]);
                    asm volatile ("fence.proxy.async.shared::cta;\n" ::: "memory");

                    if (elect_warp_leader()) {
                        asm volatile(
                            "{.reg .pred p;\n" \
                            "setp.eq.u32 p, 1, %4;\n" \
                            "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;}\n"
                        ::  "r"(d_addr), "l"(a_desc.chunk_descriptor(0)), "l"(b_desc.chunk_descriptor(0)), "r"(idesc), "r"(static_cast<uint32_t>(idx != 0))
                        );
                        #pragma unroll
                        for(int i = 1; i < 4; i++) {
                            asm volatile(
                                "{.reg .pred p;\n"
                                "setp.eq.u32 p, 1, %4;\n"
                                "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;}\n"
                            ::  "r"(d_addr), "l"(a_desc.chunk_descriptor(i)), "l"(b_desc.chunk_descriptor(i)), "r"(idesc), "r"(1) : "memory"
                            );
                        }
                        asm volatile(
                            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n"
                        ::  "l"(__cvta_generic_to_shared(&inputs_finished[input_ring])), "h"(static_cast<uint16_t>(0b11)));
                    }
                
                    input_ring = (input_ring + 1) % G::SMEM_PIPE_DEPTH;
                }
                // detail::tcgen05::commit<CLUSTER_SIZE>(outputs_arrived);
                if (!schedule.success) break;
            }
        }
    }
}








__host__ static inline void create_tensor_map(
    CUtensorMap *tma_map, const bf16 *src, int batch, int depth, int rows, int cols
) {
    using dtype = bf16;

    constexpr uint32_t  tma_dim = 3;
    void *global_addr = (void*)(src);

    constexpr CUtensorMapDataType     tma_format      = (
        std::is_same_v<dtype, bf16>  ? CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 :
        CUtensorMapDataType(-1)
    );
    constexpr CUtensorMapInterleave   tma_interleave  = CU_TENSOR_MAP_INTERLEAVE_NONE;
    constexpr CUtensorMapL2promotion  tma_l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    constexpr CUtensorMapFloatOOBfill tma_oobFill     = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    constexpr CUtensorMapSwizzle      tma_swizzle     = CU_TENSOR_MAP_SWIZZLE_128B;

    uint64_t gmem_shape [5] = {0, 0, 0, 0, 0};
    uint64_t gmem_stride[4] = {0, 0, 0, 0};
    uint32_t smem_shape [5] = {0, 0, 0, 0, 0};
    uint32_t smem_stride[5] = {1, 1, 1, 1, 1};

    constexpr uint64_t shared_tile_height = 128; 
    constexpr uint64_t shared_tile_width  = 64;

    // TMA expects the global and shared shapes to be in elements.
    constexpr int swizzle_elements = 128 / sizeof(dtype);

    gmem_shape[0] = (uint64_t)cols;
    gmem_shape[1] = (uint64_t)rows;
    gmem_shape[2] = (uint64_t)depth;
    gmem_shape[3] = (uint64_t)batch;

    gmem_stride[0] = (uint64_t)cols * sizeof(dtype);
    gmem_stride[1] = (uint64_t)rows * cols * sizeof(dtype);
    gmem_stride[2] = (uint64_t)depth * rows * cols * sizeof(dtype);

    smem_shape[0] = shared_tile_width;
    smem_shape[1] = shared_tile_height;
    smem_shape[2] = 1;
    smem_shape[3] = 1;

    const uint64_t *gmem_shape_ptr = &gmem_shape[0];
    const uint64_t *gmem_stride_ptr = &gmem_stride[0]; 
    const uint32_t *smem_shape_ptr = &smem_shape[0];
    const uint32_t *smem_stride_ptr = &smem_stride[0];

    cuTensorMapEncodeTiled(
        tma_map,
        tma_format,
        tma_dim,
        global_addr,
        gmem_shape_ptr,
        gmem_stride_ptr, 
        smem_shape_ptr,
        smem_stride_ptr,
        tma_interleave,
        tma_swizzle,
        tma_l2Promotion,
        tma_oobFill
    );
}














#include <omp.h>
#include <random>
#include <vector>
#include <unistd.h>

template <typename G>
__host__ double run_benchmark(size_t M, size_t N, size_t K, bool check_correctness = false, bool ncu = false) {
    std::cout << "--------------------  M=" << M << " N=" << N << " K=" << K << "  --------------------\n";
    std::cout << "Template: SUPERGROUP_SIZE=" << G::SUPERGROUP_SIZE << " Mb=" << G::Mb << " Nb=" << G::Nb << " Kb=" << G::Kb << 
                 " SMEM_PIPE_DEPTH=" << G::SMEM_PIPE_DEPTH << " MMA_PIPE_DEPTH=" << G::MMA_PIPE_DEPTH << " TMEM_PIPE_DEPTH=" << G::TMEM_PIPE_DEPTH << "\n";
    std::cout << "Total number of tasks: " << (M / G::Mb * N / G::Nb) << "\n";
    std::cout << "Number of iterations per task: " << (K / G::Kb) << "\n";

    // Sleep for 50 ms to limit power consumption and thermals
    usleep(50000);

    // Calculate arg_group_size
    const int arg_size = 2 * (M * K + N * K + M * N);
    const int l2_cache_size = 128 * 1024 * 1024;
    const int ideal_arg_size = l2_cache_size * 3;
    const int arg_group_count = arg_size > ideal_arg_size ? 1 : (ideal_arg_size / arg_size) + 1;

    // Allocate host memory
    std::vector<float> h_A(M * K * arg_group_count);
    std::vector<float> h_B(K * N * arg_group_count);
    std::vector<float> h_C_ref(M * N);
    std::cout << "Allocated host memory" << std::endl;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(-0.5, 0.5);

    // Initialize matrices with random values
    for (int i = 0; i < M * K * arg_group_count; ++i) h_A[i] = dis(gen);
    for (int i = 0; i < K * N * arg_group_count; ++i) h_B[i] = dis(gen);
    std::cout << "Initialized matrices" << std::endl;

    // Perform CPU matrix multiplication for reference
    if (check_correctness) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++)
                    sum += h_A[i * K + k] * h_B[j * K + k];
                h_C_ref[i * N + j] = sum;
            }
        }
        std::cout << "Performed CPU matrix multiplication" << std::endl;
    }

    // Allocate device memory
    std::vector<__nv_bfloat16*> d_A(arg_group_count);
    std::vector<__nv_bfloat16*> d_B(arg_group_count);
    std::vector<__nv_bfloat16*> d_C(arg_group_count);
    for (int i = 0; i < arg_group_count; i++) {
        cudaMalloc(&d_A[i], M*K*sizeof(__nv_bfloat16));
        cudaMalloc(&d_B[i], K*N*sizeof(__nv_bfloat16));
        cudaMalloc(&d_C[i], M*N*sizeof(__nv_bfloat16));
    }
    std::cout << "Allocated device memory" << std::endl;

    // Convert to __nv_bfloat16 and copy to device
    std::vector<__nv_bfloat16> h_A_bf16(M * K * arg_group_count);
    std::vector<__nv_bfloat16> h_B_bf16(K * N * arg_group_count);
    for (int i = 0; i < M * K * arg_group_count; ++i) h_A_bf16[i] = __float2bfloat16(h_A[i]);
    for (int i = 0; i < K * N * arg_group_count; ++i) h_B_bf16[i] = __float2bfloat16(h_B[i]);
    for (int i = 0; i < arg_group_count; i++) {
        cudaMemcpy(d_A[i], &h_A_bf16[i*M*K], M*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B[i], &h_B_bf16[i*N*K], N*K*sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    }
    std::cout << "Copied matrices to device" << std::endl;

    // Prepare kernel inputs
    std::vector<CUtensorMap> a_tmaps(arg_group_count);
    std::vector<CUtensorMap> b_tmaps(arg_group_count);
    for (int i = 0; i < arg_group_count; i++) {
        create_tensor_map(&a_tmaps[i], d_A[i], 1, 1, M, K);
        create_tensor_map(&b_tmaps[i], d_B[i], 1, 1, N, K);
    }
    G g;

    // Set kernel attributes
    cudaFuncSetAttribute(kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, g.dynamic_shared_memory());

    // Number of iterations
    int num_warmups = ncu ? 0 : 500;
    int num_iters = ncu ? 1 : 100;

    // Warmup
    for(int i = 0; i < num_warmups; i++) {
        int idx = i % arg_group_count;
        kernel<G><<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(a_tmaps[idx], b_tmaps[idx]);
    }

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for(int i = 0; i < num_iters; i++) {
        int idx = i % arg_group_count;
        kernel<G><<<g.grid(), g.block(), g.dynamic_shared_memory()>>>(a_tmaps[idx], b_tmaps[idx]);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate duration and TFLOPs
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double microseconds = milliseconds * 1000.0 / num_iters;
    double flops = double(2.0) * M * N * K;
    double tflops = (flops / microseconds) / 1e6;
    std::cout << "Average kernel execution time: " << microseconds << " us\n";
    std::cout << "Achieved performance: " << tflops << " TFLOPs\n";

    // Clean up
    for (int i = 0; i < arg_group_count; i++) {
        cudaFree(d_A[i]);
        cudaFree(d_B[i]);
        cudaFree(d_C[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return tflops;
}

__host__ int main() {
    int N;
    bool check_correctness = false;
    bool ncu = false;

    // Template parameters: SUPERGROUP_SIZE, Mb, Nb, Kb, SMEM_PIPE_DEPTH, MMA_PIPE_DEPTH, TMEM_PIPE_DEPTH
    // N = 1024;
    // run_benchmark<globals<4, 128, 128, 128, 4, 2, 2>>(N, N, N, check_correctness, ncu);
    // N = 2048;
    // run_benchmark<globals<4, 128, 256, 64, 4, 2, 8>>(N, N, N, check_correctness, ncu);
    // N = 4096;
    // run_benchmark<globals<4, 128, 256, 64, 5, 2, 2>>(N, N, N, check_correctness, ncu);
    N = 8192;
    run_benchmark<globals<8, 128, 256, 64, 6, 2, 8>>(N, N, N, check_correctness, ncu);
    // N = 16384;
    // run_benchmark<globals<8, 128, 256, 64, 4, 2, 8>>(N, N, N, check_correctness, ncu);

    return 0;
}
