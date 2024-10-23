/**
 * @file
 * @brief General memory utilities not specialized for either tiles or vectors.
 */

#pragma once

namespace kittens {

/* ----------   To prevent generic addressing, PTX  ---------- */

template<typename T> struct move {
    template<typename U> __device__ static inline void lds(T& dst, U* src);
    template<typename U> __device__ static inline void ldg(T& dst, U* src);
    template<typename U> __device__ static inline void sts(U* dst, T& src);
    template<typename U> __device__ static inline void stg(U* dst, T& src);
};
// unpacked types
template<> struct move<bf16> {
    template<typename U> __device__ static inline void lds(bf16& dst, U* src) {
        asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "l"(src) : "memory");
    }
    template<typename U> __device__ static inline void ldg(bf16& dst, U* src) {
        asm volatile("ld.global.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "l"(src) : "memory");
    }
    template<typename U> __device__ static inline void sts(U* dst, bf16& src) {
        asm volatile("st.shared.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "l"(dst) : "memory");
    }
    template<typename U> __device__ static inline void stg(U* dst, bf16& src) {
        asm volatile("st.global.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "l"(dst) : "memory");
    }
};
template<> struct move<half> {
    template<typename U> __device__ static inline void lds(half& dst, U* src) {
        asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "l"(src) : "memory");
    }
    template<typename U> __device__ static inline void sts(U* dst, half& src) {
        asm volatile("st.shared.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "l"(dst) : "memory");
    }
    template<typename U> __device__ static inline void ldg(half& dst, U* src) {
        asm volatile("ld.global.b16 %0, [%1];\n" : "=h"(*(uint16_t*)&dst) : "l"(src) : "memory");
    }
    template<typename U> __device__ static inline void stg(U* dst, half& src) {
        asm volatile("st.global.b16 [%1], %0;\n" : : "h"(*(uint16_t*)&src), "l"(dst) : "memory");
    }
};
template<> struct move<float> {
    template<typename U> __device__ static inline void lds(float& dst, U* src) {
        asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(dst) : "l"(src) : "memory");
    }
    template<typename U> __device__ static inline void sts(U* dst, float& src) {
        asm volatile("st.shared.f32 [%1], %0;\n" : : "f"(src), "l"(dst) : "memory");
    }
    template<typename U> __device__ static inline void ldg(float& dst, U* src) {
        asm volatile("ld.global.f32 %0, [%1];\n" : "=f"(dst) : "l"(src) : "memory");
    }
    template<typename U> __device__ static inline void stg(U* dst, float& src) {
        asm volatile("st.global.f32 [%1], %0;\n" : : "f"(src), "l"(dst) : "memory");
    }
};
// packed types
template<> struct move<bf16_2> {
    template<typename U> __device__ static inline void lds(bf16_2& dst, U* src) {
        asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "l"(src) : "memory");
    }
    template<typename U> __device__ static inline void sts(U* dst, bf16_2& src) {
        asm volatile("st.shared.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "l"(dst) : "memory");
    }
    template<typename U> __device__ static inline void ldg(bf16_2& dst, U* src) {
        asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "l"(src) : "memory");
    }
    template<typename U> __device__ static inline void stg(U* dst, bf16_2& src) {
        asm volatile("st.global.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "l"(dst) : "memory");
    }
};
template<> struct move<half_2> {
    template<typename U> __device__ static inline void lds(half_2& dst, U* src) {
        asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "l"(src) : "memory");
    }
    template<typename U> __device__ static inline void sts(U* dst, half_2& src) {
        asm volatile("st.shared.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "l"(dst) : "memory");
    }
    template<typename U> __device__ static inline void ldg(half_2& dst, U* src) {
        asm volatile("ld.global.b32 %0, [%1];\n" : "=r"(*(uint32_t*)&dst) : "l"(src) : "memory");
    }
    template<typename U> __device__ static inline void stg(U* dst, half_2& src) {
        asm volatile("st.global.b32 [%1], %0;\n" : : "r"(*(uint32_t*)&src), "l"(dst) : "memory");
    }
};
template<> struct move<float2> {
    template<typename U> __device__ static inline void lds(float2& dst, U* src) {
        asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];\n" : "=f"(dst.x), "=f"(dst.y) : "l"(src) : "memory");
    }
    template<typename U> __device__ static inline void sts(U* dst, float2& src) {
        asm volatile("st.shared.v2.f32 [%2], {%0, %1};\n" : : "f"(src.x), "f"(src.y), "l"(dst) : "memory");
    }
    template<typename U> __device__ static inline void ldg(float2& dst, U* src) {
        asm volatile("ld.global.v2.f32 {%0, %1}, [%2];\n" : "=f"(dst.x), "=f"(dst.y) : "l"(src) : "memory");
    }
    template<typename U> __device__ static inline void stg(U* dst, float2& src) {
        asm volatile("st.global.v2.f32 [%2], {%0, %1};\n" : : "f"(src.x), "f"(src.y), "l"(dst) : "memory");
    }
};
template<> struct move<float4> {
    template<typename U> __device__ static inline void lds(float4& dst, U* src) {
        asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w) : "l"(src) : "memory");
    }
    template<typename U> __device__ static inline void sts(U* dst, float4& src) {
        asm volatile("st.shared.v4.f32 [%4], {%0, %1, %2, %3};\n" : : "f"(src.x), "f"(src.y), "f"(src.z), "f"(src.w), "l"(dst) : "memory");
    }
    template<typename U> __device__ static inline void ldg(float4& dst, U* src) {
        asm volatile("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(dst.x), "=f"(dst.y), "=f"(dst.z), "=f"(dst.w) : "l"(src) : "memory");
    }
    template<typename U> __device__ static inline void stg(U* dst, float4& src) {
        asm volatile("st.global.v4.f32 [%4], {%0, %1, %2, %3};\n" : : "f"(src.x), "f"(src.y), "f"(src.z), "f"(src.w), "l"(dst) : "memory");
    }
};

/* ----------   Generic (non-Hopper specific) semaphore functions  ---------- */

struct semaphore {
private:
    uint64_t _;
}; // note that this is an opaque type, so the value should not be accessed directly.

/**
 * @brief Initializes a synchronization semaphore with a transaction count and sets the expected number of bytes.
 *
 * This function sets up a semaphore that is used to synchronize threads within a block during asynchronous operations.
 * It initializes the semaphore with a thread count semaphore.
 *
 * Additionally, if it is given a shared tile type, it will also call `set_bytes` to prepare for the memory transaction.
 *
 * @param[out] semaphore The semaphore variable to initialize.
 * @param[in] tc The thread counter for the semaphore.
 */
__device__ static inline void init_semaphore(semaphore& bar, int thread_count, int transaction_count) {
    if (::kittens::laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

        asm volatile (
            "mbarrier.init.shared::cta.b64 [%0], %1;\n"
            :: "r"(bar_ptr), "r"(thread_count+transaction_count)
        );
    }
}
/**
 * @brief Invalidate an mbarrier
 *
 * @param[out] semaphore The semaphore variable to initialize.
 * @param[in] tc The thread counter for the semaphore.
 */
__device__ static inline void invalidate_semaphore(semaphore& bar) {
    if (::kittens::laneid() == 0) {
        void const* const ptr = &bar;
        uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 
        asm volatile (
            "mbarrier.inval.shared::cta.b64 [%0];\n"
            :: "r"(bar_ptr)
        );
    }
}

/**
* @brief Arrives at a semaphore.
*
* Marks a warp arrival at an mbarrier
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline void arrive(semaphore& bar) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar)); 
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
        :
        : "r"(mbar_ptr)
        : "memory"
    );
}

// /**
// * @brief Arrives at a semaphore.
// *
// * Marks a warp arrival at an mbarrier
// *
// * @param semaphore Reference to the semaphore variable.
// * @param kPhaseBit The phase bit used for the semaphore.
// */
// template<int num_warps> __device__ static inline void arrive(int semaphore_id) {
//     uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar)); 
//     asm volatile (
//         "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
//         :
//         : "r"(mbar_ptr)
//         : "memory"
//     );
// }

#ifdef KITTENS_HOPPER
/**
* @brief Arrives at a semaphore.
*
* Marks a warp arrival at an mbarrier
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline void arrive(semaphore& bar, uint32_t count) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    asm volatile (
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}
#endif

/**
* @brief Waits for the requested semaphore phase.
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline void wait(semaphore& bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr)); 

#ifdef KITTENS_HOPPER
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
#else
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
        "@P1                       bra.uni DONE;\n"
        "nanosleep.u32 5;\n" // wait a few nanoseconds on pre-Hopper architectures to save instruction issue slots
        "bra.uni                   LAB_WAIT;\n"
        "DONE:\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
#endif
}

/**
* @brief Checks if the requested semaphore phase is ready.
*
* @param semaphore Reference to the semaphore variable.
* @param kPhaseBit The phase bit used for the semaphore.
*/
__device__ static inline int test_wait(semaphore& bar, int kPhaseBit) {
    void const* const ptr = &bar;
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    int result;
    asm volatile (
        "{\n"
        ".reg .pred P1;\n"
        "mbarrier.test_wait.parity.shared::cta.b64 P1, [%1], %2;\n"
        "selp.u32 %0,1,0,P1;"
        "}\n"
        : "=r"(result)
        : "r"(mbar_ptr), "r"(kPhaseBit)
    );
    return result;
}

__device__ static inline void arrive_and_wait(semaphore& bar, int kPhaseBit) {
    arrive(bar);
    wait(bar, kPhaseBit);
}

template<int N=0> __device__ static inline void load_async_wait() { // for completing (non-TMA) async loads
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N) : "memory");
    __syncwarp();
}

// meant to be used only with shared tiles and shared vectors
namespace detail {
template<typename T> struct size_info {
    static constexpr uint32_t bytes    = sizeof(std::remove_reference_t<T>);
};
template<ducks::st::all ST> struct size_info<ST> {
    static constexpr uint32_t elements = ST::num_elements;
    static constexpr uint32_t bytes    = ST::num_elements * sizeof(typename ST::dtype);
};
template<ducks::sv::all SV> struct size_info<SV> {
    static constexpr uint32_t elements = SV::length;
    static constexpr uint32_t bytes    = SV::length * sizeof(typename SV::dtype);
};
}
template<typename... Args>                       inline constexpr uint32_t size_bytes             = 0; // base case
template<typename T, typename... Args>           inline constexpr uint32_t size_bytes<T, Args...> = detail::size_info<T>::bytes + size_bytes<Args...>; // recursive case

} // namespace kittens

#ifdef KITTENS_HOPPER
#include "tma.cuh"
#endif
