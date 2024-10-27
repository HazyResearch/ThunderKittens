#pragma once

namespace kittens {
namespace prototype {

struct empty {
    __device__ empty& operator*() { return *this; }
    __device__ const empty& operator*() const { return *this; }
};

// A struct for returning padded versions of structs (used for shared memory allocation)
template<typename T, int ALIGN=1024> struct __align__(ALIGN) padder {
    T val;
    __device__ padder(T _val) : val(_val) {}
    __device__ T& operator*() { return val; }
    __device__ const T& operator*() const { return val; }
};

// A concept for a pc layout. All pc layouts must have these two types defined
template<typename T> concept kittens_layout = requires {
    typename T::globals;
    typename T::input_block;
};

// Now we need to be able to put together a complete pc layout from a (possibly partially specialized) pc layout.
namespace detail {

// Get an int parameter from the layout
#define FLAG_GETTER(flag_name, default_value)                                         \
template<typename T> concept has_##flag_name##_flag = requires { T::flag_name; };     \
template<typename T> constexpr int flag_name##_v = default_value;                     \
template<has_##flag_name##_flag T> constexpr int flag_name##_v<T> = T::flag_name;

// How to pad structs to force alignment (set to 1024 by default)
FLAG_GETTER(FORCE_ALIGN, 1024)
// Whether to print debug information (set to false by default)
FLAG_GETTER(DEBUG, 0)
// How many blocks to colocate per SM.
FLAG_GETTER(NUM_BLOCKS, 1)
// How many consumer warps to run per SM.
FLAG_GETTER(NUM_CONSUMER_WARPS, 8)
// How many producer warps to run per SM.
FLAG_GETTER(NUM_PRODUCER_WARPS, 4)
// Maximum amount of shared memory to use, per SM.
FLAG_GETTER(MAX_SHARED_MEMORY, ((kittens::MAX_SHARED_MEMORY/NUM_BLOCKS_v<T>)-512))
// How many input pipe stages to use, per block.
FLAG_GETTER(INPUT_PIPE_STAGES, 1)
// How many output pipe stages to use, per block.
FLAG_GETTER(OUTPUT_PIPE_STAGES, 1)
// How many arrivals to initialize for each consumer semaphore.
FLAG_GETTER(CONSUMER_BARRIER_ARRIVALS, NUM_CONSUMER_WARPS_v<T>)
// How many arrivals to initialize for each producer semaphore.
FLAG_GETTER(PRODUCER_BARRIER_ARRIVALS, NUM_PRODUCER_WARPS_v<T>)
// Some handy constants
template<typename T> constexpr int NUM_WARPS_v = NUM_CONSUMER_WARPS_v<T> + NUM_PRODUCER_WARPS_v<T>;
template<typename T> constexpr int NUM_THREADS_v = NUM_WARPS_v<T> * 32;
template<typename T> constexpr int NUM_CONSUMER_WARPGROUPS_v = NUM_CONSUMER_WARPS_v<T> / kittens::WARPGROUP_WARPS;

// Macro to generate the block getter template
#define BLOCK_GETTER(block_name)                                                                                    \
template<typename T> concept has_##block_name##_block = kittens_layout<T> && requires { typename T::block_name; };  \
template<typename T>                 struct block_name##_getter    { using type = empty; };                         \
template<has_##block_name##_block T> struct block_name##_getter<T> { using type = typename T::block_name; };        \
template<typename T> using block_name##_t = typename block_name##_getter<T>::type;

// Get the input block type from a layout
BLOCK_GETTER(input_block)
// Get the output block type from a layout
BLOCK_GETTER(output_block)
// Get the scratch block type from a layout
BLOCK_GETTER(scratch_block)
// Get the finish block type from a layout
BLOCK_GETTER(finish_block)
// Get the common state type from a layout
BLOCK_GETTER(common_state)
// Get the producer state type from a layout
BLOCK_GETTER(producer_state)
// Get the consumer state type from a layout
BLOCK_GETTER(consumer_state)

}

// Complete the pc layout by filling in the missing types
template<kittens_layout T> struct complete_kittens_layout : T {
    // In global memory
    using globals_t        = T::globals;
    // In shared memory
    using input_block_t    = typename detail::input_block_t<T>;
    using output_block_t   = typename detail::output_block_t<T>;
    using scratch_block_t  = typename detail::scratch_block_t<T>;
    using finish_block_t   = typename detail::finish_block_t<T>;
    // In registers
    using common_state_t   = typename detail::common_state_t<T>;
    using producer_state_t = typename detail::producer_state_t<T>;
    using consumer_state_t = typename detail::consumer_state_t<T>;
    
    // Allocation types to ensure memory alignments
    constexpr static int FORCE_ALIGN = detail::FORCE_ALIGN_v<T>;
    using input_alloc_block_t = padder<input_block_t, FORCE_ALIGN>;
    using output_alloc_block_t = padder<output_block_t, FORCE_ALIGN>;
    using scratch_alloc_block_t = padder<scratch_block_t, FORCE_ALIGN>;
};

}
}