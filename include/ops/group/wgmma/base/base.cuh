#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {
namespace wgmma {

// templated wrapper for PTX
template<typename T_D, typename T_AB, int cols, int trans_a, int trans_b, int inv=1>
struct base {
    template<int scale_b=1> __device__ static inline void rt_st(
        rt<T_D, 16, cols, ducks::rt_layout::row> &dst,
        const rt<T_AB, 16, cols, ducks::rt_layout::row> & a_rt,
        const uint64_t b_st_desc,
        int scale_d = 1
    );
    template<int scale_b=1> __device__ static inline void st_st(
        rt<T_D, 16, cols, ducks::rt_layout::row> &dst,
        const uint64_t a_st_desc,
        const uint64_t b_st_desc,
        int scale_d = 1
    );
};

// all the ptx's
#include "64x16.impl"
#include "64x32.impl"
#include "64x48.impl"
#include "64x64.impl"
#include "64x80.impl"
#include "64x96.impl"
#include "64x112.impl"
#include "64x128.impl"
#include "64x144.impl"
#include "64x160.impl"
#include "64x176.impl"
#include "64x192.impl"
#include "64x208.impl"
#include "64x224.impl"
#include "64x240.impl"
#include "64x256.impl"

} // namespace wgmma

namespace ducks {
namespace wgmma {
// input refers to either an ST directly or to a pre-generated descriptor, which can save cycles in certain situations.
template<typename T> concept input = ducks::st::all<T> || (requires {typename T::identifier;} && std::is_same_v<typename T::identifier, ducks::st_descriptor::identifier>);
template<typename T> concept complex_input = ducks::cst::all<T>;
namespace detail {
template<typename T> struct st_getter { using type = typename T::ST; };
template<ducks::st::all T> struct st_getter<T> { using type = T; };
template<ducks::cst::all T> struct st_getter<T> { using type = T::component; };
template<typename T> using get_st = typename st_getter<T>::type;
} // namespace detail
} // namespace wgmma
} // namespace ducks
} // namespace kittens