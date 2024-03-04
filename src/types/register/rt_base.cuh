#pragma once

#include <type_traits>

#include "../../common/common.cuh"
#include "rt_layouts.cuh"

namespace kittens {

/* ----------  BASE 16x16 SUBTILE STRUCT  ---------- */

struct rt_base_id {};

// base register tile is 16x16
template<packed_type T2, rt_layout _layout> struct rt_base {
    using identifier = rt_base_id;
    using layout = _layout;
    using dtype = T2;

    static constexpr int tile_size            = 16;
    static constexpr int num_elements         = tile_size*tile_size; // 256
    static constexpr int elements_per_thread  = num_elements / 32; // 8

    static constexpr int packed_per_thread    = elements_per_thread / base_types::packing<T2>::num(); // 4
    static constexpr int registers_per_thread = packed_per_thread * sizeof(T2) / 4; // 4 or 8, registers are 32-bit words

    // using an array type for both makes the access a bit more regular, which simplifies rt_vector.cuh
    // all this should be hidden from the user, anyways.
    using col_type = std::conditional<layout::row, T2[1], T2[2]>::type; // for holding row reductions
    using row_type = std::conditional<layout::row, T2[2], T2[1]>::type; // for holding column reductions

    T2 data[packed_per_thread];
};

/* ----------  WRAPPERS FOR PRETTINESS  ---------- */

template<rt_layout L=rt_row_layout> using rt_base_fl = rt_base<float2, L>; // Note float2! Otherwise you will get bugs.
template<rt_layout L=rt_row_layout> using rt_base_bf = rt_base<bf16_2, L>;

}