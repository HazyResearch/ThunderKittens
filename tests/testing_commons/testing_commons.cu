#include "testing_commons.cuh"

// Explicit specializations

template<> std::string layout_name<kittens::ducks::st_layout::naive          >() { return "naive";           }
template<> std::string layout_name<kittens::ducks::st_layout::tma_swizzle    >() { return "tma_swizzle";     }
template<> std::string layout_name<kittens::ducks::st_layout::xor_swizzle    >() { return "xor_swizzle";     }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_row_0b   >() { return "wgmma_row_0b";    }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_row_32b  >() { return "wgmma_row_32b";   }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_col_t_0b >() { return "wgmma_col_t_0b";  }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_col_t_32b>() { return "wgmma_col_t_32b"; }