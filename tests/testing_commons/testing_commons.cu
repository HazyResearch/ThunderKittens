#include "testing_commons.cuh"

// Explicit specializations

template<> std::string layout_name<kittens::ducks::st_layout::naive            >() { return "naive";            }
template<> std::string layout_name<kittens::ducks::st_layout::swizzle          >() { return "swizzle";          }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_swizzle    >() { return "wgmma_swizzle";    }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_interleave >() { return "wgmma_interleave"; }