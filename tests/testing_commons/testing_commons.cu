#include "testing_commons.cuh"

// Explicit specializations

template<> std::string layout_name<kittens::ducks::st_layout::naive      >() { return "naive";       }
template<> std::string layout_name<kittens::ducks::st_layout::xor_swizzle>() { return "xor_swizzle"; }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_0b   >() { return "wgmma_0b";    }
template<> std::string layout_name<kittens::ducks::st_layout::wgmma_32b  >() { return "wgmma_32b";   }