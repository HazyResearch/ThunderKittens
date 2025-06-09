#pragma once

#define ONEAPI_BACKEND_LEVEL_ZERO_EXT
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../ops/ops.dp.hpp"
#include "club.dp.hpp"
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, char const* const func, char const* const file,
           int const line)
{
    /*
    DPCT1000:608: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (err != 0)
    {
        /*
        DPCT1001:607: The statement could not be removed.
        */
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        /*
        DPCT1009:609: SYCL reports errors using exceptions and does not use
        error codes. Please replace the "get_error_string_dummy(...)" with a
        real error-handling function.
        */
        std::cerr << dpct::get_error_string_dummy(err) << " " << func
                  << std::endl;
        //std::exit(EXIT_FAILURE);
    }
}