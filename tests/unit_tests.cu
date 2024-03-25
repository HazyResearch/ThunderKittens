#include "testing_commons.cuh"

#include "warp/warp.impl"
// #include "warp/st_vec_tests.impl"
// #include "warp/rt_layout_tests.impl"
// #include "warp/st_layout_tests.impl"
// #include "warp/st_subtile_tests.impl"
// #include "warp/map_tests.impl"
// #include "warp/reduction_tests.impl"
// #include "warp/broadcast_tests.impl"
// #include "warp/mma_tests.impl"
// #include "block/global_to_shared.impl"

// #ifdef KITTENS_HOPPER
// #include "warp/tma_tests.impl"

// #include "warpgroup/wgmma_tests.impl"
// #include "warpgroup/tall_wgmma_tests.impl"

// #include "block/dsmem.impl"

// #include "integration/wgmma_tma_tests.impl"
// #endif

int main(int argc, char **argv) {

    should_write_outputs = argc>1; // write outputs if user says so

    test_data data;

    warp::tests(data);

    std::cout << " ---------------  SUMMARY  ---------------\n";

    std::cout << "Failed tests:\n";
    int passes = 0, fails = 0, invalids = 0;
    for(auto it = data.begin(); it != data.end(); it++) {
        if(it->result == test_result::PASSED)  passes++;
        if(it->result == test_result::INVALID) invalids++;
        if(it->result == test_result::FAILED) {
            fails++;
            std::cout << it->label << std::endl;
        }
    }
    if(fails == 0) std::cout << "ALL TESTS PASSED!\n";

    std::cout << passes   << " tests passed\n";
    std::cout << fails    << " tests failed\n";
    std::cout << invalids << " tests skipped (this is normal)\n";

    return 0;
}