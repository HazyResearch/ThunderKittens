#include "testing_flags.cuh"
#include "testing_commons.cuh"

#ifdef TEST_WARP
#include "warp/warp.cuh"
#endif
#ifdef TEST_WARPGROUP
#include "warpgroup/warpgroup.cuh"
#endif

int main(int argc, char **argv) {

    should_write_outputs = argc>1; // write outputs if user says so

    test_data data;

#ifdef TEST_WARP
    warp::tests(data);
#endif
#ifdef TEST_WARPGROUP
    warpgroup::tests(data);
#endif

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

    std::cout << invalids << " tests skipped (this is normal, and refers to tests that cannot be compiled due to invalid template parameters.)\n";
    std::cout << passes   << " tests passed\n";
    std::cout << fails    << " tests failed\n";

    return 0;
}