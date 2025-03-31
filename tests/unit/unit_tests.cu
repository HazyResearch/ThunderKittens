#include "testing_flags.cuh"
#include "testing_commons.cuh"

#ifdef TEST_THREAD
#include "thread/thread.cuh"
#endif
#ifdef TEST_GROUP
#include "group/group.cuh"
#endif

int main(int argc, char **argv) {

    should_write_outputs = argc>1; // write outputs if user says so

    test_data data;

#ifdef TEST_THREAD
    thread::tests(data);
#else
    std::cout << "INFO: Skipping ops/thread tests!\n" << std::endl;
#endif
#ifdef TEST_GROUP
    group::tests(data);
#else
    std::cout << "INFO: Skipping ops/group tests!\n" << std::endl;
#endif

    std::cout << " ------------------------------     Summary     ------------------------------\n"  << std::endl;

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
    std::cout << std::endl;

    std::cout << invalids << " test template configurations deemed invalid (this is normal)\n";
    std::cout << passes   << " tests passed\n";
    std::cout << fails    << " tests failed\n";

    return 0;
}