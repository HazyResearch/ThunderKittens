#include "block.cuh"

#ifdef TEST_BLOCK

using namespace block;

void block::tests(test_data &results) {
    std::cout << "\n ------------------------------     Starting ops/block tests!     ------------------------------\n" << std::endl;
#ifdef TEST_BLOCK_MEMORY
    memory::tests(results);
#endif
}

#endif