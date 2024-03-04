#define WIDTH 2
#define HEIGHT 4

#include "testing_commons.cuh"

#include "warp/mem_tests.impl"
#include "warp/st_vec_tests.impl"
#include "warp/rt_layout_tests.impl"
#include "warp/st_layout_tests.impl"
#include "warp/map_tests.impl"
#include "warp/reduction_tests.impl"
#include "warp/mma_tests.impl"
#include "warp/tma_tests.impl"

#include "warpgroup/wgmma_tests.impl"

#include "block/dsmem.impl"


int main() {

    int failures = 0;
    
    std::cout << " ---------------  BEGINNING WARP TESTS  ---------------\n";
    // failures += mem_tests();
    // failures += st_vec_tests();
    // failures += rt_layout_tests();
    // failures += st_layout_tests();
    // failures += map_tests();
    // failures += reduction_tests();
    // failures += mma_tests();
    failures += tma_tests();
    // std::cout << " ---------------  BEGINNING WARPGROUP TESTS  ---------------\n";
    // failures += wgmma_tests();
    // std::cout << " ---------------  BEGINNING BLOCK TESTS  ---------------\n";
    // failures += dsmem_tests();

    std::cout << " ---------------  SUMMARY  ---------------\n";
    if(failures == 0) std::cout << "ALL TESTS PASSED!\n";
    else std::cout << failures << " TESTS FAILED :(\n";

    return 0;
}