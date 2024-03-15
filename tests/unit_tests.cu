#define WIDTH 2
#define HEIGHT 4
#define WARPSIZE 32

#include "testing_commons.cuh"

#include "warp/mem_tests.impl"
#include "warp/st_vec_tests.impl"
#include "warp/rt_layout_tests.impl"
#include "warp/st_layout_tests.impl"
#include "warp/st_subtile_tests.impl"
#include "warp/map_tests.impl"
#include "warp/reduction_tests.impl"
#include "warp/broadcast_tests.impl"
#include "warp/mma_tests.impl"

#ifdef KITTENS_HOPPER
#include "warp/tma_tests.impl"

#include "warpgroup/wgmma_tests.impl"
#include "warpgroup/tall_wgmma_tests.impl"

#include "block/dsmem.impl"

#include "integration/wgmma_tma_tests.impl"
#endif

int main(int argc, char **argv) {

    should_write_outputs = argc>1; // write outputs if user says so

    int failures = 0;
    
    std::cout << " ---------------  BEGINNING WARP TESTS  ---------------\n";
    failures += mem_tests();
    failures += st_vec_tests();
    failures += rt_layout_tests();
    failures += st_layout_tests();
    failures += st_subtile_tests();
    failures += map_tests();
    failures += reduction_tests();
    failures += broadcast_tests();
    failures += mma_tests();
#ifdef KITTENS_HOPPER
    failures += tma_tests();
    std::cout << " ---------------  BEGINNING WARPGROUP TESTS  ---------------\n";
    failures += wgmma_tests();
    failures += tall_wgmma_tests();
    std::cout << " ---------------  BEGINNING BLOCK TESTS  ---------------\n";
    failures += dsmem_tests();
    std::cout << " ---------------  BEGINNING INTEGRATION TESTS  ---------------\n";
    failures += wgmma_tma_tests();
#endif

    std::cout << " ---------------  SUMMARY  ---------------\n";
    if(failures == 0) std::cout << "ALL TESTS PASSED!\n";
    else std::cout << failures << " TESTS FAILED :(\n";

    return 0;
}