/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared tiles from and storing to global memory. 
 */

template<ducks::st::all ST, ducks::gl::all GL>
__device__ static inline void load(ST &dst, const GL &src, const coord &idx) {
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src.template get<ST>(idx);
    const int row_stride = src.row_stride();
    
    // each thread needs to do 1 call per width*height / N_WARPS
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % GROUP_THREADS;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = ((dst.height * dst.width + (N_WARPS-1))) * TILE_DIM*TILE_DIM / (N_WARPS*WARP_THREADS*elem_per_memcpy); // round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * GROUP_THREADS + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        if (i<total_calls-1 || row<dst.rows) { // the first condition lets compiler short-circuit on unrolled iters
            float4 tmp;
            move<float4>::ldg(tmp, &src_ptr[row*row_stride + col]);
            move<float4>::sts(&dst[{row, col}], tmp);
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL>
__device__ static inline void store(GL &dst, const ST &src, const coord &idx) {
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst.template get<ST>(idx);
    const int row_stride = dst.row_stride();

    int laneid = threadIdx.x % GROUP_THREADS;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    int memcpy_per_row = src.cols / elem_per_memcpy;
    int total_calls = ((src.height * src.width + (N_WARPS-1))) * TILE_DIM*TILE_DIM / (N_WARPS*WARP_THREADS*elem_per_memcpy); // round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * GROUP_THREADS + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % src.cols;

        if (i<total_calls-1 || row<src.rows) { // the first condition lets compiler short-circuit on unrolled iters
            float4 tmp;
            move<float4>::lds(tmp, &src[{row, col}]);
            move<float4>::stg(&dst_ptr[row*row_stride + col], tmp);
        }
    }
}

template<ducks::st::all ST, ducks::gl::all GL>
__device__ static inline void load_async(ST &dst, const GL &src, const coord &idx) {
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src.template get<ST>(idx);
    const int row_stride = src.row_stride();

    // each thread needs to do 1 call per width*height / N_WARPS
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % GROUP_THREADS;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = ((dst.height * dst.width + (N_WARPS-1))) * TILE_DIM*TILE_DIM / (N_WARPS*WARP_THREADS*elem_per_memcpy); // round up

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * GROUP_THREADS + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        if (i<total_calls-1 || row<dst.rows) { // the first condition lets compiler short-circuit on unrolled iters
            asm volatile(
                "cp.async.cg.shared::cta.global [%0], [%1], 16;\n"
                :: "l"(__cvta_generic_to_shared(&dst[{row, col}])), "l"(&src_ptr[row*row_stride + col])
                : "memory"
            );
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}