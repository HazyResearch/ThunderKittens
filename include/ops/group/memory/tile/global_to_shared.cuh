/**
 * @file
 * @brief Group (collaborative warp) ops for loading shared tiles from and storing to global memory. 
 */

template<ducks::st::all ST, ducks::gl::all GL, int axis=2>
__device__ static inline void load(ST &dst, const GL &src, const coord &idx, int load_rows=ST::rows, typename ST::dtype fill_value=0) {
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src.template get<ST, axis>(idx);
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    const int row_stride = src.template stride<axis>();
    
    // each thread needs to do 1 call per width*height / N_WARPS
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % GROUP_THREADS;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = ((dst.height * dst.width + (N_WARPS-1))) * TILE_DIM*TILE_DIM / (N_WARPS*WARP_THREADS*elem_per_memcpy); // round up

    float4 packed_fill; // create a coalesced fill value to store repeatedly with minimal address calculation / STS instructions
    if (load_rows != ST::rows) {
        // Fill in a static array of size float4 that we can store repeatedly, to save address calculations
        static constexpr int fill_size = sizeof(float4)/sizeof(typename ST::dtype);
        typename ST::dtype fill_array[fill_size];
        #pragma unroll
        for (int i = 0; i < fill_size; i++) {
            fill_array[i] = fill_value;
        }
        packed_fill = *reinterpret_cast<float4*>(fill_array);
    }

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * GROUP_THREADS + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        if (row < load_rows) { // the first condition lets compiler short-circuit on unrolled iters
            float4 tmp;
            move<float4>::ldg(tmp, (float4*)&src_ptr[row*row_stride + col]);
            move<float4>::sts(dst.idx(dst_ptr, {row, col}), tmp);
        }
        else if (row < ST::rows) {
            move<float4>::sts(dst.idx(dst_ptr, {row, col}), packed_fill); // use the default value
        }
    }
}
template<ducks::st::all ST, ducks::gl::all GL, int axis=2>
__device__ static inline void store(const GL &dst, const ST &src, const coord &idx, int store_rows=ST::rows) {
    typename GL::dtype *dst_ptr = (typename GL::dtype*)&dst.template get<ST, axis>(idx);
    uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&src.data[0]));
    const int row_stride = dst.template stride<axis>();

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

        if (i<total_calls-1 || row<store_rows) { // the first condition lets compiler short-circuit on unrolled iters
            float4 tmp;
            move<float4>::lds(tmp, src.idx(src_ptr, {row, col}));
            move<float4>::stg((float4*)&dst_ptr[row*row_stride + col], tmp);
        }
    }
}

template<ducks::st::all ST, ducks::gl::all GL, int axis=2>
__device__ static inline void load_async(ST &dst, const GL &src, const coord &idx, int load_rows=ST::rows, typename ST::dtype fill_value=0) {
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src.template get<ST, axis>(idx);
    uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
    const int row_stride = src.template stride<axis>();

    // each thread needs to do 1 call per width*height / N_WARPS
    // attempting to improve striping into dram
    // each lane of the warp should store sequential into dram

    int laneid = threadIdx.x % GROUP_THREADS;

    // we can handle this many rows each time we run a memcpy_async
    int elem_per_memcpy = sizeof(float4)/sizeof(typename ST::dtype);
    int memcpy_per_row = dst.cols / elem_per_memcpy;
    int total_calls = ((dst.height * dst.width + (N_WARPS-1))) * TILE_DIM*TILE_DIM / (N_WARPS*WARP_THREADS*elem_per_memcpy); // round up

    float4 packed_fill; // create a coalesced fill value to store repeatedly with minimal address calculation / STS instructions
    if (dst.rows != ST::rows) {
        static constexpr int fill_size = sizeof(float4)/sizeof(typename ST::dtype);
        typename ST::dtype fill_array[fill_size];
        #pragma unroll
        for (int i = 0; i < fill_size; i++) {
            fill_array[i] = fill_value;
        }
        packed_fill = *reinterpret_cast<float4*>(fill_array);
    }

    #pragma unroll
    for(int i = 0; i < total_calls; i++) {

        int idx = i * GROUP_THREADS + laneid;
        
        int row = idx / memcpy_per_row;
        int col = (idx*elem_per_memcpy) % dst.cols;

        if (row < load_rows) { // the first condition lets compiler short-circuit on unrolled iters
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n"
                :: "r"(dst.idx(dst_ptr, {row, col})), "l"((uint64_t)&src_ptr[row*row_stride + col])
                : "memory"
            );
        }
        else if (row < ST::rows) {
            move<float4>::sts(dst.idx(dst_ptr, {row, col}), packed_fill); // use the default value
        }
    }
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}