/**
 * @file
 * @brief Group (collaborative warp) ops for loading tensor tiles into register tiles.
 */

/**
 * @brief Load data from a tensor tile into a register tile.
 *
 * @tparam RT The register tile type
 * @tparam TM The tensor memory tile type
 * @param dst[out] The destination register tile.
 * @param src[in]  The source tensor tile.
 */
template<ducks::rt::row_layout RT, ducks::tt::all TM>
__device__ inline static void load_async(RT &dst, const TM &src) {
    if constexpr (GROUP_WARPS == 1) {
        static_assert(RT::rows == TM::rows, "register tile and tensor tile must match rows");
        static_assert(RT::cols == TM::cols, "register tile and tensor tile must match cols");

        using T2 = RT::dtype;
        using U  = typename TM::dtype;
        using U2 = base_types::packing<typename TM::dtype>::packed_type;

        if constexpr (sizeof(typename TM::dtype) == 1) {
            #pragma unroll
            for(int i = 0; i < RT::height; i++) {
                #pragma unroll
                for(int j = 0; j < RT::width; j++) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x128b.x2.b32 {%0, %1, %2, %3}, [%4];\n"   // pack::16b doesn't make sense for fp8
                        : "=r"(*(uint32_t*) &dst.tiles[i][j].data[0]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[1]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[2]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[3])
                        : "r"(src.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U)))
                    );
                }
            }
        } else if constexpr (sizeof(typename TM::dtype) == 2) {
            #pragma unroll
            for(int i = 0; i < RT::height; i++) {
                #pragma unroll
                for(int j = 0; j < RT::width; j++) {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x128b.x2.pack::16b.b32 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(*(uint32_t*) &dst.tiles[i][j].data[0]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[1]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[2]),
                            "=r"(*(uint32_t*) &dst.tiles[i][j].data[3])
                        : "r"(src.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col))
                    );
                }
            }
        }
        else if constexpr (sizeof(typename TM::dtype) == 4) {
            if constexpr (std::is_same_v<U, int>) {
                #pragma unroll
                for(int i = 0; i < RT::height; i++) {
                    if constexpr (RT::width%4 == 0) {
                        #pragma unroll
                        for(int j = 0; j < RT::width; j+=4) {
                            U2 data[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n"
                                : "=r"(data[0].x), "=r"(data[0].y),
                                "=r"(data[1].x), "=r"(data[1].y),
                                "=r"(data[2].x), "=r"(data[2].y),
                                "=r"(data[3].x), "=r"(data[3].y),
                                "=r"(data[4].x), "=r"(data[4].y),
                                "=r"(data[5].x), "=r"(data[5].y),
                                "=r"(data[6].x), "=r"(data[6].y),
                                "=r"(data[7].x), "=r"(data[7].y),
                                "=r"(data[8].x), "=r"(data[8].y),
                                "=r"(data[9].x), "=r"(data[9].y),
                                "=r"(data[10].x), "=r"(data[10].y),
                                "=r"(data[11].x), "=r"(data[11].y),
                                "=r"(data[12].x), "=r"(data[12].y),
                                "=r"(data[13].x), "=r"(data[13].y),
                                "=r"(data[14].x), "=r"(data[14].y),
                                "=r"(data[15].x), "=r"(data[15].y)
                                : "r"(src.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U)))
                            );
                            #pragma unroll
                            for(int k = 0; k < 4; k++) {
                                dst.tiles[i][j+0].data[k] = base_types::convertor<T2, U2>::convert(data[k]);
                                dst.tiles[i][j+1].data[k] = base_types::convertor<T2, U2>::convert(data[k+4]);
                                dst.tiles[i][j+2].data[k] = base_types::convertor<T2, U2>::convert(data[k+8]);
                                dst.tiles[i][j+3].data[k] = base_types::convertor<T2, U2>::convert(data[k+12]);
                            }
                        }
                    }
                    else if constexpr (RT::width%2 == 0) {
                        #pragma unroll
                        for(int j = 0; j < RT::width; j+=2) {
                            U2 data[8];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];\n"
                                : "=r"(data[0].x), "=r"(data[0].y),
                                "=r"(data[1].x), "=r"(data[1].y),
                                "=r"(data[2].x), "=r"(data[2].y),
                                "=r"(data[3].x), "=r"(data[3].y),
                                "=r"(data[4].x), "=r"(data[4].y),
                                "=r"(data[5].x), "=r"(data[5].y),
                                "=r"(data[6].x), "=r"(data[6].y),
                                "=r"(data[7].x), "=r"(data[7].y)
                                : "r"(src.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U)))
                            );
                            #pragma unroll
                            for(int k = 0; k < 4; k++) {
                                dst.tiles[i][j+0].data[k] = base_types::convertor<T2, U2>::convert(data[k]);
                                dst.tiles[i][j+1].data[k] = base_types::convertor<T2, U2>::convert(data[k+4]);
                            }
                        }
                    }
                    else {
                        #pragma unroll
                        for(int j = 0; j < RT::width; j++) {
                            U2 data[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                                : "=r"(data[0].x), "=r"(data[0].y),
                                "=r"(data[1].x), "=r"(data[1].y),
                                "=r"(data[2].x), "=r"(data[2].y),
                                "=r"(data[3].x), "=r"(data[3].y)
                                : "r"(src.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U)))
                            );
                            #pragma unroll
                            for(int k = 0; k < 4; k++) {
                                dst.tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(data[k]);
                            }
                        }
                    }
                }
            }
            else if constexpr (std::is_same_v<U, float>) {
                #pragma unroll
                for(int i = 0; i < RT::height; i++) {
                    if constexpr (RT::width%4 == 0) {
                        #pragma unroll
                        for(int j = 0; j < RT::width; j+=4) {
                            U2 data[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];\n"
                                : "=f"(data[0].x), "=f"(data[0].y),
                                "=f"(data[1].x), "=f"(data[1].y),
                                "=f"(data[2].x), "=f"(data[2].y),
                                "=f"(data[3].x), "=f"(data[3].y),
                                "=f"(data[4].x), "=f"(data[4].y),
                                "=f"(data[5].x), "=f"(data[5].y),
                                "=f"(data[6].x), "=f"(data[6].y),
                                "=f"(data[7].x), "=f"(data[7].y),
                                "=f"(data[8].x), "=f"(data[8].y),
                                "=f"(data[9].x), "=f"(data[9].y),
                                "=f"(data[10].x), "=f"(data[10].y),
                                "=f"(data[11].x), "=f"(data[11].y),
                                "=f"(data[12].x), "=f"(data[12].y),
                                "=f"(data[13].x), "=f"(data[13].y),
                                "=f"(data[14].x), "=f"(data[14].y),
                                "=f"(data[15].x), "=f"(data[15].y)
                                : "r"(src.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U)))
                            );
                            #pragma unroll
                            for(int k = 0; k < 4; k++) {
                                dst.tiles[i][j+0].data[k] = base_types::convertor<T2, U2>::convert(data[k]);
                                dst.tiles[i][j+1].data[k] = base_types::convertor<T2, U2>::convert(data[k+4]);
                                dst.tiles[i][j+2].data[k] = base_types::convertor<T2, U2>::convert(data[k+8]);
                                dst.tiles[i][j+3].data[k] = base_types::convertor<T2, U2>::convert(data[k+12]);
                            }
                        }
                    }
                    else if constexpr (RT::width%2 == 0) {
                        #pragma unroll
                        for(int j = 0; j < RT::width; j+=2) {
                            U2 data[8];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];\n"
                                : "=f"(data[0].x), "=f"(data[0].y),
                                "=f"(data[1].x), "=f"(data[1].y),
                                "=f"(data[2].x), "=f"(data[2].y),
                                "=f"(data[3].x), "=f"(data[3].y),
                                "=f"(data[4].x), "=f"(data[4].y),
                                "=f"(data[5].x), "=f"(data[5].y),
                                "=f"(data[6].x), "=f"(data[6].y),
                                "=f"(data[7].x), "=f"(data[7].y)
                                : "r"(src.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U)))
                            );
                            #pragma unroll
                            for(int k = 0; k < 4; k++) {
                                dst.tiles[i][j+0].data[k] = base_types::convertor<T2, U2>::convert(data[k]);
                                dst.tiles[i][j+1].data[k] = base_types::convertor<T2, U2>::convert(data[k+4]);
                            }
                        }
                    }
                    else {
                        #pragma unroll
                        for(int j = 0; j < RT::width; j++) {
                            U2 data[4];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x2.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
                                : "=f"(data[0].x), "=f"(data[0].y),
                                "=f"(data[1].x), "=f"(data[1].y),
                                "=f"(data[2].x), "=f"(data[2].y),
                                "=f"(data[3].x), "=f"(data[3].y)
                                : "r"(src.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U)))
                            );
                            #pragma unroll
                            for(int k = 0; k < 4; k++) {
                                dst.tiles[i][j].data[k] = base_types::convertor<T2, U2>::convert(data[k]);
                            }
                        }
                    }
                }
            }
            else {
                static_assert(sizeof(U) == 999, "Unsupported 4-byte tensor memory type.");
            }
        }
    }
    else {
        static_assert(GROUP_WARPS==4 || GROUP_WARPS==8);
        constexpr int warp_rows = TM::rows/GROUP_WARPS;
        static_assert(TM::cols==RT::cols);
        static_assert(warp_rows==RT::rows);
        if constexpr (GROUP_WARPS == 4) {
            auto src_subtile = src.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*warpid(), 0);
            ::kittens::group<1>::load_async(dst, src_subtile);
        }
        else {
            auto src_subtile = src.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*(warpid()%4)+16*(warpid()/4), 0);
            ::kittens::group<1>::load_async(dst, src_subtile);
        }
    }
}


/**
 * @brief Store data into a tensor tile from a register tile.
 *
 * @tparam RT The register tile type
 * @tparam TM The tensor memory tile type
 * @param dst[out] The destination tensor tile.
 * @param src[in]  The source register tile.
 */
template<ducks::rt::all RT, ducks::tt::all TM>
__device__ inline static void store_async(TM &dst, const RT &src) {
    if constexpr (GROUP_WARPS == 1) {
        static_assert(RT::rows == TM::rows, "register tile and tensor tile must match rows");
        static_assert(RT::cols == TM::cols, "register tile and tensor tile must match cols");

        using T2 = RT::dtype;
        using T = base_types::packing<T2>::unpacked_type;
        using U = TM::dtype;
        using U2 = base_types::packing<U>::packed_type;

        if constexpr (sizeof(typename TM::dtype) <= 2) {
            #pragma unroll
            for(int i = 0; i < RT::height; i++) {
                if constexpr (RT::width%4 == 0) {
                    #pragma unroll
                    for(int j = 0; j < RT::width; j+=4) {
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                            :: "r"(dst.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[3]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[3]),
                            "r"(*(uint32_t*)&src.tiles[i][j+2].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+2].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+2].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+2].data[3]),
                            "r"(*(uint32_t*)&src.tiles[i][j+3].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+3].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+3].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+3].data[3])
                        );
                    }
                }
                else if constexpr (RT::width%2 == 0) {
                    #pragma unroll
                    for(int j = 0; j < RT::width; j+=2) {
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x4.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
                            :: "r"(dst.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+0].data[3]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j+1].data[3])
                        );
                    }
                }
                else {
                    #pragma unroll
                    for(int j = 0; j < RT::width; j++) {
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x128b.x2.b32 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(dst.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U))),
                            "r"(*(uint32_t*)&src.tiles[i][j].data[0]),
                            "r"(*(uint32_t*)&src.tiles[i][j].data[1]),
                            "r"(*(uint32_t*)&src.tiles[i][j].data[2]),
                            "r"(*(uint32_t*)&src.tiles[i][j].data[3])
                        );
                    }
                }
            }
        }
        else if constexpr (sizeof(typename TM::dtype) == 4) {
            if constexpr (std::is_same_v<U, int>) {
                #pragma unroll
                for(int i = 0; i < RT::height; i++) {
                    if constexpr(RT::width%4 == 0) {
                        #pragma unroll
                        for(int j = 0; j < RT::width; j+=4) {
                            U2 data[16];
                            #pragma unroll
                            for(int k = 0; k < 4; k++) {
                                data[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                                data[k+4] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+1].data[k]);
                                data[k+8] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+2].data[k]);
                                data[k+12] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+3].data[k]);
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};\n"
                                :: "r"(dst.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U))),
                                "r"(data[0].x), "r"(data[0].y),
                                "r"(data[1].x), "r"(data[1].y),
                                "r"(data[2].x), "r"(data[2].y),
                                "r"(data[3].x), "r"(data[3].y),
                                "r"(data[4].x), "r"(data[4].y),
                                "r"(data[5].x), "r"(data[5].y),
                                "r"(data[6].x), "r"(data[6].y),
                                "r"(data[7].x), "r"(data[7].y),
                                "r"(data[8].x), "r"(data[8].y),
                                "r"(data[9].x), "r"(data[9].y),
                                "r"(data[10].x), "r"(data[10].y),
                                "r"(data[11].x), "r"(data[11].y),
                                "r"(data[12].x), "r"(data[12].y),
                                "r"(data[13].x), "r"(data[13].y),
                                "r"(data[14].x), "r"(data[14].y),
                                "r"(data[15].x), "r"(data[15].y)
                            );
                        }
                    }
                    else if constexpr(RT::width%2 == 0) {
                        #pragma unroll
                        for(int j = 0; j < RT::width; j+=2) {
                            U2 data[8];
                            #pragma unroll
                            for(int k = 0; k < 4; k++) {
                                data[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                                data[k+4] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+1].data[k]);
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x4.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                                :: "r"(dst.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U))),
                                "r"(data[0].x), "r"(data[0].y),
                                "r"(data[1].x), "r"(data[1].y),
                                "r"(data[2].x), "r"(data[2].y),
                                "r"(data[3].x), "r"(data[3].y),
                                "r"(data[4].x), "r"(data[4].y),
                                "r"(data[5].x), "r"(data[5].y),
                                "r"(data[6].x), "r"(data[6].y),
                                "r"(data[7].x), "r"(data[7].y)
                            );
                        }
                    }
                    else {
                        #pragma unroll
                        for(int j = 0; j < RT::width; j++) {
                            U2 data[4];
                            #pragma unroll
                            for(int k = 0; k < 4; k++) {
                                data[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x2.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
                                :: "r"(dst.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U))),
                                "r"(data[0].x), "r"(data[0].y),
                                "r"(data[1].x), "r"(data[1].y),
                                "r"(data[2].x), "r"(data[2].y),
                                "r"(data[3].x), "r"(data[3].y)
                            );
                        }
                    }
                }
            }
            else if constexpr (std::is_same_v<U, float>) {
                #pragma unroll
                for(int i = 0; i < RT::height; i++) {
                    if constexpr(RT::width%4 == 0) {
                        #pragma unroll
                        for(int j = 0; j < RT::width; j+=4) {
                            U2 data[16];
                            #pragma unroll
                            for(int k = 0; k < 4; k++) {
                                data[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                                data[k+4] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+1].data[k]);
                                data[k+8] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+2].data[k]);
                                data[k+12] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+3].data[k]);
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};\n"
                                :: "r"(dst.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U))),
                                "f"(data[0].x), "f"(data[0].y),
                                "f"(data[1].x), "f"(data[1].y),
                                "f"(data[2].x), "f"(data[2].y),
                                "f"(data[3].x), "f"(data[3].y),
                                "f"(data[4].x), "f"(data[4].y),
                                "f"(data[5].x), "f"(data[5].y),
                                "f"(data[6].x), "f"(data[6].y),
                                "f"(data[7].x), "f"(data[7].y),
                                "f"(data[8].x), "f"(data[8].y),
                                "f"(data[9].x), "f"(data[9].y),
                                "f"(data[10].x), "f"(data[10].y),
                                "f"(data[11].x), "f"(data[11].y),
                                "f"(data[12].x), "f"(data[12].y),
                                "f"(data[13].x), "f"(data[13].y),
                                "f"(data[14].x), "f"(data[14].y),
                                "f"(data[15].x), "f"(data[15].y)
                            );
                        }
                    }
                    else if constexpr(RT::width%2 == 0) {
                        #pragma unroll
                        for(int j = 0; j < RT::width; j+=2) {
                            U2 data[8];
                            #pragma unroll
                            for(int k = 0; k < 4; k++) {
                                data[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                                data[k+4] = base_types::convertor<U2, T2>::convert(src.tiles[i][j+1].data[k]);
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x4.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};\n"
                                :: "r"(dst.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U))),
                                "f"(data[0].x), "f"(data[0].y),
                                "f"(data[1].x), "f"(data[1].y),
                                "f"(data[2].x), "f"(data[2].y),
                                "f"(data[3].x), "f"(data[3].y),
                                "f"(data[4].x), "f"(data[4].y),
                                "f"(data[5].x), "f"(data[5].y),
                                "f"(data[6].x), "f"(data[6].y),
                                "f"(data[7].x), "f"(data[7].y)
                            );
                        }
                    }
                    else {
                        #pragma unroll
                        for(int j = 0; j < RT::width; j++) {
                            U2 data[4];
                            #pragma unroll
                            for(int k = 0; k < 4; k++) {
                                data[k] = base_types::convertor<U2, T2>::convert(src.tiles[i][j].data[k]);
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x2.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};\n"
                                :: "r"(dst.addr + ((i * RT::tile_size_row) << 16) + (j * RT::tile_size_col)/(4/(uint32_t)sizeof(U))),
                                "f"(data[0].x), "f"(data[0].y),
                                "f"(data[1].x), "f"(data[1].y),
                                "f"(data[2].x), "f"(data[2].y),
                                "f"(data[3].x), "f"(data[3].y)
                            );
                        }
                    }
                }
            }
            else {
                static_assert(sizeof(U) == 999, "Unsupported 4-byte tensor memory type.");
            }
        }
    }
    else {
        static_assert(GROUP_WARPS==4 || GROUP_WARPS==8);
        constexpr int warp_rows = TM::rows/GROUP_WARPS;
        static_assert(TM::cols==RT::cols);
        static_assert(warp_rows==RT::rows);
        if constexpr (GROUP_WARPS == 4) {
            auto dst_subtile = dst.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*warpid(), 0);
            ::kittens::group<1>::store_async(dst_subtile, src);
        }
        else {
            auto dst_subtile = dst.template subtile<tt<typename TM::dtype, warp_rows, TM::cols>>(32*(warpid()%4)+16*(warpid()/4), 0);
            ::kittens::group<1>::store_async(dst_subtile, src);
        }
    }
}

/**
 * @brief Load contiguous columns from one tensor-memory row per lane.
 *
 * @tparam RV The naive register vector type.
 * @param dst[out] Per-lane destination registers.
 * @param src[in] Source tensor tile.
 * @param col_offset[in] First tensor-memory column to load.
 */
template<ducks::rv::naive_layout RV, ducks::tt::all TM>
__device__ inline static void load_async(RV &dst, const TM &src, int col_offset=0) {
    static_assert(std::is_same_v<typename TM::dtype, float>, "Tensor-row loads require a float tensor tile");
    static_assert(std::is_same_v<typename RV::dtype, float>, "Tensor-row loads require a float register vector");
    static_assert(RV::length == 32*16 || RV::length == 32*32,
                  "Tensor-row loads support 16 or 32 columns per lane");
    static_assert(TM::rows == 32*GROUP_WARPS, "Tensor-row loads require one tensor-memory row per lane");
    constexpr int lane_cols = RV::length / 32;

    if constexpr (GROUP_WARPS == 1) {
        auto load_src = src.template subtile<tt<typename TM::dtype, 32, lane_cols>>(0, col_offset);
        if constexpr (lane_cols == 16) {
            asm volatile(
                "{tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];}"
                : "=f"(dst.data[0][0]),  "=f"(dst.data[1][0]),  "=f"(dst.data[2][0]),  "=f"(dst.data[3][0]),
                  "=f"(dst.data[4][0]),  "=f"(dst.data[5][0]),  "=f"(dst.data[6][0]),  "=f"(dst.data[7][0]),
                  "=f"(dst.data[8][0]),  "=f"(dst.data[9][0]),  "=f"(dst.data[10][0]), "=f"(dst.data[11][0]),
                  "=f"(dst.data[12][0]), "=f"(dst.data[13][0]), "=f"(dst.data[14][0]), "=f"(dst.data[15][0])
                : "r"(load_src.addr));
        }
        else {
            asm volatile(
                "{tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];}"
                : "=f"(dst.data[0][0]),  "=f"(dst.data[1][0]),  "=f"(dst.data[2][0]),  "=f"(dst.data[3][0]),
                  "=f"(dst.data[4][0]),  "=f"(dst.data[5][0]),  "=f"(dst.data[6][0]),  "=f"(dst.data[7][0]),
                  "=f"(dst.data[8][0]),  "=f"(dst.data[9][0]),  "=f"(dst.data[10][0]), "=f"(dst.data[11][0]),
                  "=f"(dst.data[12][0]), "=f"(dst.data[13][0]), "=f"(dst.data[14][0]), "=f"(dst.data[15][0]),
                  "=f"(dst.data[16][0]), "=f"(dst.data[17][0]), "=f"(dst.data[18][0]), "=f"(dst.data[19][0]),
                  "=f"(dst.data[20][0]), "=f"(dst.data[21][0]), "=f"(dst.data[22][0]), "=f"(dst.data[23][0]),
                  "=f"(dst.data[24][0]), "=f"(dst.data[25][0]), "=f"(dst.data[26][0]), "=f"(dst.data[27][0]),
                  "=f"(dst.data[28][0]), "=f"(dst.data[29][0]), "=f"(dst.data[30][0]), "=f"(dst.data[31][0])
                : "r"(load_src.addr));
        }
    }
    else {
        static_assert(GROUP_WARPS == 4, "Tensor-row loads support warp or warpgroup scope");
        auto src_subtile = src.template subtile<tt<typename TM::dtype, 32, TM::cols>>(32*warpid(), 0);
        ::kittens::group<1>::load_async(dst, src_subtile, col_offset);
    }
}

#if defined(KITTENS_SM103) || defined(KITTENS_SM107)
/**
 * @brief Load 16 contiguous columns and their maximum absolute value from one tensor-memory row per lane.
 *
 * @param dst[out] Per-lane destination registers.
 * @param max_abs[out] Maximum absolute value of the loaded registers.
 * @param src[in] Source tensor tile.
 * @param col_offset[in] First tensor-memory column to load.
 */
template<ducks::rv::naive_layout RV, ducks::tt::all TM>
__device__ inline static void load_async_max_abs(RV &dst, float &max_abs, const TM &src, int col_offset=0) {
    static_assert(std::is_same_v<typename TM::dtype, float>, "Tensor-row reduction loads require a float tensor tile");
    static_assert(std::is_same_v<typename RV::dtype, float>, "Tensor-row reduction loads require a float register vector");
    static_assert(RV::length == 32*16, "Tensor-row reduction loads require 16 columns per lane");
    static_assert(TM::rows == 32*GROUP_WARPS, "Tensor-row reduction loads require one tensor-memory row per lane");
    constexpr int lane_cols = RV::length / 32;

    if constexpr (GROUP_WARPS == 1) {
        auto load_src = src.template subtile<tt<typename TM::dtype, 32, lane_cols>>(0, col_offset);
        asm volatile(
            "{tcgen05.ld.red.sync.aligned.32x32b.x16.max.abs.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, [%17];}"
            : "=f"(dst.data[0][0]),  "=f"(dst.data[1][0]),  "=f"(dst.data[2][0]),  "=f"(dst.data[3][0]),
              "=f"(dst.data[4][0]),  "=f"(dst.data[5][0]),  "=f"(dst.data[6][0]),  "=f"(dst.data[7][0]),
              "=f"(dst.data[8][0]),  "=f"(dst.data[9][0]),  "=f"(dst.data[10][0]), "=f"(dst.data[11][0]),
              "=f"(dst.data[12][0]), "=f"(dst.data[13][0]), "=f"(dst.data[14][0]), "=f"(dst.data[15][0]),
              "=f"(max_abs)
            : "r"(load_src.addr));
    }
    else {
        static_assert(GROUP_WARPS == 4, "Tensor-row reduction loads support warp or warpgroup scope");
        auto src_subtile = src.template subtile<tt<typename TM::dtype, 32, TM::cols>>(32*warpid(), 0);
        ::kittens::group<1>::load_async_max_abs(dst, max_abs, src_subtile, col_offset);
    }
}
#endif
