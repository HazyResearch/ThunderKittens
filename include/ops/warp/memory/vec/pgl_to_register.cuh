#pragma once

#include "../util/reduce.cuh"

namespace kittens {

template<ReduceOp OP, ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<RV>>
__device__ static inline void ld_reduce_op(RV &dst, const PGL &src, int dev_id, const COORD &idx) {
    using T2 = base_types::packing<typename RV::dtype>::packed_type;
    using U = base_types::packing<typename PGL::dtype>::unpacked_type;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;

    U *src_mc_ptr = src.mc_ptr_at(idx.template unit_coord<-1, 3>(), dev_id);
    int laneid = ::kittens::laneid();
    
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
            int o_dim = w*4 + (laneid/4) / 2;
            int i_dim = (laneid/4) % 2;
            if(idx < dst.outer_dim*16) {
                U2 dst_buf;
                multimem_ld_reduce_op<U2, OP>::apply(&dst_buf, (U2*)&src_mc_ptr[idx]);
                dst[o_dim][i_dim] = base_types::convertor<T2, U2>::convert(dst_buf);
            }
        }
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int leader = 8*(w%4) + (laneid%4); // repeats every 64 columns
            dst[w][0] = packed_shfl_sync(MASK_ALL, dst[w][0], leader);
            dst[w][1] = packed_shfl_sync(MASK_ALL, dst[w][1], leader+4);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < (dst.outer_dim+1)/2; w++) {
            int idx = w*16 + (laneid%4)*4 + (laneid/4)/2;
            int o_dim = w*2 + (laneid%4) / 2;
            
            if(idx < dst.outer_dim*8) {
                U2 dst_buf;
                multimem_ld_reduce_op<U2, OP>::apply(&dst_buf, (U2*)&src_mc_ptr[idx]);
                
                if(laneid%2 == 0) {
                    dst[o_dim][0].x = base_types::convertor<T, U>::convert(dst_buf.x);
                } else {
                    dst[o_dim][0].y = base_types::convertor<T, U>::convert(dst_buf.y);
                }
            }
        }
        
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            int leader = (laneid/4)*4 + 2*(w%2);
            dst[w][0].x = __shfl_sync(MASK_ALL, dst[w][0].x, leader);
            dst[w][0].y = __shfl_sync(MASK_ALL, dst[w][0].y, leader+1);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < dst.outer_dim; w++) {
            if(w < dst.outer_dim-1 || dst.length%32 == 0 || laneid<16) {
                U2 packed_value;
                if (laneid % 2 == 0) {
                    multimem_ld_reduce_op<U2, OP>::apply(&packed_value, (U2*)&src_mc_ptr[w*32 + laneid]);
                    dst[w][0] = base_types::convertor<T, U>::convert(packed_value.x);                    
                    __shfl_sync(MASK_ALL, packed_value.y, laneid + 1);
                } else {
                    dst[w][0] = base_types::convertor<T, U>::convert(__shfl_sync(MASK_ALL, U{}, laneid - 1));
                }
            }
        }
    }
   
}

template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<RV>>
__device__ static inline void all_reduce_add(RV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<ReduceOp::ADD>(dst, src, dev_id, idx);
}

template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<RV>>
__device__ static inline void all_reduce_min(RV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<ReduceOp::MIN>(dst, src, dev_id, idx);
}

template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<RV>>
__device__ static inline void all_reduce_max(RV &dst, const PGL &src, int dev_id, const COORD &idx) {
    ld_reduce_op<ReduceOp::MAX>(dst, src, dev_id, idx);
}

template<ReduceOp OP, ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void reduce_op(const PGL &dst, const RV &src, int dev_id, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename PGL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;
    
    U *dst_mc_ptr = dst.mc_ptr_at(idx.template unit_coord<-1, 3>(), dev_id);
    int laneid = ::kittens::laneid();
    
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
            int o_dim = w*4 + (laneid/4) / 2;
            int i_dim = (laneid/4) % 2;
            
            if(idx < src.outer_dim*16) {
                U2 tmp = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
                multimem_reduce_op<U2, OP>::apply((U2*)&dst_mc_ptr[idx], &tmp);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+1)/2; w++) {
            int idx = w*32 + (laneid%4)*8 + (laneid/4);
            int o_dim = w*2 + (laneid%4) / 2;
            
            if(idx < src.outer_dim*16) {
                U2 dst_buf;
                dst_buf.x = base_types::convertor<U, T>::convert(src[o_dim][0].x);
                dst_buf.y = base_types::convertor<U, T>::convert(src[o_dim][0].y);
                
                multimem_reduce_op<U2, OP>::apply((U2*)&dst_mc_ptr[idx], &dst_buf);
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            if(w < src.outer_dim-1 || src.length%32 == 0 || laneid<16) {
                T value = base_types::convertor<U, T>::convert(src[w][0]);
                
                if (laneid % 2 == 0) {
                    U neighbor_value = __shfl_sync(MASK_ALL, value, laneid + 1);
                    U2 tmp{value, neighbor_value};
                    multimem_reduce_op<U2, OP>::apply((U2*)&dst_mc_ptr[w*32 + laneid], &tmp);
                } else {
                    __shfl_sync(MASK_ALL, value, laneid - 1);
                }
            }
        }
    }
}

template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<RV>>
__device__ static inline void atomic_add(const PGL &dst, const RV &src, int dev_id, const COORD &idx) {
    reduce_op<ReduceOp::ADD>(dst, src, dev_id, idx);
}

template<ducks::rv::all RV, ducks::pgl::all PGL, ducks::coord::vec COORD=coord<RV>>
__device__ inline static void broadcast(const PGL &dst, const RV &src, int dev_id, const COORD &idx) {
    using T2 = RV::dtype;
    using U = typename PGL::dtype;
    using U2 = base_types::packing<U>::packed_type;
    using T = base_types::packing<T2>::unpacked_type;
    
    U *dst_mc_ptr = dst.mc_ptr_at(idx.template unit_coord<-1, 3>(), dev_id);
    int laneid = ::kittens::laneid();
    
    if constexpr (std::is_same_v<typename RV::layout, align_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+3)/4; w++) {
            int idx = w*64 + (laneid/4)*8 + 2*(laneid%4);
            int o_dim = w*4 + (laneid/4) / 2;
            int i_dim = (laneid/4) % 2;
            if(idx < src.outer_dim*16)
                *(U2*)&dst_mc_ptr[idx] = base_types::convertor<U2, T2>::convert(src[o_dim][i_dim]);
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, ortho_l>) {
        #pragma unroll
        for(auto w = 0; w < (src.outer_dim+1)/2; w++) {
            int idx = w*32 + (laneid%4)*8 + (laneid/4);
            int o_dim = w*2 + (laneid%4) / 2;
            if(idx < src.outer_dim*16) {
                U tmp;
                if(laneid%2==0) tmp = base_types::convertor<U, T>::convert(src[o_dim][0].x);
                else tmp = base_types::convertor<U, T>::convert(src[o_dim][0].y);
                dst_mc_ptr[idx] = tmp;
            }
        }
    }
    else if constexpr (std::is_same_v<typename RV::layout, naive_l>) {
        #pragma unroll
        for(auto w = 0; w < src.outer_dim; w++) {
            if(w < src.outer_dim-1 || src.length%32 == 0 || laneid<16) {
                dst_mc_ptr[w*32 + laneid] = base_types::convertor<U, T>::convert(src[w][0]);
            }
        }
    }
}



} // namespace kittens