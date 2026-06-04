/**
 * @file Group-level tcgen05 MMA operations.
*/

template<int trans_a, int n_trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b, semaphore &sem) {
    if(laneid() == 0) ::kittens::mma<trans_a, n_trans_b, D, A, B, acc, ncta>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, int acc=1>
__device__ static inline void mma2(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, trans_b, D, A, B, acc, 2>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm(D &d, const A &a, const B &b, semaphore &sem) {
    mma<trans_a, trans_b, D, A, B, 0>(d, a, b, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<trans_a, trans_b, D, A, B, 0>(d, a, b, sem);
}
template<int trans_a, int n_trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int acc=1, int ncta=1, int mma_k=64>
__device__ static inline void mma(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    if(laneid() == 0) ::kittens::mma<trans_a, n_trans_b, D, A, B, SA, SB, acc, ncta, mma_k>(d, a, b, sa, sb, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int acc=1, int mma_k=64>
__device__ static inline void mma2(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mma<trans_a, trans_b, D, A, B, SA, SB, acc, 2, mma_k>(d, a, b, sa, sb, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int mma_k=64>
__device__ static inline void mm(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mma<trans_a, trans_b, D, A, B, SA, SB, 0, 1, mma_k>(d, a, b, sa, sb, sem);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int mma_k=64>
__device__ static inline void mm2(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mma2<trans_a, trans_b, D, A, B, SA, SB, 0, mma_k>(d, a, b, sa, sb, sem);
}

template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 1, 1, 96>(d, a, b, sa, sb, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma2_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, SA, SB, 1, 96>(d, a, b, sa, sb, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::N, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::T, D, A, B, 1>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::T, D, A, B, 1>(d, a, b, sem);
}

template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::N, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_ABt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::N, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mm<transpose::N, transpose::T, D, A, B, SA, SB, 96>(d, a, b, sa, sb, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm2_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb, semaphore &sem) {
    mm2<transpose::N, transpose::T, D, A, B, SA, SB, 96>(d, a, b, sa, sb, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtB(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::N, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma<transpose::T, transpose::T, D, A, B, 0>(d, a, b, sem);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtBt(D &d, const A &a, const B &b, semaphore &sem) {
    mma2<transpose::T, transpose::T, D, A, B, 0>(d, a, b, sem);
}

// no sem versions


template<int trans_a, int n_trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, int acc=1, int ncta=1>
__device__ static inline void mma(D &d, const A &a, const B &b) {
    if(laneid() == 0) ::kittens::mma<trans_a, n_trans_b, D, A, B, acc, ncta>(d, a, b);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, int acc=1>
__device__ static inline void mma2(D &d, const A &a, const B &b) {
    mma<trans_a, trans_b, D, A, B, acc, 2>(d, a, b);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm(D &d, const A &a, const B &b) {
    mma<trans_a, trans_b, D, A, B, 0>(d, a, b);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2(D &d, const A &a, const B &b) {
    mma2<trans_a, trans_b, D, A, B, 0>(d, a, b);
}
template<int trans_a, int n_trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int acc=1, int ncta=1, int mma_k=64>
__device__ static inline void mma(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    if(laneid() == 0) ::kittens::mma<trans_a, n_trans_b, D, A, B, SA, SB, acc, ncta, mma_k>(d, a, b, sa, sb);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int acc=1, int mma_k=64>
__device__ static inline void mma2(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma<trans_a, trans_b, D, A, B, SA, SB, acc, 2, mma_k>(d, a, b, sa, sb);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int mma_k=64>
__device__ static inline void mm(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma<trans_a, trans_b, D, A, B, SA, SB, 0, 1, mma_k>(d, a, b, sa, sb);
}
template<int trans_a, int trans_b, ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB, int mma_k=64>
__device__ static inline void mm2(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma2<trans_a, trans_b, D, A, B, SA, SB, 0, mma_k>(d, a, b, sa, sb);
}

template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AB(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AB(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_ABt(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_ABt(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma<transpose::N, transpose::T, D, A, B, SA, SB, 1, 1, 96>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mma2_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mma2<transpose::N, transpose::T, D, A, B, SA, SB, 1, 96>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtB(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtB(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::N, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma_AtBt(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::T, D, A, B, 1>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mma2_AtBt(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::T, D, A, B, 1>(d, a, b);
}

template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AB(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AB(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_ABt(D &d, const A &a, const B &b) {
    mma<transpose::N, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_ABt(D &d, const A &a, const B &b) {
    mma2<transpose::N, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mm<transpose::N, transpose::T, D, A, B, SA, SB, 96>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B, ducks::tt::all SA, ducks::tt::all SB>
__device__ static inline void mm2_ABt_k96(D &d, const A &a, const B &b, const SA &sa, const SB &sb) {
    mm2<transpose::N, transpose::T, D, A, B, SA, SB, 96>(d, a, b, sa, sb);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtB(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtB(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::N, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm_AtBt(D &d, const A &a, const B &b) {
    mma<transpose::T, transpose::T, D, A, B, 0>(d, a, b);
}
template<ducks::tt::all D, typename A, ducks::st_descriptor::input B>
__device__ static inline void mm2_AtBt(D &d, const A &a, const B &b) {
    mma2<transpose::T, transpose::T, D, A, B, 0>(d, a, b);
}
