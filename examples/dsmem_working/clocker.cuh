#pragma once

#include <cuda.h>
#include <string>
#include <fstream>

#ifndef DISABLE_CUDA_CLOCKING

template<int a=64> __device__ constexpr int __timingdetail__N_constructor__() { return (a+31)/32; }
#define INITTHREADTIMER(...) \
    const int __timingdetail__N__ = __timingdetail__N_constructor__<__VA_ARGS__>(); \
    const int __timingdetail__laneid__ = ( \
        threadIdx.z*blockDim.y*blockDim.x + \
        threadIdx.y*blockDim.x + \
        threadIdx.x \
    ) % 32; \
    int __timingdetail__data__[__timingdetail__N__][2]; \
    for(int __timingdetail__i__ = 0; __timingdetail__i__ < __timingdetail__N__; __timingdetail__i__++) { \
        __timingdetail__data__[__timingdetail__i__][0] = 0; \
        __timingdetail__data__[__timingdetail__i__][1] = 0; \
    } \
    int __timingdetail__lastclock__ = clock(), __timingdetail__newclock__;

#define CLOCKRESET() __timingdetail__lastclock__ = clock();

#define CLOCKPOINT(ID, STRING...) \
    __timingdetail__newclock__ = clock(); \
    const int __timingdetail__##ID##__  = (ID); \
    if (__timingdetail__##ID##__ % 32 == __timingdetail__laneid__) { \
        __timingdetail__data__[__timingdetail__##ID##__ / 32][0] += 1; \
        __timingdetail__data__[__timingdetail__##ID##__ / 32][1] += ( \
            __timingdetail__newclock__ - __timingdetail__lastclock__ \
        ); \
    } \
    CLOCKRESET();

#define FINISHTHREADTIMER() \
    __nanosleep(1000); \
    const int __timingdetail__warpid__ = ( \
        (blockIdx.z*gridDim.y*gridDim.x*blockDim.z*blockDim.y*blockDim.x + \
        blockIdx.y*gridDim.x*blockDim.z*blockDim.y*blockDim.x + \
        blockIdx.x*blockDim.z*blockDim.y*blockDim.x + \
        threadIdx.z*blockDim.y*blockDim.x + \
        threadIdx.y*blockDim.x + \
        threadIdx.x) / 32 \
    ); \
    for(int __timingdetail__i__ = 0; __timingdetail__i__ < __timingdetail__N__; __timingdetail__i__++) { \
        __timingdetail__globaldata__[2*__timingdetail__N__*32*__timingdetail__warpid__+2*__timingdetail__i__*32+2*__timingdetail__laneid__+0] = \
            __timingdetail__data__[__timingdetail__i__][0]; \
        __timingdetail__globaldata__[2*__timingdetail__N__*32*__timingdetail__warpid__+2*__timingdetail__i__*32+2*__timingdetail__laneid__+1] = \
            __timingdetail__data__[__timingdetail__i__][1]; \
    } \
    __syncthreads();

#define TIMINGDETAIL_ARGS int * __timingdetail__globaldata__

struct TimingData {
    const int totalThreads;
    const int numBreakpoints; // divided by 32, this is per-thread
    int* data;
    TimingData(dim3 gridDim, dim3 blockDim, int numBP=64) : totalThreads(gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z), numBreakpoints((numBP+31)/32) {
        cudaMalloc((void **) &data, 2 * numBreakpoints * totalThreads * sizeof(int));
    }
    ~TimingData() {
        cudaFree(data);
    }
    void write(std::string src_file, std::string output_file="profile.csv") {
        int * host_data = new int[2 * numBreakpoints * totalThreads];
        cudaMemcpy(host_data, data, 2 * numBreakpoints * totalThreads * sizeof(int), cudaMemcpyDeviceToHost);
        std::ofstream file;
        file.open(output_file);
        file << src_file << "\n";
        for (int i = 0; i < totalThreads/32; i++) {
            for (int j = 0; j < numBreakpoints*32; j++) {
                file << host_data[2 * numBreakpoints*32 * i + 2 * j + 0] << "," << host_data[2 * numBreakpoints*32 * i + 2 * j + 1] << ",";
            }
            file << "\n";
        }
        file.close();
        delete[] host_data;
    }
};

#else

#define INITTHREADTIMER(...)

#define CLOCKRESET()

#define CLOCKPOINT(ID, STRING...)

#define FINISHTHREADTIMER()

#define TIMINGDETAIL_ARGS int * __timingdetail__globaldata__ = NULL

struct TimingData {
    const int totalThreads;
    const int numBreakpoints;
    int* data;
    TimingData(dim3 gridDim, dim3 blockDim, int numBP=64) : totalThreads(gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z), numBreakpoints((numBP+31)/32*32) {data=NULL;}
    ~TimingData() {}
    void write(std::string src_file, std::string output_file="profile.csv") {}
};

#endif