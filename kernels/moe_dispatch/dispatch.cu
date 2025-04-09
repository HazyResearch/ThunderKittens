#include "kittens.cuh"
using namespace kittens;

constexpr int WORLD_SIZE = 8;
constexpr int DP_SIZE = 1;

struct globals {
    using g_pgl = pgl<gl<uint32_t, -1, -1, -1, -1>>;
    using g_gl = gl<uint32_t, -1, -1, -1, -1>;
    using g_buffer = pgl<gl<uint32_t, -1, -1, -1, -1>>;

    g_pgl outNumTokensPerExpert;

    g_pgl expertX;
    g_pgl expertXScale;

    g_pgl dpX;
    g_pgl dpXScale;

    g_pgl indices;

    g_buffer numTokensBuffer;
    g_buffer numRecvBuffer;
    g_buffer xBufferIn;
    g_buffer xBufferOut;

    size_t maxNumTokens;
    size_t numExperts;
    size_t numExpertsPerToken;
    int dev_idx;
};

__global__ void dispatch_kernel(const __grid_constant__ globals g) {

}