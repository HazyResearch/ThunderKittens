#include <cooperative_groups.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvtx3/nvToolsExt.h>

#include "all_to_all/internode.h"
#include "core/device_utils.h"
#include "core/nvshmem_utils.h"
#include "core/utils.h"

using namespace pplx;

template <unsigned NUM_WARPS, bool DO_SEND, bool DO_RECV>
__global__ __launch_bounds__(NUM_WARPS * 32, 1) void dispatchKernel(
    int32_t *outNumTokensPerExpert,
    size_t outNumTokensPerExpertStrideElem,
    std::byte *expertX,
    size_t expertXStrideElem,
    size_t expertXStrideRow,
    std::byte *expertXScale,
    size_t expertXScaleStrideElem,
    size_t expertXScaleStrideRow,
    std::byte *dpX,
    size_t dpXStrideElem,
    std::byte *dpXScale,
    size_t dpXScaleStrideElem,
    uint32_t *indices,
    size_t indicesStrideElem,
    size_t indicesStrideRow,
    size_t maxNumTokens,
    size_t numExperts,
    unsigned rank,
    unsigned worldSize,
    unsigned dpSize,
    size_t hiddenDim,
    size_t hiddenDimScale,
    size_t numExpertsPerToken,
    unsigned *boundM,
    unsigned m,
    uint32_t *numTokensPerDP,
    uint32_t *sourceExpert,
    uint32_t *sourceIndex,
    uint32_t *sourceOffset,
    uint32_t *sourceGroup,
    uint64_t *numTokensBuffer,
    uint64_t *numRecvBuffer,
    std::byte *xBufferIn,
    std::byte *xBufferOut
) {
  // Determine the rank, DP rank and per-rank constants.
  const unsigned numLocalExperts = numExperts / worldSize;
  const unsigned numDPGroups = worldSize / dpSize;
  const unsigned dpGroup = rank / dpSize;
  const unsigned dpRank = rank % dpSize;
  const unsigned tokenDim = hiddenDim + hiddenDimScale;
  const unsigned tokenStride =
      device::round_up<unsigned>(tokenDim + sizeof(uint32_t), sizeof(int4));
  const unsigned WARP_SIZE = 32;
  const unsigned warpId = threadIdx.x / WARP_SIZE;
  const unsigned laneId = threadIdx.x % WARP_SIZE;

  // Determine the number of tokens populated which are to be sent.
  const unsigned numTokens = boundM ? __ldg(boundM) : m;
  ROSE_DEVICE_ASSERT(numTokens <= maxNumTokens);
  ROSE_DEVICE_ASSERT(
      hiddenDimScale == 0 || numTokens == 0 || (expertXScale != nullptr && dpXScale != nullptr)
  );

  // Zero out the shared memory buffer.
  if constexpr (DO_SEND) {
    extern __shared__ uint32_t tokenIndex[];
    for (uint32_t i = threadIdx.x; i < numExperts; i += blockDim.x) {
      tokenIndex[i] = 0;
    }
    __syncthreads();

    if (warpId + 1 == NUM_WARPS) {
      // The experts are split across the available blocks.
      // The warp counts the number of tokens assigned to each expert.
      for (unsigned dstExpert = blockIdx.x * dpSize + dpRank; dstExpert < numExperts;
           dstExpert += gridDim.x * dpSize) {
        const uint32_t dstRank = dstExpert / numLocalExperts;
        const uint32_t dstLocalExpert = dstExpert % numLocalExperts;

        unsigned count = 0;

#pragma unroll
        for (uint32_t i = laneId; i < numTokens * numExpertsPerToken; i += WARP_SIZE) {
          unsigned expert = __ldg(&indices[i]);
          if (expert == dstExpert) {
            count += 1;
          }
        }

        unsigned numTokensPerExpert = device::warp_sum(count);
        uint64_t *dstCount = &numTokensBuffer[dstLocalExpert * numDPGroups + dpGroup];

        if (laneId == 0) {
          nvshmemx_signal_op(dstCount, numTokensPerExpert + 1, NVSHMEM_SIGNAL_SET, dstRank);
        }
      }

      // Clear out some buffers.
      if (blockIdx.x == 0) {
        for (uint32_t i = laneId; i < numLocalExperts; i += WARP_SIZE) {
          outNumTokensPerExpert[i] = 0;
        }
      }
    } else {
      // Send the tokens to the destination ranks through RDMA.
      const unsigned numGroupWarps = NUM_WARPS - 1;
      const unsigned numGroupThreads = numGroupWarps * WARP_SIZE;
      for (unsigned i = 0; i < numTokens; i++) {
        // If the token is assigned to this block, handle it.
        if (i % (gridDim.x * dpSize) == (blockIdx.x * dpSize + dpRank)) {
          // Copy the token to the symmetric buffer.
          std::byte *xInPtr = xBufferIn + i * tokenStride;
          const int4 *srcX = (int4 *)(dpX + i * dpXStrideElem);
          const int4 *srcXScale = (int4 *)(dpXScale + i * dpXScaleStrideElem);
          for (unsigned d = threadIdx.x; d * sizeof(int4) < tokenDim; d += numGroupThreads) {
            if (d * sizeof(int4) < hiddenDim) {
              ((int4 *)xInPtr)[d] = srcX[d];
            } else {
              ((int4 *)xInPtr)[d] = srcXScale[d - hiddenDim / sizeof(int4)];
            }
          }
          if (threadIdx.x == 0) {
            *((uint32_t *)(xInPtr + tokenDim)) = i;
          }

          // Synchronize the warps within this warp group.
          asm volatile("bar.sync 1, %0;" ::"r"(numGroupThreads));

          // Send the token to the other ranks, one send per warp.
          for (unsigned j = warpId; j < numExpertsPerToken; j += numGroupWarps) {
            const uint32_t dstExpert = __ldg(&indices[i * numExpertsPerToken + j]);
            const uint32_t dstRank = dstExpert / numLocalExperts;
            const uint32_t dstLocalExpert = dstExpert % numLocalExperts;

            const uint32_t index = tokenIndex[dstExpert];
            const uint32_t group = dstLocalExpert * numDPGroups + dpGroup;
            const unsigned loc = group * maxNumTokens + index;

            std::byte *destPointer = xBufferOut + loc * tokenStride;
            nvshmemx_putmem_signal_nbi_warp(
                destPointer,
                xInPtr,
                tokenStride,
                &numRecvBuffer[group],
                1,
                NVSHMEM_SIGNAL_ADD,
                dstRank
            );
          }
        }

        // Replicate the token count calculation across all blocks.
        if (warpId == 0 && laneId < numExpertsPerToken) {
          uint32_t dstExpert = __ldg(&indices[i * numExpertsPerToken + laneId]);
          tokenIndex[dstExpert]++;
        }
      }
    }
  }

  if constexpr (DO_RECV) {
    // Wait for the token counts to be sent.
    for (unsigned expertAndGroup = blockIdx.x * blockDim.x + threadIdx.x;
         expertAndGroup < numLocalExperts * numDPGroups;
         expertAndGroup += blockDim.x * gridDim.x) {
      const uint32_t srcDpGroup = expertAndGroup % numDPGroups;
      const uint32_t srcLocalExpert = expertAndGroup / numDPGroups;
      const size_t slot = srcLocalExpert * numDPGroups + srcDpGroup;

      // Fetch the token count per DP, which is non-zero to indicate receipt.
      // Afterwards, wait for exactly that many tokens to be sent to us.
      nvshmem_uint64_wait_until(&numTokensBuffer[slot], NVSHMEM_CMP_NE, 0);
      size_t numTokens = numTokensBuffer[slot] - 1;
      nvshmem_uint64_wait_until(&numRecvBuffer[slot], NVSHMEM_CMP_EQ, numTokens);

      // Store the token count locally.
      numTokensPerDP[slot] = numTokens;
      atomicAdd(&outNumTokensPerExpert[srcLocalExpert], numTokens);

      // Clean the buffers.
      numTokensBuffer[slot] = 0;
      numRecvBuffer[slot] = 0;
    }
    cg::this_grid().sync();

    // Copy the tokens from the symmetric buffer to the output buffer.
    unsigned expert = 0;
    unsigned dp = 0;
    unsigned offset = 0;
    unsigned start = 0;
    const uint32_t maxBatchTokens = numLocalExperts * numDPGroups * maxNumTokens;
    for (uint32_t token = blockIdx.x; token < maxBatchTokens; token += gridDim.x) {
      // Find the expert, DP group and index for this token.
      unsigned j = token - offset;
      while (offset + __ldg(&numTokensPerDP[expert * numDPGroups + dp]) <= token) {
        offset += __ldg(&numTokensPerDP[expert * numDPGroups + dp]);
        j = token - offset;
        if (++dp == numDPGroups) {
          dp = 0;
          start = offset;
          if (++expert == numLocalExperts) {
            break;
          }
        }
      }
      if (expert >= numLocalExperts) {
        break;
      }

      // Copy the token to the output buffer.
      const uint32_t group = expert * numDPGroups + dp;
      const std::byte *xTokenBuffer =
          (const std::byte *)xBufferOut + (group * maxNumTokens + j) * tokenStride;

      const unsigned loc = token - start;
      const int4 *srcX = (int4 *)xTokenBuffer;
      int4 *dstX = (int4 *)(expertX + expert * expertXStrideRow + loc * expertXStrideElem);
      int4 *dstXScale =
          (int4 *)(expertXScale + expert * expertXScaleStrideRow + loc * expertXScaleStrideElem);
      for (unsigned k = threadIdx.x; k * sizeof(int4) < tokenDim; k += blockDim.x) {
        if (k * sizeof(int4) < hiddenDim) {
          dstX[k] = srcX[k];
        } else {
          dstXScale[k - hiddenDim / sizeof(int4)] = srcX[k];
        }
      }

      if (threadIdx.x == 0) {
        sourceIndex[token] = *((uint32_t *)(xTokenBuffer + tokenDim));
        sourceExpert[token] = expert + 1;
        sourceOffset[token] = loc;
        sourceGroup[token] = dp;
      }
    }
  }
}

void AllToAllInterNode::dispatch(
    const Strided1D<int32_t> &outNumTokensPerExpert,
    const Strided2D<std::byte> &expertX,
    const Strided2D<std::byte> &expertXScale,
    const Strided1D<std::byte> &dpX,
    const Strided1D<std::byte> &dpXScale,
    const Strided2D<uint32_t> &indices,
    unsigned m,
    const unsigned *boundM,
    SplitMode splitMode,
    cudaStream_t stream
) {
  constexpr unsigned NUM_WARPS = 10;
  const unsigned numBlocks = std::min(
      std::max(
          ceil_div<unsigned>(numExperts, NUM_WARPS), (unsigned)(maxNumTokens * expertsPerToken)
      ),
      132u
  );
  dim3 dimGrid(numBlocks, 1, 1);
  dim3 dimBlock(NUM_WARPS * 32, 1, 1);

  void *args[] = {
      const_cast<int32_t **>(&outNumTokensPerExpert.data),
      const_cast<size_t *>(&outNumTokensPerExpert.strideElem),
      const_cast<std::byte **>(&expertX.data),
      const_cast<size_t *>(&expertX.strideElem),
      const_cast<size_t *>(&expertX.strideRow),
      const_cast<std::byte **>(&expertXScale.data),
      const_cast<size_t *>(&expertXScale.strideElem),
      const_cast<size_t *>(&expertXScale.strideRow),
      const_cast<std::byte **>(&dpX.data),
      const_cast<size_t *>(&dpX.strideElem),
      const_cast<std::byte **>(&dpXScale.data),
      const_cast<size_t *>(&dpXScale.strideElem),
      const_cast<uint32_t **>(&indices.data),
      const_cast<size_t *>(&indices.strideElem),
      const_cast<size_t *>(&indices.strideRow),
      const_cast<size_t *>(&maxNumTokens),
      const_cast<size_t *>(&numExperts),
      const_cast<unsigned *>(&rank),
      const_cast<unsigned *>(&worldSize),
      const_cast<unsigned *>(&dpSize),
      const_cast<size_t *>(&hiddenDimBytes),
      const_cast<size_t *>(&hiddenDimScaleBytes),
      const_cast<size_t *>(&expertsPerToken),
      const_cast<unsigned **>(&boundM),
      &m,
      &numTokensPerDP,
      &sourceExpert,
      &sourceIndex,
      &sourceOffset,
      &sourceGroup,
      &numTokensBuffer,
      &numDispatchRecvBuffer,
      &xDispatchIn,
      &xDispatchOut,
  };

  nvtxRangePush("dispatch");
  switch (splitMode) {
  case SplitMode::SEND:
    CUDACHECK(cudaLaunchKernel(
        (void *)&dispatchKernel<NUM_WARPS, true, false>,
        dimGrid,
        dimBlock,
        args,
        sizeof(uint32_t) * numExperts,
        stream
    ));
    break;
  case SplitMode::RECV:
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&dispatchKernel<NUM_WARPS, false, true>, dimGrid, dimBlock, args, 0, stream
    ));
    break;
  case SplitMode::NONE:
    CUDACHECK(cudaLaunchCooperativeKernel(
        (void *)&dispatchKernel<NUM_WARPS, true, true>,
        dimGrid,
        dimBlock,
        args,
        sizeof(uint32_t) * numExperts,
        stream
    ));
    break;
  default:
    ROSE_UNREACHABLE("invalid split mode");
  }
  nvtxRangePop();
}