#pragma once

#include <humming/utils/all.cuh>


__host__ __device__ constexpr uint32_t constexpr_log2(uint32_t value) {
  uint32_t result = 0;
  while (value > 1) {
    value >>= 1;
    result++;
  }
  return result;
}


template <
    uint32_t kValuesPerLane,
    uint32_t kHadamardBlockSize,
    uint32_t kNumWarps,
    uint32_t kWarpOffset,
    bool kTileBarriers = true>
CUDA_INLINE void hadamard(float *values, float *shared) {
  static_assert(kHadamardBlockSize % kValuesPerLane == 0);
  constexpr uint32_t kLanes = kHadamardBlockSize / kValuesPerLane;
  constexpr uint32_t kThreads = kNumWarps * 32;
  constexpr uint32_t kTiles = kThreads / kLanes;
  static_assert(kLanes >= 1 && kLanes <= 1024);
  static_assert((kLanes & (kLanes - 1)) == 0);
  static_assert(kValuesPerLane >= 1);
  static_assert((kValuesPerLane & (kValuesPerLane - 1)) == 0);
  static_assert(kLanes <= 32 || kLanes % 32 == 0);
  static_assert(kThreads % kLanes == 0);
  static_assert(kThreads <= 1024);

  uint32_t thread = threadIdx.x - kWarpOffset * 32;
  uint32_t lane = thread % kLanes;
  uint32_t tile = thread / kLanes;

  constexpr uint32_t kValueStages = constexpr_log2(kValuesPerLane);
  PRAGMA_UNROLL
  for (uint32_t stage = 0; stage < kValueStages; stage++) {
    uint32_t partner = 1u << stage;
    PRAGMA_UNROLL
    for (uint32_t value = 0; value < kValuesPerLane; value++) {
      uint32_t other = value ^ partner;
      if (value < other) {
        float lo = values[value];
        float hi = values[other];
        values[value] = lo + hi;
        values[other] = lo - hi;
      }
    }
  }

  constexpr uint32_t kLaneStages = constexpr_log2(kLanes);
  constexpr uint32_t kShuffleStages = kLaneStages < 5 ? kLaneStages : 5;
  constexpr uint32_t kShuffleWidth = kLanes < 32 ? kLanes : 32;
  constexpr uint32_t kActive = 0xFFFFFFFFu;
  PRAGMA_UNROLL
  for (uint32_t stage = 0; stage < kShuffleStages; stage++) {
    uint32_t partner = 1u << stage;
    bool low = (lane & partner) == 0;
    float sign = low ? 1.f : -1.f;
    PRAGMA_UNROLL
    for (uint32_t value = 0; value < kValuesPerLane; value++) {
      float other = __shfl_xor_sync(kActive, values[value], partner, kShuffleWidth);
      values[value] = fmaf(values[value], sign, other);
    }
  }

  if constexpr (kLanes > 32) {
    static_assert(kTiles <= 16);
    uint32_t barrier = tile + 1;
    uint32_t tile_base = tile * kHadamardBlockSize;
    uint32_t own = tile_base + lane * kValuesPerLane;
    PRAGMA_UNROLL
    for (uint32_t stage = 5; stage < kLaneStages; stage++) {
      uint32_t partner = 1u << stage;
      uint32_t other = tile_base + (lane ^ partner) * kValuesPerLane;
      bool low = (lane & partner) == 0;
      float sign = low ? 1.f : -1.f;
      PRAGMA_UNROLL
      for (uint32_t value = 0; value < kValuesPerLane; value++)
        shared[own + value] = values[value];
      if constexpr (kTileBarriers && kTiles > 1 && kTiles < 16) {
        asm volatile("bar.sync %0, %1;" : : "r"(barrier), "r"(kLanes) : "memory");
      } else {
        __syncthreads();
      }
      PRAGMA_UNROLL
      for (uint32_t value = 0; value < kValuesPerLane; value++) {
        float partner_value = shared[other + value];
        values[value] = fmaf(values[value], sign, partner_value);
      }
      if constexpr (kTileBarriers && kTiles > 1 && kTiles < 16) {
        asm volatile("bar.sync %0, %1;" : : "r"(barrier), "r"(kLanes) : "memory");
      } else {
        __syncthreads();
      }
    }
  }

  float normalizer = rsqrtf(static_cast<float>(kHadamardBlockSize));
  PRAGMA_UNROLL
  for (uint32_t value = 0; value < kValuesPerLane; value++)
    values[value] *= normalizer;
}
