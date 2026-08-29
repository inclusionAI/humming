#pragma once

#include <humming/kernel/process_input/activation.cuh>
#include <humming/kernel/process_input/hadamard.cuh>
#include <humming/kernel/process_input/layout.cuh>
#include <humming/kernel/process_input/quantization.cuh>


template <class Config>
struct ProcessInputConfig {
  using SourceType = typename Config::SourceType;
  using TargetType = typename Config::TargetType;
  using ActivationImpl = typename Config::Activation;
  static constexpr QuantizationMode kQuantization = Config::kQuantMode;
  static constexpr bool kUseTilePartition = Config::kUseTilePartition;
  static constexpr bool kDynamicTokenMode = kQuantization == QuantizationMode::DynamicToken;
  static constexpr bool kDynamicGroupMode = kQuantization == QuantizationMode::DynamicGroup || kQuantization == QuantizationMode::StaticTensorDynamicGroup;
  static constexpr bool kDynamicGroupTokenMode = kQuantization == QuantizationMode::DynamicGroupToken;
  static constexpr bool kStaticTensorScale = kQuantization == QuantizationMode::StaticTensor || kQuantization == QuantizationMode::StaticTensorDynamicGroup;
  static constexpr bool kDynamicGroupScale = kDynamicGroupMode || kDynamicGroupTokenMode;
  static constexpr bool kStagedGroupToken = kDynamicGroupTokenMode && kUseTilePartition;
  static constexpr ScaleMode kConfiguredScaleMode = kDynamicTokenMode ? ScaleMode::DynamicToken : kDynamicGroupTokenMode ? ScaleMode::DynamicGroupToken : kDynamicGroupMode ? ScaleMode::DynamicGroup : ScaleMode::Static;
  static constexpr ScaleMode kScaleMode = kStagedGroupToken ? ScaleMode::DynamicGroup : kConfiguredScaleMode;
  static constexpr bool kDynamicOutputScale = kScaleMode == ScaleMode::DynamicGroup || kScaleMode == ScaleMode::DynamicGroupToken;
  static constexpr bool kFusedGroupToken = kScaleMode == ScaleMode::DynamicGroupToken;
  using ConfiguredDynamicGroupScaleType = std::conditional_t<kDynamicGroupScale, typename Config::GroupScaleType, Float32>;
  using DynamicGroupScaleType = std::conditional_t<kStagedGroupToken, M3BFloat16, ConfiguredDynamicGroupScaleType>;
  using OutputScaleType = std::conditional_t<kDynamicOutputScale, DynamicGroupScaleType, Float32>;
  using QuantScaleType = std::conditional_t<kFusedGroupToken, M3BFloat16, OutputScaleType>;

  static constexpr uint32_t kHiddenSize = Config::kHiddenSize;
  static constexpr uint32_t kQuantGroupSize = Config::kQuantGroupSize;
  static constexpr uint32_t kHadamardBlockSize = Config::kHadamardBlockSize;
  static constexpr uint32_t kTileSize = Config::kTileSize;
  static constexpr bool kHadamard = kHadamardBlockSize > 1;
  static constexpr uint32_t kThreadsPerTask = Config::kThreadsPerTask;
  static constexpr uint32_t kValuesPerThread = Config::kValuesPerThread;
  static constexpr uint32_t kTokensPerBlock = Config::kTokensPerBlock;
  static constexpr uint32_t kThreads = kThreadsPerTask * kTokensPerBlock;
  static constexpr LayoutType kLayout = Config::kLayout;
  static constexpr bool kScatterSingleOutput = Config::kScatterSingleOutput;
  static constexpr ActivationType kActivation = ActivationImpl::kType;
  static constexpr bool kBinaryActivation = kActivation == ActivationType::BinarySplit || kActivation == ActivationType::BinaryInterleaved;
  static constexpr uint32_t kInputRowSize = kHiddenSize * (kBinaryActivation ? 2 : 1);
  static constexpr bool kPlainScatter = kLayout == LayoutType::Scatter && kActivation == ActivationType::None;
  static constexpr bool kDirectScatter = kScatterSingleOutput || (kPlainScatter && !kHadamard);
  static constexpr bool kStaticScale = kStaticTensorScale;
  static constexpr ScaleMode kQuantScaleMode = kScaleMode == ScaleMode::DynamicGroupToken ? ScaleMode::DynamicGroup : kScaleMode;
  static constexpr QuantizationPhase kPhase = Config::kQuantizationPhase;
  static constexpr GroupScaleLayout kGroupScaleLayout = Config::kScaleLayout;
  static constexpr bool kUsePdl = Config::kUsePdl;
  static constexpr bool kQuantize = kQuantization != QuantizationMode::Disabled;
  static constexpr uint32_t kOutputPacking = kQuantize ? 8 / TargetType::kBits : 1;
  static constexpr bool kAllowByteOutput = std::is_same<TargetType, Float8E3M4>::value;
  static constexpr uint32_t kTilesPerBlock = Config::kTilesPerBlock;

  static_assert(supported_scale_type<DynamicGroupScaleType>);
  static_assert(kScaleMode == ScaleMode::DynamicGroup || kScaleMode == ScaleMode::DynamicGroupToken || std::is_same<DynamicGroupScaleType, Float32>::value);
  static_assert(kPhase == QuantizationPhase::Fused || kScaleMode == ScaleMode::DynamicToken);
  static_assert(!std::is_same<DynamicGroupScaleType, M3BFloat16>::value || kPhase == QuantizationPhase::Fused);
  static_assert(
      kGroupScaleLayout == GroupScaleLayout::RowMajor ||
      kScaleMode == ScaleMode::DynamicGroup ||
      kScaleMode == ScaleMode::DynamicGroupToken);
  static_assert(kQuantize || kScaleMode == ScaleMode::Static);
  static_assert(kQuantize || !kStaticTensorScale);
  static_assert(kQuantize || kPhase == QuantizationPhase::Fused);
  static_assert(kQuantize || kGroupScaleLayout == GroupScaleLayout::RowMajor);
  static_assert(!kUseTilePartition || (kScaleMode != ScaleMode::DynamicToken && kScaleMode != ScaleMode::DynamicGroupToken));
  static_assert(kScaleMode != ScaleMode::DynamicGroupToken || kPhase == QuantizationPhase::Fused);
  static_assert(!kUseTilePartition || kTokensPerBlock == 1);
  static_assert(kTilesPerBlock >= 1);
  static_assert(!kScatterSingleOutput || kLayout == LayoutType::Scatter);
  static_assert(
      kGroupScaleLayout != GroupScaleLayout::MxPacked ||
      std::is_same<DynamicGroupScaleType, Float8E4M3>::value ||
      std::is_same<DynamicGroupScaleType, Float8E8M0>::value ||
      std::is_same<DynamicGroupScaleType, M3BFloat16>::value);

  using Activation = InputActivation<ActivationImpl, kHiddenSize, kValuesPerThread>;

  using Layout = InputLayout<
      kLayout,
      kHiddenSize,
      kThreadsPerTask,
      kValuesPerThread,
      kTokensPerBlock,
      Config::kLayoutWidth,
      kScatterSingleOutput,
      kDirectScatter,
      Config::kExpertLayoutInt64,
      Config::kIndexInt64,
      Config::kZeroInvalid,
      kUseTilePartition,
      kTileSize,
      kTilesPerBlock>;
};


template <class Config, class ScaleType>
CUDA_INLINE uint64_t group_scale_index(uint64_t output_row, uint32_t group, uint64_t scale_stride) {
  constexpr uint64_t kGroupsPerToken = Config::kHiddenSize / Config::kQuantGroupSize;
  if constexpr (std::is_same<ScaleType, M3BFloat16>::value || Config::kGroupScaleLayout == GroupScaleLayout::RowMajor) {
    return output_row * kGroupsPerToken + group;
  } else if constexpr (Config::kGroupScaleLayout == GroupScaleLayout::MMajor) {
    return group * scale_stride + output_row;
  } else {
    static_assert(Config::kGroupScaleLayout == GroupScaleLayout::MxPacked);
    return (group / 4) * (scale_stride * 4) + output_row * 4 + group % 4;
  }
}


template <class Config>
__global__ __launch_bounds__(Config::kThreads) void process_input_kernel(
    const void *input,
    void *output,
    const float *__restrict__ static_tensor_scales,
    void *__restrict__ output_scales,
    float *__restrict__ token_scales,
    const void *__restrict__ expert_layout,
    const void *__restrict__ indices,
    uint64_t num_input_rows,
    uint64_t num_output_rows,
    uint32_t num_experts,
    uint32_t max_tokens_per_expert,
    uint64_t group_scale_stride) {
  using SourceType = typename Config::SourceType;
  using TargetType = typename Config::TargetType;
  using Activation = typename Config::Activation;
  using Layout = typename Config::Layout;
  using OutputScaleType = typename Config::OutputScaleType;
  using QuantScaleType = typename Config::QuantScaleType;

  constexpr uint32_t K = Config::kHiddenSize;
  constexpr uint32_t G = Config::kQuantGroupSize;
  constexpr uint32_t H = Config::kHadamardBlockSize;
  constexpr uint32_t T = Config::kTileSize;
  constexpr uint32_t V = Config::kValuesPerThread;
  constexpr uint32_t kThreads = Config::kThreads;
  constexpr uint32_t kNumWarps = kThreads / 32;
  constexpr uint32_t kGroupsPerToken = K / G;
  constexpr uint32_t kScaleSize = Config::kScaleMode == ScaleMode::DynamicToken ? Config::kThreadsPerTask * V : G;
  constexpr uint32_t kPackedBytes = V * TargetType::kBits / 8;
  constexpr uint32_t kTransformScratch = Config::kHadamard && H / V > 32 ? kThreads * V : 0;
  constexpr bool kReduce = Config::kQuantize && Config::kScaleMode != ScaleMode::Static && Config::kPhase != QuantizationPhase::Quantize;
  constexpr uint32_t kGroupReduceScratch = kReduce && kScaleSize / V > 32 ? kNumWarps : 0;
  constexpr uint32_t kTokenReduceScratch = Config::kScaleMode == ScaleMode::DynamicGroupToken && Config::kThreadsPerTask > 32 ? kNumWarps : 0;
  constexpr uint32_t kReduceScratch = kGroupReduceScratch > kTokenReduceScratch ? kGroupReduceScratch : kTokenReduceScratch;
  constexpr uint32_t kScratchElements = kTransformScratch + kReduceScratch > 0 ? kTransformScratch + kReduceScratch : 1;
  constexpr bool kRawCopy = !Config::kQuantize && Config::kActivation == ActivationType::None && !Config::kHadamard;
  constexpr bool kPureHadamard =
      !Config::kQuantize &&
      Config::kActivation == ActivationType::None &&
      Config::kHadamard &&
      Config::kLayout == LayoutType::Normal &&
      Config::kUseTilePartition &&
      Layout::kFullColumns;
  constexpr bool kFp32Hadamard = kPureHadamard && std::is_same<SourceType, float>::value;

  static_assert(kThreads >= 32 && kThreads <= 1024 && kThreads % 32 == 0);
  static_assert(Config::kUseTilePartition || Config::kThreadsPerTask * V >= K);
  static_assert(K % G == 0 && G % V == 0);
  static_assert(K % T == 0 && T % V == 0);
  static_assert(V >= 1 && (V & (V - 1)) == 0);
  static_assert(!Config::kQuantize || V * TargetType::kBits % 8 == 0);
  if constexpr (Config::kHadamard) {
    static_assert(H >= 2 && (H & (H - 1)) == 0);
    static_assert(K % H == 0 && H % V == 0);
    static_assert(!Config::kUseTilePartition || Layout::kColumnsPerTask % H == 0);
  }

  if constexpr (Config::kUsePdl) griddepcontrol_wait();

  __shared__ typename Layout::SharedStorage layout_storage;
  __shared__ float scratch[kScratchElements];

  InputLayoutParams layout_params{expert_layout, indices, num_input_rows, num_output_rows, num_experts, max_tokens_per_expert};
  Layout layout(layout_params, &layout_storage);
  layout.prepare();
  auto thread = layout.thread();
  auto routes = layout.routes(thread);

  if constexpr (kRawCopy) {
    alignas(16) SourceType values[V];
    if (thread.load) {
      const SourceType *input2 = reinterpret_cast<const SourceType *>(input);
      load_raw_values<SourceType, V>(values, thread.input(input2));
    }

    PRAGMA_UNROLL
    for (uint32_t route = 0; route < Layout::kOutputsPerToken; route++) {
      auto write = layout.write(thread, routes, route);
      if (write.write) {
        SourceType *destination = reinterpret_cast<SourceType *>(write.template output<sizeof(SourceType) * 8>(output));
        store_raw_values<SourceType, V>(destination, values, write.zero);
      }
    }
  } else {
    float values[V];
    bool hadamard_valid = true;
    if constexpr (kFp32Hadamard) {
      constexpr uint32_t kLanesPerTransform = H / V;
      constexpr uint32_t kTransformsPerBlock = kThreads / kLanesPerTransform;
      uint64_t transform = static_cast<uint64_t>(blockIdx.x) * kTransformsPerBlock + threadIdx.x / kLanesPerTransform;
      hadamard_valid = transform < num_output_rows * (K / H);
    }
    if (thread.load && hadamard_valid) {
      const SourceType *input2 = reinterpret_cast<const SourceType *>(input);
      if constexpr (Config::kActivation == ActivationType::None) {
        load_values<SourceType, V>(values, thread.input(input2));
      } else {
        Activation::template load<SourceType>(values, input2, thread.input_row, thread.column);
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t value = 0; value < V; value++)
        values[value] = 0.f;
    }

    if constexpr (Config::kHadamard) {
      constexpr bool kTileBarriers = !kPureHadamard && H / V >= 128;
      hadamard<V, H, kNumWarps, 0, kTileBarriers>(values, scratch);
      if constexpr (kTransformScratch > 0 && kReduceScratch > 0) __syncthreads();
    }

    if constexpr (Config::kQuantize) {
      ScaleStorage<QuantScaleType> collected_scale{};
      if constexpr (Config::kPhase == QuantizationPhase::Quantize) {
        PRAGMA_UNROLL
        for (uint32_t route = 0; route < Layout::kOutputsPerToken; route++) {
          auto write = layout.write(thread, routes, route);
          if (write.write) {
            collected_scale = load_scale<Float32>(output_scales, write.output_row);
            break;
          }
        }
      }

      uint32_t group = thread.num_values == 0 ? 0 : thread.column / G;
      float static_scale = 1.f;
      if constexpr (Config::kStaticTensorScale)
        static_scale *= __ldg(static_tensor_scales + thread.expert);

      auto result = quant_group<
          TargetType,
          QuantScaleType,
          V,
          kScaleSize,
          kNumWarps,
          0,
          Config::kStaticScale,
          Config::kQuantScaleMode,
          Config::kPhase>(values, scratch + kTransformScratch, static_scale, collected_scale);
      ScaleStorage<OutputScaleType> group_output_scale{};
      float token_scale = 0.f;
      if constexpr (Config::kScaleMode == ScaleMode::DynamicGroupToken) {
        float m3_scale = decode_scale<QuantScaleType>(result.scale);
        float token_maximum = token_group_scale_max<
            G,
            V,
            Config::kThreadsPerTask,
            kNumWarps,
            0>(m3_scale, scratch + kTransformScratch);
        token_scale = token_scale_from_m3_bits(__float_as_uint(token_maximum));
        float local_scale = token_scale > 0.f ? m3_scale / token_scale : 0.f;
        group_output_scale = encode_scale<OutputScaleType>(local_scale);
      } else {
        group_output_scale = result.scale;
      }
      bool scale_leader;
      if constexpr (Config::kScaleMode == ScaleMode::DynamicToken) {
        scale_leader = thread.column == 0;
      } else if constexpr (Config::kUseTilePartition && Config::kTilesPerBlock == 1) {
        scale_leader = threadIdx.x == 0;
      } else {
        scale_leader = thread.column % G == 0;
      }

      PRAGMA_UNROLL
      for (uint32_t route = 0; route < Layout::kOutputsPerToken; route++) {
        auto write = layout.write(thread, routes, route);
        if (write.write) {
          if constexpr (Config::kScaleMode != ScaleMode::Static && Config::kPhase != QuantizationPhase::Quantize) {
            if (scale_leader) {
              uint64_t scale_index = write.output_row;
              if constexpr (
                  Config::kScaleMode == ScaleMode::DynamicGroup ||
                  Config::kScaleMode == ScaleMode::DynamicGroupToken) {
                constexpr bool kLinearGroupScale =
                    Config::kLayout == LayoutType::Normal &&
                    Config::kUseTilePartition &&
                    Config::kGroupScaleLayout == GroupScaleLayout::RowMajor &&
                    Layout::kFullColumns;
                if constexpr (kLinearGroupScale) {
                  constexpr uint32_t kLanesPerGroup = G / V;
                  scale_index = static_cast<uint64_t>(blockIdx.x) * Config::kTilesPerBlock + thread.lane / kLanesPerGroup;
                } else {
                  scale_index = group_scale_index<Config, OutputScaleType>(write.output_row, group, group_scale_stride);
                }
              }
              ScaleStorage<OutputScaleType> stored_scale = write.zero ? ScaleStorage<OutputScaleType>{} : group_output_scale;
              store_scale<OutputScaleType>(output_scales, scale_index, stored_scale);
            }
            if constexpr (Config::kScaleMode == ScaleMode::DynamicGroupToken)
              if (thread.column == 0) token_scales[write.output_row] = write.zero ? 0.f : token_scale;
          }
          if constexpr (Config::kPhase != QuantizationPhase::CollectAbsmax) {
            store_packed<kPackedBytes>(write.template output<TargetType::kBits>(output), result.packed, write.zero);
          }
        }
      }
    } else {
      PRAGMA_UNROLL
      for (uint32_t route = 0; route < Layout::kOutputsPerToken; route++) {
        auto write = layout.write(thread, routes, route);
        if (write.write && hadamard_valid) {
          SourceType *destination = reinterpret_cast<SourceType *>(write.template output<sizeof(SourceType) * 8>(output));
          store_values<SourceType, V>(destination, values, write.zero);
        }
      }
    }
  }

  if constexpr (Config::kUsePdl) {
    __syncthreads();
    if (threadIdx.x == 0) griddepcontrol_launch_dependents();
  }
}


// Scale-only second stage for DynamicGroup E4M3 + DynamicToken.  One warp
// owns one logical token; layout routing is repeated without reading input.
template <class Config, uint32_t kTokensPerBlock>
__global__ __launch_bounds__(kTokensPerBlock * 32) void finalize_group_token_scales_kernel(
    const void *__restrict__ intermediate_scales,
    void *__restrict__ group_scales,
    float *__restrict__ token_scales,
    const void *__restrict__ expert_layout,
    const void *__restrict__ indices,
    uint64_t num_input_rows,
    uint64_t num_output_rows,
    uint32_t num_experts,
    uint32_t max_tokens_per_expert,
    uint64_t group_scale_stride) {
  using Layout = typename Config::Layout;
  using ScaleLayout = InputLayout<
      Layout::kType,
      1,
      32,
      1,
      kTokensPerBlock,
      Layout::kScatterWidth,
      Layout::kScatterSingleOutput,
      Layout::kScatterSingleOutput,
      Layout::kExpertLayoutInt64,
      Layout::kIndexInt64,
      Layout::kZeroInvalid>;
  constexpr uint32_t kGroupsPerToken = Config::kHiddenSize / Config::kQuantGroupSize;

  static_assert(Config::kScaleMode == ScaleMode::DynamicGroup);
  static_assert(std::is_same<typename Config::OutputScaleType, M3BFloat16>::value);
  static_assert(Config::kPhase == QuantizationPhase::Fused);
  static_assert(kGroupsPerToken >= 1);
  static_assert(kTokensPerBlock >= 1 && kTokensPerBlock <= 32);

  if constexpr (Config::kUsePdl) griddepcontrol_wait();

  __shared__ typename ScaleLayout::SharedStorage layout_storage;

  InputLayoutParams layout_params{
      expert_layout,
      indices,
      num_input_rows,
      num_output_rows,
      num_experts,
      max_tokens_per_expert};
  ScaleLayout layout(layout_params, &layout_storage);
  layout.prepare();
  uint32_t token_in_block = threadIdx.x / 32;
  uint32_t lane = threadIdx.x & 31;
  auto task = layout.token(token_in_block);

  uint64_t source_row = ScaleLayout::kInvalidRow;
  PRAGMA_UNROLL
  for (uint32_t route = 0; route < ScaleLayout::kOutputsPerToken; route++)
    if (task.output_rows[route] != ScaleLayout::kInvalidRow && source_row == ScaleLayout::kInvalidRow)
      source_row = task.output_rows[route];
  bool active = source_row != ScaleLayout::kInvalidRow;
  uint64_t source_index = 0;
  if (active) source_index = group_scale_index<Config, M3BFloat16>(source_row, 0, group_scale_stride);
  const uint16_t *input = reinterpret_cast<const uint16_t *>(intermediate_scales) + source_index;
  uint32_t maximum = 0;
  if (active)
    for (uint32_t group = lane; group < kGroupsPerToken; group += 32)
      maximum = max(maximum, static_cast<uint32_t>(input[group]));
#if __CUDA_ARCH__ >= 800
  maximum = warp_max(maximum);
#else
  PRAGMA_UNROLL
  for (uint32_t step = 16; step >= 1; step >>= 1)
    maximum = max(maximum, __shfl_down_sync(0xFFFFFFFFu, maximum, step));
  maximum = __shfl_sync(0xFFFFFFFFu, maximum, 0);
#endif

  float token_scale = 0.f;
  if (maximum != 0) token_scale = token_scale_from_m3(static_cast<uint16_t>(maximum));
  PRAGMA_UNROLL
  for (uint32_t route = 0; route < ScaleLayout::kOutputsPerToken; route++) {
    uint64_t output_row = task.output_rows[route];
    if (active && output_row != ScaleLayout::kInvalidRow) {
      if (lane == 0) token_scales[output_row] = token_scale;
      for (uint32_t group = lane; group < kGroupsPerToken; group += 32) {
        float scale = token_scale > 0.f ? decode_scale<M3BFloat16>(input[group]) / token_scale : 0.f;
        uint64_t scale_index = group_scale_index<Config, Float8E4M3>(output_row, group, group_scale_stride);
        store_scale<Float8E4M3>(group_scales, scale_index, encode_scale<Float8E4M3>(scale));
      }
    }
  }

  if constexpr (Config::kUsePdl) {
    __syncthreads();
    if (threadIdx.x == 0) griddepcontrol_launch_dependents();
  }
}
