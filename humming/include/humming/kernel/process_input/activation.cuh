#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <humming/utils/all.cuh>


enum class ActivationType : uint32_t {
  None = 0,
  Unary = 1,
  // [gate_0 ... gate_K-1, up_0 ... up_K-1].
  BinarySplit = 2,
  // [gate_0, up_0, ... gate_K-1, up_K-1].
  BinaryInterleaved = 3,
};


template <class SourceType, uint32_t kValues>
CUDA_INLINE void load_values(float *values, const SourceType *input) {
  static_assert(std::is_same<SourceType, float>::value || std::is_same<SourceType, __half>::value || std::is_same<SourceType, __nv_bfloat16>::value);
  using LoadType = typename LoadTypeChooser<sizeof(SourceType) * kValues>::Type;
  constexpr uint32_t kValuesPerLoad = sizeof(LoadType) / sizeof(SourceType);
  static_assert(kValues % kValuesPerLoad == 0);

  PRAGMA_UNROLL
  for (uint32_t value = 0; value < kValues; value += kValuesPerLoad) {
    LoadType loaded = *reinterpret_cast<const LoadType *>(input + value);
    if constexpr (std::is_same<SourceType, __nv_bfloat16>::value && kValuesPerLoad >= 2) {
      const uint32_t *words = reinterpret_cast<const uint32_t *>(&loaded);
      PRAGMA_UNROLL
      for (uint32_t word = 0; word < kValuesPerLoad / 2; word++) {
        uint32_t low, high;
        asm("shl.b32 %0, %2, 16; and.b32 %1, %2, 0xffff0000;"
            : "=&r"(low), "=&r"(high)
            : "r"(words[word]));
        values[value + word * 2] = __uint_as_float(low);
        values[value + word * 2 + 1] = __uint_as_float(high);
      }
    } else {
      const SourceType *items = reinterpret_cast<const SourceType *>(&loaded);
      PRAGMA_UNROLL
      for (uint32_t item = 0; item < kValuesPerLoad; item++)
        values[value + item] = static_cast<float>(items[item]);
    }
  }
}


template <class ValueType, uint32_t kValues>
CUDA_INLINE void load_raw_values(ValueType *values, const ValueType *input) {
  using LoadType = typename LoadTypeChooser<sizeof(ValueType) * kValues>::Type;
  constexpr uint32_t kValuesPerLoad = sizeof(LoadType) / sizeof(ValueType);
  static_assert(kValues % kValuesPerLoad == 0);

  PRAGMA_UNROLL
  for (uint32_t value = 0; value < kValues; value += kValuesPerLoad)
    *reinterpret_cast<LoadType *>(values + value) = *reinterpret_cast<const LoadType *>(input + value);
}


template <class ValueType, uint32_t kValues>
CUDA_INLINE void store_raw_values(ValueType *output, const ValueType *values, bool zero) {
  using StoreType = typename LoadTypeChooser<sizeof(ValueType) * kValues>::Type;
  constexpr uint32_t kValuesPerStore = sizeof(StoreType) / sizeof(ValueType);
  static_assert(kValues % kValuesPerStore == 0);

  PRAGMA_UNROLL
  for (uint32_t value = 0; value < kValues; value += kValuesPerStore) {
    StoreType stored{};
    if (!zero) stored = *reinterpret_cast<const StoreType *>(values + value);
    *reinterpret_cast<StoreType *>(output + value) = stored;
  }
}


template <class ValueType, uint32_t kValues>
CUDA_INLINE void store_values(ValueType *output, const float *values, bool zero) {
  using StoreType = typename LoadTypeChooser<sizeof(ValueType) * kValues>::Type;
  constexpr uint32_t kValuesPerStore = sizeof(StoreType) / sizeof(ValueType);
  static_assert(kValues % kValuesPerStore == 0);

  PRAGMA_UNROLL
  for (uint32_t value = 0; value < kValues; value += kValuesPerStore) {
    alignas(16) ValueType converted[kValuesPerStore];
    if constexpr (std::is_same<ValueType, __nv_bfloat16>::value && kValuesPerStore >= 2) {
      PRAGMA_UNROLL
      for (uint32_t item = 0; item < kValuesPerStore; item += 2)
        reinterpret_cast<__nv_bfloat162 *>(converted)[item / 2] =
            __floats2bfloat162_rn(values[value + item], values[value + item + 1]);
    } else if constexpr (std::is_same<ValueType, __half>::value && kValuesPerStore >= 2) {
      PRAGMA_UNROLL
      for (uint32_t item = 0; item < kValuesPerStore; item += 2)
        reinterpret_cast<__half2 *>(converted)[item / 2] =
            __floats2half2_rn(values[value + item], values[value + item + 1]);
    } else {
      PRAGMA_UNROLL
      for (uint32_t item = 0; item < kValuesPerStore; item++)
        converted[item] = static_cast<ValueType>(values[value + item]);
    }
    StoreType stored{};
    if (!zero) stored = *reinterpret_cast<StoreType *>(converted);
    *reinterpret_cast<StoreType *>(output + value) = stored;
  }
}


template <class Activation, uint32_t kHiddenSize, uint32_t kValuesPerThread>
class InputActivation {
public:
  static constexpr ActivationType kType = Activation::kType;
  static constexpr bool kBinary =
      kType == ActivationType::BinarySplit || kType == ActivationType::BinaryInterleaved;
  static constexpr uint32_t kInputElementsPerRow = kBinary ? kHiddenSize * 2 : kHiddenSize;

  template <class SourceType>
  CUDA_INLINE static void load(float *values, const SourceType *input, uint64_t input_row, uint32_t column) {
    const SourceType *row = input + input_row * kInputElementsPerRow;
    if constexpr (!kBinary) {
      load_values<SourceType, kValuesPerThread>(values, row + column);
      if constexpr (kType == ActivationType::Unary) {
        PRAGMA_UNROLL
        for (uint32_t value = 0; value < kValuesPerThread; value++)
          values[value] = Activation::apply(values[value]);
      }
    } else if constexpr (kType == ActivationType::BinarySplit) {
      constexpr uint32_t kVectorValues = std::is_same<SourceType, float>::value ? 4 : 8;
      constexpr uint32_t kChunk = kValuesPerThread < kVectorValues ? kValuesPerThread : kVectorValues;
      static_assert(kValuesPerThread % kChunk == 0);
      PRAGMA_UNROLL
      for (uint32_t value = 0; value < kValuesPerThread; value += kChunk) {
        float gate[kChunk], up[kChunk];
        load_values<SourceType, kChunk>(gate, row + column + value);
        load_values<SourceType, kChunk>(up, row + kHiddenSize + column + value);
        PRAGMA_UNROLL
        for (uint32_t item = 0; item < kChunk; item++)
          values[value + item] = Activation::apply(gate[item], up[item]);
      }
    } else {
      constexpr uint32_t kSourceVectorValues = std::is_same<SourceType, float>::value ? 4 : 8;
      constexpr uint32_t kNaturalChunk = kSourceVectorValues / 2;
      constexpr uint32_t kChunk = kValuesPerThread < kNaturalChunk ? kValuesPerThread : kNaturalChunk;
      static_assert(kValuesPerThread % kChunk == 0);
      PRAGMA_UNROLL
      for (uint32_t value = 0; value < kValuesPerThread; value += kChunk) {
        float pairs[kChunk * 2];
        load_values<SourceType, kChunk * 2>(pairs, row + (column + value) * 2);
        PRAGMA_UNROLL
        for (uint32_t item = 0; item < kChunk; item++)
          values[value + item] = Activation::apply(pairs[item * 2], pairs[item * 2 + 1]);
      }
    }
  }
};
