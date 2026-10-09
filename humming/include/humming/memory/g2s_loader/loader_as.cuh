#pragma once

#include <humming/utils/all.cuh>


template <class Ctx, bool kSecondary = false>
class G2SMemoryLoaderAS {
private:
  using SharedStorage = typename Ctx::SharedStorage;
  using ProblemShape = typename Ctx::ProblemShape;
  using BlockShape = typename Ctx::BlockShape;
  using PadShape = typename Ctx::PadShape;
  using ElementA = typename Ctx::ElementA;

  static constexpr bool kUseBlockScaledMma = Ctx::kUseBlockScaledMma;
  static constexpr bool kUseWarpSpec = Ctx::kUseWarpSpec;
  static constexpr bool kUseCpAsync = Ctx::kUseCpAsync;
  static constexpr bool kIsIndexedGemm = Ctx::kIsIndexedGemm;
  static constexpr bool kIsGroupedGemm = Ctx::kIsGroupedGemm;

  static constexpr uint32_t kNumLoadThreads = Ctx::kNumLoadThreads;
  static constexpr uint32_t kLoadThreadOffset = Ctx::kLoadThreadOffset;

  static constexpr bool kConfiguredInputScale = kSecondary ? Ctx::kHasInputScale2 : Ctx::kHasInputScale;
  static constexpr bool kIsTensorScale = kSecondary ? Ctx::kIsTensorInputScale2 : Ctx::kIsTensorInputScale;
  static constexpr bool kHasInputScale = kConfiguredInputScale && !kIsTensorScale;
  static constexpr bool kIsChannelScale = kHasInputScale && (kSecondary || !Ctx::kIsGroupInputScale);
  static constexpr bool kIsGroupScale = kHasInputScale && !kSecondary && Ctx::kIsGroupInputScale;
  static constexpr bool kUseMxScale = kUseBlockScaledMma && kIsGroupScale;
  static constexpr bool kMMajorInputScale = Ctx::kUseMMajorInputScale && kIsGroupScale;
  static_assert(!kMMajorInputScale || !kIsIndexedGemm);
  static constexpr bool kConfiguredUseTma = kSecondary ? Ctx::kUseTmaAS2 : Ctx::kUseTmaAS;
  static constexpr bool kUseTma = kConfiguredUseTma && kHasInputScale && !kIsIndexedGemm;
  static_assert(!kConfiguredUseTma || !kIsTensorScale);
  static_assert(!kUseTma || kMMajorInputScale || kIsChannelScale || kUseBlockScaledMma);
  static constexpr bool kUseRowMajorTma = kUseTma && kUseMxScale && !kMMajorInputScale;
  static_assert(!kUseRowMajorTma || SharedStorage::kUseUmmaRowMajorSmemInputScale);
  static constexpr uint32_t kGroupSize = kIsGroupScale ? Ctx::kInputScaleGroupSize : ProblemShape::K;

  static_assert(ProblemShape::K == kGroupSize || (ProblemShape::K - PadShape::K) % kGroupSize == 0);
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;
  static constexpr uint32_t kProblemNumGroups = CEIL_DIV(ProblemShape::K - PadShape::K, kGroupSize);
  static constexpr uint32_t kNumGroups = CEIL_DIV(BlockShape::K, kGroupSize);
  static constexpr uint32_t kMxScaleVec = kPartMmaShapeK / kGroupSize;
  static constexpr uint32_t kLoadsPerGroup = kUseMxScale ? MAX(1u, 4 / kNumGroups) : CEIL_DIV(kGroupSize, BlockShape::K);
  static constexpr uint32_t kRowLoadIters = CEIL_DIV(BlockShape::M, kNumLoadThreads);
  static constexpr uint32_t kScaleMAlignment = 4;
  static constexpr uint32_t kScaleBlockM = BlockShape::M + (kIsGroupedGemm ? kScaleMAlignment : 0);
  static constexpr uint32_t kScaleBlockMVecs = kScaleBlockM / kScaleMAlignment;
  static_assert(BlockShape::M % kScaleMAlignment == 0);
  // A TMA box holds at most 256 elements per dimension: wider M-major tiles use 64-bit elements.
  static constexpr bool kUseWideTmaElements = kUseTma && kUseMxScale && kMMajorInputScale && kScaleBlockM > 256;
  static_assert(!kUseTma || kScaleBlockM <= 256 || kUseMxScale);

  // One uint4 gather per routed row covers four single-group stages. Stream-K is
  // excluded so every tile starts on a vector boundary; the plain smem stores
  // rely on the CTA barrier of the non-mbarrier pipeline.
  static constexpr bool kUseIndexedScaleCache =
      kIsIndexedGemm && kIsGroupScale && !kUseMxScale && Ctx::kWgmmaUseLateAS && !Ctx::kUseMBarrier &&
      !Ctx::kUseStreamK && kNumGroups == 1 && kProblemNumGroups % 4 == 0;

  using LoadType = typename LoadTypeChooser<kNumGroups * 4>::Type;

public:
  Ctx &ctx;
  const CUtensorMap *tensor_map_ptr;
  const uint32_t *gmem_ptr_raw;
  const uint32_t *gmem_ptr;

  uint32_t shape_m;
  uint32_t total_shape_m;
  uint32_t block_shape_m;
  uint32_t row_offset;
  uint32_t load_row_offset;
  uint32_t load_row_index[kRowLoadIters];
  uint32_t col_offset = 0;
  uint32_t counter = 0;
  uint4 indexed_scale_cache[kRowLoadIters];

  CUDA_INLINE
  G2SMemoryLoaderAS(Ctx &ctx)
      : ctx(ctx),
        shape_m(ctx.params.shape_m),
        total_shape_m(CEIL_DIV(ctx.params.shape_m, kScaleMAlignment) * kScaleMAlignment) {
    const void *ptr = kSecondary ? ctx.params.as2 : ctx.params.as;
    if constexpr (kUseTma) {
      tensor_map_ptr = reinterpret_cast<const CUtensorMap *>(ptr);
    } else {
      gmem_ptr_raw = reinterpret_cast<const uint32_t *>(ptr);
    }
  }

  template <bool kShouldAdvance = true>
  CUDA_INLINE void load(void *smem_ptr, void *mbar_ptr) {
    counter = kLoadsPerGroup != 1 ? (counter + 1) % kLoadsPerGroup : 0;
    if constexpr (kUseMxScale) {
      if constexpr (kUseTma) load_mx_tma(smem_ptr, mbar_ptr);
      else if constexpr (kMMajorInputScale) load_mx_legacy_m_major(smem_ptr);
      else if constexpr (SharedStorage::kUseUmmaRowMajorSmemInputScale) load_mx_legacy_row_major(smem_ptr);
      else load_mx_legacy(smem_ptr);
    } else if constexpr (kUseTma) load_tma(smem_ptr, mbar_ptr);
    else if constexpr (kUseIndexedScaleCache) load_legacy_indexed_cached(smem_ptr);
    else load_legacy(smem_ptr);
    if constexpr (kShouldAdvance) advance();
  }

  CUDA_INLINE void load_mx_legacy_row_major(void *smem_ptr) {
    constexpr uint32_t kRowWords = BlockShape::K / (sizeof(uint32_t) * kGroupSize);
    constexpr uint32_t kGmemStrideWords = CEIL_DIV(kProblemNumGroups, 4);
    constexpr bool kHasAlignedRows = kGmemStrideWords % 4 == 0;
    constexpr bool kUseWideRows = SharedStorage::kUseUmmaWideRowInputScale;
    constexpr uint32_t kSlotWords = kUseWideRows ? 4 : kRowWords;
    constexpr uint32_t kLoadWords = kHasAlignedRows && kSlotWords % 4 == 0 ? 4 : 1;
    using RowLoadType = typename LoadTypeChooser<kLoadWords * 4>::Type;
    // cp.async bypasses L1 only for 16-byte loads: load the aligned vector holding this stage.
    const uint32_t vector_word = kUseWideRows ? (col_offset / 4) % 4 : 0;
    const uint32_t source_shift = kLoadWords == 4 ? vector_word : 0;
    const uint32_t slot_shift = kLoadWords == 4 ? 0 : vector_word;
    constexpr uint32_t kRowLoads = (kLoadWords == 4 ? kSlotWords : kRowWords) / kLoadWords;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kRowLoadIters; i++) {
      uint32_t row = i * kNumLoadThreads + ctx.load_thread_id();
      uint32_t source_row = kIsIndexedGemm ? load_row_index[i] : row;
      bool is_valid_row = kIsIndexedGemm ? (row < BlockShape::M && source_row < shape_m) : (row < block_shape_m);
      const uint32_t *source_words = gmem_ptr + source_row * kGmemStrideWords - source_shift;
      uint32_t *slot_words = reinterpret_cast<uint32_t *>(smem_ptr) + row * kSlotWords + slot_shift;
      const auto *source = reinterpret_cast<const RowLoadType *>(source_words);
      auto *destination = reinterpret_cast<RowLoadType *>(slot_words);
      PRAGMA_UNROLL
      for (uint32_t part = 0; part < kRowLoads; part++) {
        legacy_load_pred<kUseCpAsync>(source + part, destination + part, is_valid_row);
      }
    }
  }

  CUDA_INLINE void load_mx_legacy(void *smem_ptr) {
    uint32_t thread_id = ctx.load_thread_id();
    uint32_t *smem_ptr_load = reinterpret_cast<uint32_t *>(smem_ptr);
    const uint32_t *gmem_ptr_load = reinterpret_cast<const uint32_t *>(gmem_ptr);

    constexpr uint32_t kNumRows = CEIL_DIV(BlockShape::K / kPartMmaShapeK * kMxScaleVec, 4);
    constexpr uint32_t kMxGmemStride = CEIL_DIV(kProblemNumGroups, 4);
    constexpr uint32_t kNumInts = BlockShape::M * kNumRows;

    if constexpr (kNumInts <= kNumLoadThreads) {
      uint32_t logical_offset = thread_id;
      uint32_t smem_row = logical_offset / BlockShape::M;
      uint32_t smem_col = logical_offset % BlockShape::M;
      uint32_t smem_offset = smem_row * kScaleBlockM + smem_col;

      uint32_t gmem_row = smem_col;
      if constexpr (kIsIndexedGemm) gmem_row = ctx.get_rd_row_index()[smem_col];
      uint32_t gmem_col = smem_row;
      uint32_t gmem_offset = gmem_row * kMxGmemStride + gmem_col;
      uint32_t row_bound = kIsIndexedGemm ? shape_m : block_shape_m;
      uint32_t pred = logical_offset < kNumInts && gmem_row < row_bound;

      legacy_load_pred<kUseCpAsync>(gmem_ptr_load + gmem_offset, smem_ptr_load + smem_offset, pred);
    } else {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kNumRows; i++) {
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < CEIL_DIV(BlockShape::M, kNumLoadThreads); j++) {
          uint32_t m_index = j * kNumLoadThreads + thread_id;
          uint32_t gmem_row = kIsIndexedGemm ? load_row_index[j] : m_index;
          uint32_t gmem_offset = gmem_row * kMxGmemStride + i;
          uint32_t smem_offset = i * kScaleBlockM + m_index;
          uint32_t row_bound = kIsIndexedGemm ? shape_m : block_shape_m;
          uint32_t pred = gmem_row < row_bound;

          legacy_load_pred<kUseCpAsync>(gmem_ptr_load + gmem_offset, smem_ptr_load + smem_offset, pred);
        }
      }
    }
  }

  CUDA_INLINE void load_mx_legacy_m_major(void *smem_ptr) {
    uint32_t thread_id = ctx.load_thread_id();
    const int4 *gmem_ptr_load = reinterpret_cast<const int4 *>(gmem_ptr);
    int4 *smem_ptr_load = reinterpret_cast<int4 *>(smem_ptr);
    constexpr uint32_t kNumRows = CEIL_DIV(BlockShape::K / kPartMmaShapeK * kMxScaleVec, 4);
    uint32_t total_shape_m_vecs = total_shape_m / kScaleMAlignment;
    uint32_t load_shape_m = load_row_offset < total_shape_m ? MIN(total_shape_m - load_row_offset, kScaleBlockM) : 0;
    PRAGMA_UNROLL
    for (uint32_t r = 0; r < kNumRows; r++) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < CEIL_DIV(kScaleBlockMVecs, kNumLoadThreads); i++) {
        uint32_t m_vec = i * kNumLoadThreads + thread_id;
        uint32_t gmem_offset = r * total_shape_m_vecs + m_vec;
        uint32_t smem_offset = r * kScaleBlockMVecs + m_vec;
        legacy_load_pred<kUseCpAsync>(
            gmem_ptr_load + gmem_offset,
            smem_ptr_load + smem_offset,
            m_vec < kScaleBlockMVecs && m_vec * kScaleMAlignment < load_shape_m);
      }
    }
  }

  CUDA_INLINE void load_tma(void *smem_ptr, void *mbar_ptr) {
    static_assert(!kIsIndexedGemm && (kMMajorInputScale || kIsChannelScale));
    constexpr uint32_t kLoadThread = Ctx::kUseUmmaSplitLoads && !kIsChannelScale ? 32 : 0;
    if (ctx.load_thread_id() == kLoadThread) {
      if constexpr (kIsChannelScale) tma_load_1d(tensor_map_ptr, smem_ptr, mbar_ptr, load_row_offset);
      else tma_load_2d<>(tensor_map_ptr, smem_ptr, mbar_ptr, load_row_offset, col_offset);
    }
  }

  CUDA_INLINE uint32_t m_major_tma_row() { return kUseWideTmaElements ? load_row_offset / 2 : load_row_offset; }

  CUDA_INLINE uint32_t row_major_tma_column() {
    return SharedStorage::kUseUmmaWideRowInputScale ? col_offset / 16 * 4 : col_offset / 4;
  }

  CUDA_INLINE void load_mx_tma(void *smem_ptr, void *mbar_ptr) {
    static_assert(!kIsIndexedGemm);
    constexpr uint32_t kLoadThread = Ctx::kUseUmmaSplitLoads ? 32 : 0;
    if (ctx.load_thread_id() != kLoadThread) return;
    if constexpr (kMMajorInputScale) tma_load_2d<>(tensor_map_ptr, smem_ptr, mbar_ptr, m_major_tma_row(), col_offset / 4);
    else tma_load_2d<>(tensor_map_ptr, smem_ptr, mbar_ptr, row_major_tma_column(), load_row_offset);
  }

  CUDA_INLINE void prefetch_tma() {
    if constexpr (kUseTma) {
      if (ctx.load_thread_id() == 0) {
        if constexpr (kUseRowMajorTma) tma_prefetch_2d(tensor_map_ptr, row_major_tma_column(), load_row_offset);
        else if constexpr (kUseMxScale) tma_prefetch_2d(tensor_map_ptr, m_major_tma_row(), col_offset / 4);
        else if constexpr (kIsChannelScale) tma_prefetch_1d(tensor_map_ptr, load_row_offset);
        else tma_prefetch_2d(tensor_map_ptr, load_row_offset, col_offset);
      }
    }
  }

  CUDA_INLINE void load_legacy_indexed_cached(void *smem_ptr) {
    uint32_t thread_id = ctx.load_thread_id();
    uint32_t *smem_ptr_load = reinterpret_cast<uint32_t *>(smem_ptr);
    const uint32_t component = col_offset % 4;
    PRAGMA_UNROLL
    for (uint32_t i = 0; i < kRowLoadIters; i++) {
      uint32_t row = i * kNumLoadThreads + thread_id;
      uint32_t source_row = load_row_index[i];
      if (row < BlockShape::M && source_row < shape_m) {
        if (component == 0)
          indexed_scale_cache[i] = *reinterpret_cast<const uint4 *>(gmem_ptr + source_row * kProblemNumGroups);
        const uint4 v = indexed_scale_cache[i];
        smem_ptr_load[row] = component == 0 ? v.x : component == 1 ? v.y : component == 2 ? v.z : v.w;
      }
    }
  }

  CUDA_INLINE void load_legacy(void *smem_ptr) {
    uint32_t thread_id = ctx.load_thread_id();
    if constexpr (!kIsIndexedGemm && (kMMajorInputScale || kIsChannelScale)) {
      constexpr uint32_t kWindowNumGroups = kMMajorInputScale ? kNumGroups : 1;
      const uint32_t total_shape_m_vecs = total_shape_m / kScaleMAlignment;
      const int4 *gmem_ptr_load = reinterpret_cast<const int4 *>(gmem_ptr);
      int4 *smem_ptr_load = reinterpret_cast<int4 *>(smem_ptr);

      PRAGMA_UNROLL
      for (uint32_t g = 0; g < kWindowNumGroups; g++) {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < CEIL_DIV(kScaleBlockMVecs, kNumLoadThreads); i++) {
          uint32_t m_vec = i * kNumLoadThreads + thread_id;
          if (m_vec < kScaleBlockMVecs) {
            uint32_t global_row = load_row_offset + m_vec * kScaleMAlignment;
            uint32_t gmem_offset = m_vec + (kMMajorInputScale ? g * total_shape_m_vecs : 0);
            uint32_t smem_offset = g * kScaleBlockMVecs + m_vec;
            legacy_load_pred<kUseCpAsync>(gmem_ptr_load + gmem_offset, smem_ptr_load + smem_offset, global_row < total_shape_m);
          }
        }
      }
    } else {
      constexpr uint32_t kSmemStride = kNumGroups / (sizeof(LoadType) / 4);
      constexpr uint32_t kGmemStride = kProblemNumGroups / (sizeof(LoadType) / 4);

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < CEIL_DIV(BlockShape::M, kNumLoadThreads); i++) {
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < kSmemStride; j++) {
          uint32_t smem_offset = (i * kNumLoadThreads + thread_id) * kSmemStride + j;
          uint32_t smem_row = smem_offset / kSmemStride;
          uint32_t smem_col = smem_offset % kSmemStride;

          uint32_t gmem_row = kIsIndexedGemm ? load_row_index[i] : smem_row;
          uint32_t gmem_offset = gmem_row * kGmemStride + smem_col;

          const LoadType *gmem_ptr_load = reinterpret_cast<const LoadType *>(gmem_ptr);
          LoadType *smem_ptr_load = reinterpret_cast<LoadType *>(smem_ptr);
          bool pred = kIsIndexedGemm ? (gmem_row < shape_m) : (smem_row < block_shape_m);
          legacy_load_pred<kUseCpAsync>(gmem_ptr_load + gmem_offset, smem_ptr_load + smem_offset, pred);
        }
      }
    }
  }

  CUDA_INLINE
  void advance() {
    if (kIsGroupScale && (kLoadsPerGroup == 1 || counter == 0)) {
      col_offset += kUseMxScale ? CEIL_DIV(kNumGroups, 4) * 4 : kNumGroups;
      if constexpr (!kUseTma) {
        if constexpr (kUseMxScale) {
          if constexpr (kMMajorInputScale) gmem_ptr += CEIL_DIV(kNumGroups, 4) * total_shape_m;
          else gmem_ptr += CEIL_DIV(kNumGroups, 4);
        } else if constexpr (kMMajorInputScale) {
          gmem_ptr += kNumGroups * total_shape_m;
        } else {
          gmem_ptr += kNumGroups;
        }
      }
    }
  }

  CUDA_INLINE
  void seek(uint32_t, uint32_t m_block_id, uint32_t k_block_id, uint32_t current_shape_m, uint32_t m_offset) {
    if constexpr (kUseMxScale) counter = 0;
    if constexpr (kIsGroupScale) {
      if constexpr (BlockShape::K >= kGroupSize) {
        col_offset = k_block_id * kNumGroups;
      } else {
        col_offset = (k_block_id * BlockShape::K) / kGroupSize;
      }
    } else {
      col_offset = 0;
    }

    if constexpr (kIsGroupedGemm) {
      shape_m = current_shape_m;
      row_offset = m_offset;
    } else {
      row_offset = m_block_id * BlockShape::M;
    }
    block_shape_m = row_offset < shape_m ? MIN(shape_m - row_offset, BlockShape::M) : 0;
    if constexpr (!kIsIndexedGemm && (kMMajorInputScale || kIsChannelScale)) {
      load_row_offset = row_offset - row_offset % kScaleMAlignment;
    } else {
      load_row_offset = row_offset;
    }

    if constexpr (!kIsIndexedGemm) {
      if constexpr (kUseMxScale) {
        if constexpr (!kUseTma) {
          if constexpr (kMMajorInputScale)
            gmem_ptr = gmem_ptr_raw + ((col_offset / 4) * total_shape_m + MIN(load_row_offset, total_shape_m));
          else
            gmem_ptr = gmem_ptr_raw + (row_offset * CEIL_DIV(kProblemNumGroups, 4) + col_offset / 4);
        }
      } else if constexpr (kUseTma) {
        // tma loads via tensor map; gmem_ptr unused
      } else if constexpr (kMMajorInputScale) {
        gmem_ptr = gmem_ptr_raw + (col_offset * total_shape_m + MIN(load_row_offset, total_shape_m));
      } else if constexpr (kIsChannelScale) {
        gmem_ptr = gmem_ptr_raw + MIN(load_row_offset, shape_m);
      } else {
        gmem_ptr = gmem_ptr_raw + ((row_offset * kProblemNumGroups) + col_offset);
      }
    } else {
      gmem_ptr = gmem_ptr_raw + (kUseMxScale ? col_offset / 4 : col_offset);

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < kRowLoadIters; i++) {
        uint32_t smem_row = i * kNumLoadThreads + ctx.load_thread_id();
        load_row_index[i] = smem_row < BlockShape::M ? ctx.get_rd_row_index()[smem_row] : shape_m;
      }
    }
  }
};
