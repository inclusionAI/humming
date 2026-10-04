#pragma once

#include <humming/arith/exp_offset.cuh>
#include <humming/datatype/base_conversion.cuh>
#include <humming/datatype/dequant.cuh>
#include <humming/datatype/dtypes.cuh>
#include <humming/utils/all.cuh>


template <class Ctx>
class MainloopArithmetic : F16Conversion<typename Ctx::ElementC> {
private:
  using MmaOpClass = typename Ctx::MmaOpClass;
  using BlockShape = typename Ctx::BlockShape;
  using WarpShape = typename Ctx::WarpShape;
  using ElementA = typename Ctx::ElementA;
  using ElementB = typename Ctx::ElementB;
  using ElementC = typename Ctx::ElementC;
  using ElementBS = typename Ctx::ElementBS;

  using scalar_t = typename F16Conversion<ElementC>::scalar_t;
  using scalar_t2 = typename F16Conversion<ElementC>::scalar_t2;
  using ValTypeC = typename MmaOpClass::ValTypeC;
  using MmaShape = typename MmaOpClass::MmaShape;

  static constexpr bool kUseWgmma = Ctx::kUseWgmma;
  static constexpr bool kUseMxmma = Ctx::kUseMxmma;
  static constexpr bool kIsF16Accum = MmaOpClass::kCTypeBits == 16;
  static constexpr uint32_t kPartMmaShapeK = 256 / ElementA::kBits;

  static constexpr bool kHasInputScale = ElementA::kBits != 16;
  static constexpr bool kIsGroupInputScale = kHasInputScale && Ctx::kInputScaleGroupSize > 0;
  static constexpr bool kIsGroupWeightScale = Ctx::kIsGroupWeightScale;
  static constexpr bool kIsBlockWeightScale = Ctx::kIsBlockWeightScale;
  static constexpr bool kIsChannelWeightScale = Ctx::kIsChannelWeightScale;
  static constexpr bool kIsGroupOrBlockWeightScale = kIsGroupWeightScale || kIsBlockWeightScale;

  static constexpr bool kHasZeroPoint = Ctx::kHasZeroPoint;
  static constexpr bool kIsFpZeroPoint = Ctx::kIsFpZeroPoint;
  static constexpr bool kUseIntWeightScale = Ctx::kUseIntWeightScale;
  static constexpr bool kUseFusedE8m0Scale = Ctx::kUseFusedE8m0Scale;
  static constexpr bool kUseNativeDequantB =
      Ctx::kUseNativeDequant && kUseNativeWeightDequant<ElementB, ElementA>;
  static constexpr bool kUseNativeDequantBS =
      Ctx::kUseNativeDequant && kIsGroupWeightScale && !kUseFusedE8m0Scale &&
      kNativeDequantSupported<ElementBS, ElementA>;

  static constexpr uint32_t kInputScaleGroupSize = kIsGroupInputScale ? Ctx::kInputScaleGroupSize : 1;
  static constexpr uint32_t kWeightScaleGroupSize = kIsGroupOrBlockWeightScale ? Ctx::kWeightScaleGroupSize : 1;
  static constexpr uint32_t kNumZPGroupsPerMma = Ctx::kUseBlockScaledMma && ElementA::kBits == 4 && kIsGroupWeightScale ? kPartMmaShapeK / kWeightScaleGroupSize : 1;

  static constexpr bool kUsePackedKLayout = Ctx::kUsePackedKLayout;
  static constexpr uint32_t kPackedKFactor = Ctx::kPackedKFactor;
  static constexpr uint32_t kNumKSlabs = WarpShape::K / kPartMmaShapeK;
  static constexpr uint2 kExpOffset = get_mainloop_exp_offset<
      ElementA, ElementB, ElementBS, kHasZeroPoint,
      kIsF16Accum, kIsGroupInputScale,
      kIsGroupOrBlockWeightScale && !kUseFusedE8m0Scale,
      MmaOpClass::kNativeMixed, kUseNativeDequantB, kUseNativeDequantBS>();

  static constexpr uint32_t kDequantBSBits = (ElementA::kBits < 16 && !kIsF16Accum) ? 32 : 16;
  static constexpr uint32_t kNumSubBlocksM = CEIL_DIV(WarpShape::M, 16);
  static constexpr uint32_t kNumSubBlocksN = WarpShape::N / 16;
  static constexpr uint32_t kNumASPerSubBlock = kUseWgmma ? 4 : 2;
  static constexpr uint32_t kNumBSPerSubBlock = !kUseFusedE8m0Scale && !kUseWgmma && ElementA::kBits < 16 ? 4 : 2;
  static constexpr uint32_t kNumASPerGroup = kNumSubBlocksM * kNumASPerSubBlock;
  static constexpr uint32_t kNumBSPerGroup = kNumSubBlocksN * kNumBSPerSubBlock;

public:
  uint32_t as[2][kNumASPerGroup];
  uint32_t q_as[kNumASPerGroup];
  static constexpr uint32_t kNumLoadedBS = kUsePackedKLayout && kUseFusedE8m0Scale ? 2 * kNumKSlabs : kNumBSPerGroup;
  alignas(16) uint32_t bs[2][MAX(kNumLoadedBS, 8) * ElementBS::kBits / 32];
  uint32_t dq_bs[MAX(kNumBSPerGroup, 8) * kDequantBSBits / 32];
  uint32_t zp[2][(kIsFpZeroPoint ? 4 : CEIL_DIV(ElementB::kBits, 4)) * kNumZPGroupsPerMma];

  uint32_t _dummy;

  CUDA_INLINE
  uint4 prepare_zp_for_dequant(uint32_t buffer_id, uint32_t index) {
    uint32_t zp_vals[4] = {0, 0, 0, 0};
    constexpr uint32_t kNumZPBits = 4 * CEIL_DIV(ElementB::kBits, 4);
    constexpr uint32_t kNumZPsPerInt = 32 / kNumZPBits;

    buffer_id = kIsChannelWeightScale ? 0 : buffer_id;

    if constexpr (kHasZeroPoint && !kIsFpZeroPoint) {
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 2; i++) {
        PRAGMA_UNROLL
        for (uint32_t j = 0; j < 2; j++) {
          uint32_t dzp_val;
          if constexpr (kNumZPGroupsPerMma > 1) {
            constexpr uint32_t kValuesPerLane = 32 / ElementA::kBits;
            constexpr uint32_t kZPBytesPerGroup = CEIL_DIV(kNumZPBits, 8);
            uint32_t k_offset = i * (kPartMmaShapeK / 2) + threadIdx.x % 4 * kValuesPerLane;
            uint32_t zp_group = k_offset / kWeightScaleGroupSize;
            uint32_t byte_id = index * kNumZPGroupsPerMma * kZPBytesPerGroup + zp_group * kZPBytesPerGroup;
            dzp_val = zp[buffer_id][byte_id / 4] >> (byte_id % 4 * 8);
            dzp_val = dzp_val >> (j * kNumZPBits);
          } else {
            uint32_t n_id = 0;
            if (WarpShape::N == 64 && ElementB::kBits > 4 && index >= 2) n_id = 1;
            dzp_val = zp[buffer_id][n_id];
            dzp_val = dzp_val >> ((index * 2 + j) % kNumZPsPerInt * kNumZPBits);
          }
          uint32_t zp_idx = kUseWgmma ? (i * 2 + j) : (j * 2 + i);
          zp_vals[zp_idx] = dequant_single_zero_point<ElementB, ElementA>(dzp_val);
        }
      }
    }

    return *reinterpret_cast<uint4 *>(zp_vals);
  };

  CUDA_INLINE
  void may_process_bs_before_apply_on_b(uint32_t j, uint32_t buffer_id) {

    if (j == 0) {
      if constexpr (ElementA::kBits == 16 && kIsBlockWeightScale) {
        scalar_t2 *dq_bs_scalar2_ptr = reinterpret_cast<scalar_t2 *>(dq_bs);
        float bs_float = reinterpret_cast<float *>(bs[buffer_id])[0];
        dq_bs_scalar2_ptr[0] = this->float2num2(bs_float);
      }
    }

    if (j % 2 == 0) {
      if constexpr (ElementA::kBits == 16 && ElementBS::kBits == 8 && kIsGroupWeightScale && !kUseFusedE8m0Scale) {
        if constexpr (kUseNativeDequantBS) {
          dequant_native<ElementBS, ElementA>(bs[buffer_id], dq_bs, 0);
        } else {
          dequant<ElementBS, ElementA>(bs[buffer_id], dq_bs, 0);
        }

        if constexpr (kExpOffset.y) {
          // A bf16 factor holds at most 2^127, so a larger offset takes a second factor.
          constexpr uint32_t kFirstExpOffset = MIN(kExpOffset.y, 127u);
          const scalar_t2 scale_factor = prepare_exp_scale_factor<scalar_t2, kFirstExpOffset>();
          scalar_t2 *dq_bs_scalar2_ptr = reinterpret_cast<scalar_t2 *>(dq_bs);

          PRAGMA_UNROLL
          for (uint32_t j = 0; j < 4; j++) {
            dq_bs_scalar2_ptr[j] = __hmul2(dq_bs_scalar2_ptr[j], scale_factor);
          }

          if constexpr (kExpOffset.y > kFirstExpOffset) {
            const scalar_t2 scale_factor2 = prepare_exp_scale_factor<scalar_t2, kExpOffset.y - kFirstExpOffset>();
            PRAGMA_UNROLL
            for (uint32_t j = 0; j < 4; j++) {
              dq_bs_scalar2_ptr[j] = __hmul2(dq_bs_scalar2_ptr[j], scale_factor2);
            }
          }
        }
      };
    };
  };

  CUDA_INLINE
  void may_apply_bs_and_zp_on_b(uint32_t *regs_b, uint32_t j, uint32_t buffer_id) {
    may_process_bs_before_apply_on_b(j, buffer_id);

    if constexpr (ElementA::kBits == 16 && kExpOffset.x) {
      uint32_t exp_val = kExpOffset.x + ((1 << (ElementA::kExponentBits - 1)) - 1);
      uint32_t scale_factor_uint = 0x00010001 * (exp_val << ElementA::kMantissaBits);
      scalar_t2 scale_factor = *reinterpret_cast<scalar_t2 *>(&scale_factor_uint);
      scalar_t2 *b_f16_ptr = reinterpret_cast<scalar_t2 *>(regs_b);

      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 4; i++) {
        b_f16_ptr[i] = __hmul2(b_f16_ptr[i], scale_factor);
      }
    }

    // apply bs and/or zp only when we use fp16/bf16 activation.
    if constexpr (ElementA::kBits == 16 && (kIsGroupWeightScale || kIsBlockWeightScale || kIsFpZeroPoint)) {

      scalar_t2 *b_f16_ptr = reinterpret_cast<scalar_t2 *>(regs_b);
      scalar_t2 bs_f16_ptr[2];
      scalar_t2 bzp_f16_ptr[2];

      if constexpr (kIsGroupWeightScale && ElementBS::kBits == 8 && !kUseFusedE8m0Scale && !kUseNativeDequantBS) {
        // The scale dequantization leaves each group of four scales in (0, 2, 1, 3) order.
        scalar_t *bs_half_ptr = reinterpret_cast<scalar_t *>(dq_bs);
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < 2; i++) {
          bs_f16_ptr[i] = this->num2num2(bs_half_ptr[j < 4 ? j / 2 * 4 + i * 2 + j % 2 : j * 2 + i]);
        }
      } else if constexpr (kIsGroupWeightScale) {
        scalar_t *bs_half_ptr = reinterpret_cast<scalar_t *>(ElementBS::kBits == 8 ? &dq_bs[j] : &bs[buffer_id][j]);
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < 2; i++) {
          bs_f16_ptr[i] = this->num2num2(bs_half_ptr[i]);
        }
      }

      if constexpr (kIsFpZeroPoint) {
        static_assert(kHasZeroPoint);
        static_assert(!ElementB::kIsSigned && ElementB::kIsIntegerType);
        uint32_t zp_buffer_id = kIsChannelWeightScale ? 0 : buffer_id;
        scalar_t2 bzp2_ptr = *reinterpret_cast<scalar_t2 *>(&zp[zp_buffer_id][j]);

        if constexpr (std::is_same<ElementA, BFloat16>::value && ElementB::kNumBits >= 7) {
          uint32_t exp_val = ((int32_t)kExpOffset.x - 133) + ((1 << (ElementA::kExponentBits - 1)) - 1);
          uint32_t scale_factor_uint = 0x00010001 * (exp_val << ElementA::kMantissaBits);
          scalar_t2 scale_factor = *reinterpret_cast<scalar_t2 *>(&scale_factor_uint);
          bzp2_ptr = __hmul2(bzp2_ptr, scale_factor);
        }

        scalar_t *bzp_ptr = reinterpret_cast<scalar_t *>(&bzp2_ptr);
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < 2; i++) {
          bzp_f16_ptr[i] = this->num2num2(bzp_ptr[i]);
        }
      }

      // Each warp apply bs or/and zp on a 16x16 weight block per time
      // The thread_id 0 process the following weight index of each block
      // [(0, 1), (0, 2)], [(0, 8), (0, 9)]
      // [(8, 1), (8, 2)], [(8, 8), (8, 9)]
      PRAGMA_UNROLL
      for (uint32_t i = 0; i < 2; i++) {
        PRAGMA_UNROLL
        for (uint32_t k = 0; k < 2; k++) {
          if constexpr (kIsFpZeroPoint) {
            scalar_t2 bzp_single = bzp_f16_ptr[kUseWgmma ? k : i];
            b_f16_ptr[i * 2 + k] = __hsub2(b_f16_ptr[i * 2 + k], bzp_single);
          }
          if constexpr (kIsGroupWeightScale) {
            scalar_t2 bs_single = bs_f16_ptr[kUseWgmma ? k : i];
            b_f16_ptr[i * 2 + k] = __hmul2(b_f16_ptr[i * 2 + k], bs_single);
          }
          if constexpr (kIsBlockWeightScale) {
            scalar_t2 bs_single = reinterpret_cast<scalar_t2 *>(dq_bs)[0];
            b_f16_ptr[i * 2 + k] = __hmul2(b_f16_ptr[i * 2 + k], bs_single);
          }
        };
      };
    };
  };

  CUDA_INLINE
  void may_process_as_and_bs_before_apply_on_c(uint32_t m, uint32_t n, uint32_t k, uint32_t iter_id) {
    uint32_t buffer_id = iter_id % 2;
    uint32_t k_index, is_last_iter;
    if constexpr (kUsePackedKLayout) {
      k_index = (k + 1) * MmaShape::K;
      is_last_iter = (k + 1 == kNumKSlabs);
    } else {
      k_index = iter_id * kPartMmaShapeK + (k + 1) * MmaShape::K;
      is_last_iter = k_index == WarpShape::K;
    }
    uint32_t is_as_group_end = kIsGroupInputScale && k_index % kInputScaleGroupSize == 0;
    constexpr bool kProcessGroupWeightScale = kIsGroupWeightScale && !kUseFusedE8m0Scale;
    constexpr bool kProcessBlockWeightScale = kIsBlockWeightScale && !kUseFusedE8m0Scale;
    uint32_t is_bs_group_end = (kProcessGroupWeightScale || kProcessBlockWeightScale) && k_index % kWeightScaleGroupSize == 0;

    bool should_apply_as = kIsGroupInputScale && (is_last_iter || is_as_group_end);
    bool should_apply_bs = (kProcessGroupWeightScale || kProcessBlockWeightScale) && (is_last_iter || is_bs_group_end);
    if (!should_apply_as && !should_apply_bs) return;

    if (m == 0 && n == 0 && should_apply_as) {
      if constexpr (kIsBlockWeightScale) {
        float *as_float_ptr = reinterpret_cast<float *>(as[buffer_id]);
        float *q_as_float_ptr = reinterpret_cast<float *>(q_as);
        float &block_bs_float = reinterpret_cast<float *>(&bs[buffer_id])[0];
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < WarpShape::M / 8 * (kUseWgmma ? 2 : 1); i++) {
          as_float_ptr[i] = as_float_ptr[i] * block_bs_float;
        }
      }

      if constexpr (kIsF16Accum) {
        float2 *as_float2_ptr = reinterpret_cast<float2 *>(as[buffer_id]);
        float *as_float_ptr = reinterpret_cast<float *>(as[buffer_id]);
        scalar_t2 *as_scalar2_ptr = reinterpret_cast<scalar_t2 *>(as[buffer_id]);
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < WarpShape::M / 8; i++) {
          if constexpr (kUseWgmma) {
            as_scalar2_ptr[i] = this->float22num2(as_float2_ptr[i]);
          } else {
            as_scalar2_ptr[i] = this->float2num2(as_float_ptr[i]);
          }

          if constexpr (kExpOffset.y && !kIsGroupWeightScale) {
            scalar_t2 scale_factor = prepare_exp_scale_factor<scalar_t2, kExpOffset.y>();
            as_scalar2_ptr[i] = __hmul2(as_scalar2_ptr[i], scale_factor);
          }
        };
      };
    }

    if (m == 0 && n == 0 && should_apply_bs) {
      if constexpr (kUseIntWeightScale) {
        int16_t *bs_vals = reinterpret_cast<int16_t *>(bs[buffer_id]);
        int32_t *dq_bs_vals = reinterpret_cast<int32_t *>(dq_bs);
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kNumBSPerGroup; i++)
          dq_bs_vals[i] = (int32_t)bs_vals[i];
      } else if constexpr (kIsF16Accum && ElementBS::kBits == 16) {
        scalar_t2 *dq_bs_scalar2_ptr = reinterpret_cast<scalar_t2 *>(dq_bs);
        scalar_t2 *bs_scalar2_ptr = reinterpret_cast<scalar_t2 *>(bs[buffer_id]);

        if constexpr (kExpOffset.y) {
          scalar_t2 scale_factor = prepare_exp_scale_factor<scalar_t2, kExpOffset.y>();
          PRAGMA_UNROLL
          for (uint32_t i = 0; i < kNumBSPerGroup / 2; i++)
            dq_bs_scalar2_ptr[i] = __hmul2(bs_scalar2_ptr[i], scale_factor);
        }
      } else if constexpr (kIsF16Accum && kIsBlockWeightScale) {
        scalar_t2 *dq_bs_scalar2_ptr = reinterpret_cast<scalar_t2 *>(dq_bs);
        scalar_t2 scale_factor = prepare_exp_scale_factor<scalar_t2, kExpOffset.y>();
        dq_bs_scalar2_ptr[0] = this->float2num2(reinterpret_cast<float *>(bs[buffer_id])[0]);
        dq_bs_scalar2_ptr[0] = __hmul2(dq_bs_scalar2_ptr[0], scale_factor);
      } else if constexpr (kIsF16Accum && ElementBS::kBits == 8 && kIsGroupWeightScale) {
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < CEIL_DIV(kNumBSPerGroup, 8); i++) {
          dequant<ElementBS, ElementC>(bs[buffer_id], dq_bs, i);

          scalar_t *dq_bs_scalar_ptr = reinterpret_cast<scalar_t *>(dq_bs);
          PRAGMA_UNROLL
          for (uint32_t j = 0; j < 2; j++) {
            scalar_t tmp = dq_bs_scalar_ptr[2 + 4 * j];
            dq_bs_scalar_ptr[2 + 4 * j] = dq_bs_scalar_ptr[1 + 4 * j];
            dq_bs_scalar_ptr[1 + 4 * j] = tmp;
          }
        }

        scalar_t2 *dq_bs_scalar2_ptr = reinterpret_cast<scalar_t2 *>(dq_bs);
        constexpr uint32_t exp_offset = get_dtype_dequant_exp_offset<ElementC, ElementBS>();
        scalar_t2 scale_factor = prepare_exp_scale_factor<scalar_t2, exp_offset>();
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kNumBSPerGroup / 2; i++)
          dq_bs_scalar2_ptr[i] = __hmul2(dq_bs_scalar2_ptr[i], scale_factor);

        if constexpr (kExpOffset.y) {
          scalar_t2 scale_factor = prepare_exp_scale_factor<scalar_t2, kExpOffset.y>();
          PRAGMA_UNROLL
          for (uint32_t i = 0; i < kNumBSPerGroup / 2; i++)
            dq_bs_scalar2_ptr[i] = __hmul2(dq_bs_scalar2_ptr[i], scale_factor);
        }

      } else if constexpr (!kIsF16Accum && ElementBS::kBits == 8) {
        using F8x4 = typename F8Conversion<ElementBS>::scalar_t4;
        F8x4 *bs_vals = reinterpret_cast<F8x4 *>(bs[buffer_id]);
        float4 *dq_bs_vals = reinterpret_cast<float4 *>(dq_bs);

        constexpr uint32_t kFullPackets = kNumBSPerGroup / 4;
        constexpr uint32_t kTailScales = kNumBSPerGroup % 4;

        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kFullPackets; i++)
          dq_bs_vals[i] = F8Conversion<ElementBS>::num42float4(bs_vals[i]);

        if constexpr (kTailScales != 0) {
          uint32_t packed = 0;
          const uint8_t *src = reinterpret_cast<const uint8_t *>(bs[buffer_id]);
          uint8_t *dst = reinterpret_cast<uint8_t *>(&packed);

          PRAGMA_UNROLL
          for (uint32_t i = 0; i < kTailScales; i++)
            dst[i] = src[kFullPackets * 4 + i];

          dq_bs_vals[kFullPackets] =
              F8Conversion<ElementBS>::num42float4(*reinterpret_cast<F8x4 *>(&packed));
        }
      } else if constexpr (!kIsF16Accum && ElementBS::kBits == 16) {
        using F16x2 = typename F16Conversion<ElementBS>::scalar_t2;
        F16x2 *bs_vals = reinterpret_cast<F16x2 *>(bs[buffer_id]);
        float2 *dq_bs_vals = reinterpret_cast<float2 *>(dq_bs);
        PRAGMA_UNROLL
        for (uint32_t i = 0; i < kNumBSPerGroup / 2; i++)
          dq_bs_vals[i] = F16Conversion<ElementBS>::num22float2(bs_vals[i]);
      }
    }
  };

  CUDA_INLINE
  void may_apply_as_and_bs_on_mma_c(uint32_t *regs_c_ptr, uint32_t m, uint32_t n, uint32_t k, uint32_t iter_id) {
    if constexpr (ElementA::kBits == 16) return;
    if constexpr (!kIsGroupInputScale && !kIsGroupWeightScale && !kIsBlockWeightScale) return;
    if constexpr (kUseWgmma) return;
    constexpr bool kApplyGroupWeightScaleOnC = kIsGroupWeightScale && !kUseFusedE8m0Scale;
    constexpr bool kApplyBlockWeightScaleOnC = kIsBlockWeightScale && !kUseFusedE8m0Scale;
    constexpr bool kApplyGroupInputScaleOnC = kIsGroupInputScale;
    if constexpr (kUseFusedE8m0Scale && !kApplyGroupInputScaleOnC) return;

    may_process_as_and_bs_before_apply_on_c(m, n, k, iter_id);

    uint32_t buffer_id = iter_id % 2;
    uint32_t k_index = iter_id * kPartMmaShapeK + (k + 1) * MmaShape::K;
    uint32_t is_last_iter = k_index == WarpShape::K;
    uint32_t is_as_group_end = kApplyGroupInputScaleOnC && k_index % kInputScaleGroupSize == 0;
    uint32_t is_bs_group_end = (kApplyGroupWeightScaleOnC || kApplyBlockWeightScaleOnC) && k_index % kWeightScaleGroupSize == 0;

    bool should_apply_as = kApplyGroupInputScaleOnC && (is_last_iter || is_as_group_end);
    bool should_apply_bs = (kApplyGroupWeightScaleOnC || kApplyBlockWeightScaleOnC) && (is_last_iter || is_bs_group_end);
    if (!should_apply_as && !should_apply_bs) return;

    using CRegistersArrayType = typename MmaOpClass::CRegisters[2][WarpShape::M / MmaShape::M][WarpShape::N / MmaShape::N];
    auto &regs_c = *reinterpret_cast<CRegistersArrayType *>(regs_c_ptr);

    PRAGMA_UNROLL
    for (uint32_t index = 0; index < MmaShape::M * MmaShape::N / 64; index++) {
      // PPU's accumulator fragments enumerate N before M.
      uint32_t inner_m = USE_PPU ? index / (MmaShape::N / 8) : index % (MmaShape::M / 8);
      uint32_t inner_n = USE_PPU ? index % (MmaShape::N / 8) : index / (MmaShape::M / 8);

      if constexpr (kIsF16Accum) {
        scalar_t2 &part_regs_c0 = reinterpret_cast<scalar_t2 *>(regs_c[0][m][n])[index];
        scalar_t2 &part_regs_c1 = reinterpret_cast<scalar_t2 *>(regs_c[1][m][n])[index];

        scalar_t2 as_val = reinterpret_cast<scalar_t2 *>(as[buffer_id])[m * MmaShape::M / 8 + inner_m];
        scalar_t2 bs_val;
        if constexpr (kApplyBlockWeightScaleOnC) {
          bs_val = reinterpret_cast<scalar_t2 *>(dq_bs)[0];
        } else if constexpr (kApplyGroupWeightScaleOnC) {
          uint32_t scale_pair = n * MmaShape::N / 8 + inner_n;
          const uint32_t *scale_regs = ElementBS::kBits == 8 || kExpOffset.y ? dq_bs : bs[buffer_id];
          if constexpr (USE_PPU) {
            bs_val = reinterpret_cast<const scalar_t2 *>(scale_regs)[scale_pair];
          } else {
            const scalar_t *scale_values = reinterpret_cast<const scalar_t *>(scale_regs);
            bs_val = __halves2half2(scale_values[scale_pair], scale_values[scale_pair + kNumBSPerGroup / 2]);
          }
        }

        if constexpr (kApplyGroupInputScaleOnC && kApplyGroupWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, __hmul2(as_val, bs_val), part_regs_c1);
        } else if constexpr (!kApplyGroupInputScaleOnC && kApplyGroupWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, bs_val, part_regs_c1);
        } else if constexpr (!kApplyGroupInputScaleOnC && kApplyBlockWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, bs_val, part_regs_c1);
        } else if constexpr (kApplyGroupInputScaleOnC && !kApplyGroupWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, as_val, part_regs_c1);
        }
        reinterpret_cast<uint32_t *>(&part_regs_c0)[0] = 0;

      } else {
        int2 &part_int_regs_c0 = reinterpret_cast<int2 *>(regs_c[0][m][n])[index];
        int2 &part_int_regs_c1 = reinterpret_cast<int2 *>(regs_c[1][m][n])[index];
        float2 &part_regs_c0 = reinterpret_cast<float2 *>(regs_c[0][m][n])[index];
        float2 &part_regs_c1 = reinterpret_cast<float2 *>(regs_c[1][m][n])[index];
        if constexpr (kUseIntWeightScale) {
          static_assert(std::is_same<ValTypeC, int32_t>::value);
          static_assert(!kIsGroupInputScale);
          static_assert(kIsGroupWeightScale);

          uint32_t scale_pair = n * MmaShape::N / 8 + inner_n;
          int2 scale_values;
          if constexpr (USE_PPU) {
            scale_values = reinterpret_cast<const int2 *>(dq_bs)[scale_pair];
          } else {
            const int32_t *scales = reinterpret_cast<const int32_t *>(dq_bs);
            scale_values = {scales[scale_pair], scales[scale_pair + kNumBSPerGroup / 2]};
          }
          part_int_regs_c1.x += scale_values.x * part_int_regs_c0.x;
          part_int_regs_c1.y += scale_values.y * part_int_regs_c0.y;

          part_int_regs_c0.x = 0;
          part_int_regs_c0.y = 0;
        } else {
          if constexpr (std::is_same<ValTypeC, int32_t>::value) {
            part_regs_c0.x = __int2float_rn(part_int_regs_c0.x);
            part_regs_c0.y = __int2float_rn(part_int_regs_c0.y);
          }

          float &as_val = reinterpret_cast<float *>(as[buffer_id])[m * MmaShape::M / 8 + inner_m];
          if constexpr (kApplyGroupWeightScaleOnC) {
            uint32_t scale_pair = n * MmaShape::N / 8 + inner_n;
            float2 scale_values;
            if constexpr (USE_PPU) {
              scale_values = reinterpret_cast<const float2 *>(dq_bs)[scale_pair];
            } else {
              const float *scales = reinterpret_cast<const float *>(dq_bs);
              scale_values = {scales[scale_pair], scales[scale_pair + kNumBSPerGroup / 2]};
            }
            if constexpr (kApplyGroupInputScaleOnC) {
              part_regs_c1.x += as_val * scale_values.x * part_regs_c0.x;
              part_regs_c1.y += as_val * scale_values.y * part_regs_c0.y;
            } else {
              part_regs_c1.x += scale_values.x * part_regs_c0.x;
              part_regs_c1.y += scale_values.y * part_regs_c0.y;
            }
          } else if constexpr (!kApplyGroupInputScaleOnC && kApplyBlockWeightScaleOnC) {
            float &block_bs_float = reinterpret_cast<float *>(&bs[buffer_id])[0];
            part_regs_c1.x += block_bs_float * part_regs_c0.x;
            part_regs_c1.y += block_bs_float * part_regs_c0.y;
          } else if constexpr (kApplyGroupInputScaleOnC && !kApplyGroupWeightScaleOnC) {
            part_regs_c1.x += as_val * part_regs_c0.x;
            part_regs_c1.y += as_val * part_regs_c0.y;
          }

          part_regs_c0.x = 0;
          part_regs_c0.y = 0;
        }
      }
    }
  }

  CUDA_INLINE
  void may_apply_as_and_bs_on_wgmma_c(uint32_t *regs_c_ptr, uint32_t m, uint32_t k, uint32_t iter_id, uint32_t delta_m) {
    if constexpr (ElementA::kBits == 16) return;
    if constexpr (!kIsGroupInputScale && !kIsGroupWeightScale && !kIsBlockWeightScale) return;
    if constexpr (!kUseWgmma) return;
    constexpr bool kApplyGroupWeightScaleOnC = kIsGroupWeightScale && !kUseFusedE8m0Scale;
    constexpr bool kApplyBlockWeightScaleOnC = kIsBlockWeightScale && !kUseFusedE8m0Scale;
    constexpr bool kApplyGroupInputScaleOnC = kIsGroupInputScale;
    if constexpr (kUseFusedE8m0Scale && !kApplyGroupInputScaleOnC) return;

    if constexpr (!(Ctx::kUseWgmmaSsNLayout && kIsBlockWeightScale))
      may_process_as_and_bs_before_apply_on_c(m, 0, k, iter_id);

    uint32_t buffer_id = iter_id % 2;
    uint32_t k_index, is_last_iter;
    if constexpr (kUsePackedKLayout) {
      k_index = (k + 1) * MmaShape::K;
      is_last_iter = (k + 1 == kNumKSlabs);
    } else {
      k_index = iter_id * kPartMmaShapeK + (k + 1) * MmaShape::K;
      is_last_iter = k_index == WarpShape::K;
    }
    uint32_t is_as_group_end = kApplyGroupInputScaleOnC && k_index % kInputScaleGroupSize == 0;
    uint32_t is_bs_group_end = (kApplyGroupWeightScaleOnC || kApplyBlockWeightScaleOnC) && k_index % kWeightScaleGroupSize == 0;

    bool should_apply_as = kApplyGroupInputScaleOnC && (is_last_iter || is_as_group_end);
    bool should_apply_bs = (kApplyGroupWeightScaleOnC || kApplyBlockWeightScaleOnC) && (is_last_iter || is_bs_group_end);
    if (!should_apply_as && !should_apply_bs) return;

    using CRegistersArrayType = typename MmaOpClass::CRegisters[2][WarpShape::N * 4 / MmaShape::N][WarpShape::M / MmaShape::M];
    auto &regs_c = *reinterpret_cast<CRegistersArrayType *>(regs_c_ptr);

    PRAGMA_UNROLL
    for (uint32_t index = 0; index < MmaShape::M * 16 / 64; index++) {
      uint32_t inner_n = index / 2;
      uint32_t inner_m = index % 2;

      if constexpr (Ctx::kUseWgmmaSsNLayout && kIsBlockWeightScale) {
        // SS fragments can cross N scale blocks within one warp. Keep each
        // fragment's scale separate instead of folding one scale into AS.
        float weight_scale = reinterpret_cast<float *>(bs[buffer_id])[m];
        float2 input_scale = {1.0f, 1.0f};
        if constexpr (kApplyGroupInputScaleOnC)
          input_scale = reinterpret_cast<float2 *>(as[buffer_id])[inner_n];
        float2 scale = {input_scale.x * weight_scale, input_scale.y * weight_scale};
        if constexpr (kIsF16Accum) {
          scalar_t2 partial = reinterpret_cast<scalar_t2 *>(regs_c[0][m][0])[index];
          scalar_t2 &final = reinterpret_cast<scalar_t2 *>(regs_c[1][m][0])[index];
          final = __hfma2(partial, this->float22num2(scale), final);
        } else {
          float2 partial;
          if constexpr (std::is_same<ValTypeC, int32_t>::value) {
            int2 values = reinterpret_cast<int2 *>(regs_c[0][m][0])[index];
            partial = {__int2float_rn(values.x), __int2float_rn(values.y)};
          } else partial = reinterpret_cast<float2 *>(regs_c[0][m][0])[index];
          float2 &final = reinterpret_cast<float2 *>(regs_c[1][m][0])[index];
          final.x += partial.x * scale.x;
          final.y += partial.y * scale.y;
        }
      } else if constexpr (kIsF16Accum) {
        scalar_t2 &part_regs_c0 = reinterpret_cast<scalar_t2 *>(regs_c[0][m][0])[index];
        scalar_t2 &part_regs_c1 = reinterpret_cast<scalar_t2 *>(regs_c[1][delta_m + m][0])[index];

        scalar_t2 as_val = reinterpret_cast<scalar_t2 *>(as[buffer_id])[inner_n];
        scalar_t2 bs_val;
        if constexpr (!kIsGroupWeightScale && kIsBlockWeightScale) {
          bs_val = reinterpret_cast<scalar_t2 *>(dq_bs)[0];
        } else if constexpr (ElementBS::kBits == 8 || kExpOffset.y) {
          bs_val = this->num2num2(reinterpret_cast<scalar_t *>(dq_bs)[(delta_m + m) * 2 + inner_m]);
        } else {
          bs_val = this->num2num2(reinterpret_cast<scalar_t *>(bs[buffer_id])[(delta_m + m) * 2 + inner_m]);
        }

        float &block_bs_float = reinterpret_cast<float *>(&bs[buffer_id])[0];

        if constexpr (kApplyGroupInputScaleOnC && kApplyGroupWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, __hmul2(as_val, bs_val), part_regs_c1);
        } else if constexpr (!kApplyGroupInputScaleOnC && kApplyGroupWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, bs_val, part_regs_c1);
        } else if constexpr (!kApplyGroupInputScaleOnC && kApplyBlockWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, bs_val, part_regs_c1);
        } else if constexpr (kApplyGroupInputScaleOnC && !kApplyGroupWeightScaleOnC) {
          part_regs_c1 = __hfma2(part_regs_c0, as_val, part_regs_c1);
        }
      } else {
        int2 &part_int_regs_c0 = reinterpret_cast<int2 *>(regs_c[0][m][0])[index];
        int2 &part_int_regs_c1 = reinterpret_cast<int2 *>(regs_c[1][delta_m + m][0])[index];
        float2 &part_regs_c0 = reinterpret_cast<float2 *>(regs_c[0][m][0])[index];
        float2 &part_regs_c1 = reinterpret_cast<float2 *>(regs_c[1][delta_m + m][0])[index];

        if constexpr (kUseIntWeightScale) {
          static_assert(std::is_same<ValTypeC, int32_t>::value);
          static_assert(!kIsGroupInputScale);
          static_assert(kIsGroupWeightScale);

          int32_t bs_val = reinterpret_cast<int32_t *>(dq_bs)[(delta_m + m) * 2 + inner_m];
          part_int_regs_c1.x += bs_val * part_int_regs_c0.x;
          part_int_regs_c1.y += bs_val * part_int_regs_c0.y;
        } else {
          if constexpr (std::is_same<ValTypeC, int32_t>::value) {
            part_regs_c0.x = __int2float_rn(part_int_regs_c0.x);
            part_regs_c0.y = __int2float_rn(part_int_regs_c0.y);
          }

          float2 &as_vals = *reinterpret_cast<float2 *>(&as[buffer_id][inner_n * 2]);
          float &bs_val = reinterpret_cast<float *>(dq_bs)[(delta_m + m) * 2 + inner_m];
          float &block_bs_float = reinterpret_cast<float *>(&bs[buffer_id])[0];

          if constexpr (kApplyGroupInputScaleOnC && kApplyGroupWeightScaleOnC) {
            part_regs_c1.x += as_vals.x * bs_val * part_regs_c0.x;
            part_regs_c1.y += as_vals.y * bs_val * part_regs_c0.y;
          } else if constexpr (!kApplyGroupInputScaleOnC && kApplyGroupWeightScaleOnC) {
            part_regs_c1.x += bs_val * part_regs_c0.x;
            part_regs_c1.y += bs_val * part_regs_c0.y;
          } else if constexpr (!kApplyGroupInputScaleOnC && kApplyBlockWeightScaleOnC) {
            part_regs_c1.x += block_bs_float * part_regs_c0.x;
            part_regs_c1.y += block_bs_float * part_regs_c0.y;
          } else if constexpr (kApplyGroupInputScaleOnC && !kApplyGroupWeightScaleOnC) {
            part_regs_c1.x += as_vals.x * part_regs_c0.x;
            part_regs_c1.y += as_vals.y * part_regs_c0.y;
          }
        }
      }
    }
  }

  template <class T = uint32_t>
  CUDA_INLINE T *regs_bs_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<T *>(bs[buffer_id]);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_as_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<T *>(as[buffer_id]);
  };

  template <class T = uint32_t>
  CUDA_INLINE T *regs_zp_as_ptr(uint32_t buffer_id) {
    return reinterpret_cast<T *>(zp[kIsChannelWeightScale ? 0 : buffer_id]);
  };
};
