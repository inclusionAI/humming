# HummingKernel Configuration

HummingKernel configurations are divided into three categories:

- **LayerConfig**: Parameters that affect weight layout, data types, and shapes.
- **ComputeConfig**: Parameters that do not directly affect weights but significantly impact kernel behavior or computation precision.
- **TuningConfig**: Parameters that only affect performance.

## LayerConfig

| Parameter | Description |
|-----------|-------------|
| `a_dtype`, `b_dtype` | Activation and weight data types. See the project README for supported combinations. |
| `c_dtype` | Output matrix data type. Only `float16` and `bfloat16` are supported. |
| `bs_dtype` | Weight scale data type. Supports `float16` / `bfloat16` / `float8e8m0` / `float8e4m3` / `float8e5m2`. |
| `shape_n`, `shape_k` | The N and K dimensions of the GEMM after padding. |
| `pad_shape_n`, `pad_shape_k` | Humming pads the weight matrix to a suitable shape (e.g., `shape_n` is typically padded to a multiple of 256, `shape_k` to a multiple of 128). These parameters specify the size of the padded portion, i.e., the actual effective weight shape is `shape_n - pad_shape_n` and `shape_k - pad_shape_k`. Note that the last dimension of input and output matrices should match the unpadded shape. |
| `num_experts` | Number of experts for MoE. Set to `0` or `None` for non-MoE. |
| `input_scale_group_size` | Group size for activation quantization. Not applicable when using FP16/BF16. Must be a power of 2 and greater than the minimum group size requirement for the activation type. Set to `0` for channelwise/tokenwise quantization. |
| `weight_scale_type` | Supports several modes: `group`, `channel`, `block`, `tensor`, `group_tensor`. `group_tensor` means both groupwise scale and tensorwise scale (global scale) are present. |
| `weight_scale_group_size` | For groupwise or blockwise, this specifies the quantization group size along the K dimension. Ignored for channelwise or tensorwise. |
| `weight_scale_group_size_n` | Only used for blockwise quantization. Specifies the quantization group size along the N dimension. |
| `use_int_weight_scale` | Whether to use integer-type scale. Only applicable for INT8 or INT4 activations with `weight_scale_group_size > 0`. Used to accelerate computation in certain cases. The weight scale must be preprocessed as follows: |
| `has_zero_point` | Whether to enable zero point. When enabled, the dequantization changes from `x * scale` to `(x - zp) * scale`. Humming supports two zero point types (see below). |
| `is_fp_zero_point` | Whether to use FP-type zero point. See `has_zero_point` for details. |
| `has_bias` | Whether to use fused bias addition. |
| `mma_type` | Can be `mma`, `wgmma`, `tcgen05`, or `mxmma`. This selects the weight layout and preferred tensor-core backend. |

On SM100-family GPUs, BF16 activations and outputs with CUDA 12.9 or newer default to
`tcgen05`. Its kernels compile for `sm_100f` and share the existing `mma` packed
weights, scales, and zero points. The per-shape heuristic uses MMA for decode,
thin expert batches, and shapes without a measured TCGen5 advantage. Grouped-masked
GEMMs retain MMA by default because their M dimension can represent buffer capacity. Setting
`mma_type="mma"` retains the existing MMA heuristics. Explicit tuning can select
`mma_type="tcgen05"` with a supported TCGen5 geometry. Automatic TCGen5 tuning
initially covers `uint4`, `uint8`, `float4e2m1`, `float8e4m3`, and `float8e5m2`;
other supported weights retain MMA by default.

TCGen5 dequantizes the shared packed weights into two TMEM operand buffers while
reusing the existing epilogue. Populated M128 tiles with K >= 1024 can use two
CTAs per SM when their shared-memory allocations fit; smaller workloads retain
one CTA. The heuristic uses five pipeline stages, or four for uint8 M128 and
for eight-bit weights with two CTAs. Indexed FP4 also uses four stages with two
CTAs.

**`use_int_weight_scale` preprocessing:**

```python
dtype = weight_scale.dtype
assert dtype in [torch.bfloat16, torch.float16]
weight_scale = (weight_scale / weight_scale.max() * 2048).round()
weight_scale = weight_scale.to(torch.int16).view(dtype)
```

**Zero point types (`has_zero_point`):**

- **INT type**: Only supports INT-type quantized weights, with the same bit width as the quantization bit width.
- **FP type**: FP16/BF16 type, only supported when using FP16/BF16 as the activation type.

## ComputeConfig

| Parameter | Description |
|-----------|-------------|
| `gemm_type` | Supports `dense`, `indexed`, `grouped_contiguous`, `grouped_masked`. |
| `use_f16_accum` | Whether to use FP16 accumulator for MMA. Applicable when activation type is `fp16` / `float8e4m3` and output type is `float16`. |
| `use_batch_invariant` | Whether to enable batch invariance support. |

## TuningConfig

For indexed GEMMs, `sorted_ids` and `expert_ids` must be aligned to the selected
`block_shape_m`. Gate/up and down projections can select different tile heights;
reuse their routing tensors only when those heights match.

### Block and Warp Shapes

`block_shape` and `warp_shape` are 3D tuples representing the M/N/K dimensions, with the following constraints:

- `block_shape[i]` must be a power-of-2 multiple of `warp_shape[i]`.
- `block_shape_n` must be at least 64.
- When using WGMMA, `block_shape_n` must be at least 4x `warp_shape_n`.
- TCGen5 uses `block_shape=(64 or 128, 128, 64)` and
  `warp_shape=(block_shape_m, 32, 64)`, with warp specialization, at least three
  stages, and FP32 accumulation. PDL, multicast, and reduction overlap are not
  supported by this schedule.
- `warp_shape_m` must be a multiple of MMA shape M.
- Valid values for `warp_shape_n` and `warp_shape_k` depend on the activation type:

| Activation Type | `warp_shape_n` | `warp_shape_k` |
|----------------|----------------|----------------|
| `float16` / `bfloat16` | 32, 64 | 32, 64 |
| `float8e4m3` / `float8e5m2` / `int8` | 16, 32, 64 | 64, 128 |
| `float4e2m1` / `int4` | 16, 32, 64 | 128, 256 |

### Pipeline and Synchronization

| Parameter | Description |
|-----------|-------------|
| `num_stages` | Number of pipeline stages. Must be at least 2. When using `use_warp_spec` with WGMMA, must be at least 3. |
| `use_warp_spec` | Whether to enable Warp Specialization. Requires SM90+. |
| `use_mbarrier` | Whether to use MBarrier. Requires SM80+. |
| `use_cp_async` | Whether to use CP Async. Requires SM80+. |
| `num_ctas_per_sm` | Number of CTAs (Cooperative Thread Arrays / Thread Blocks) launched per SM. |
| `num_write_splits` | Whether to split result writes into batches. Only supports 1 or 2. Primarily used on SM75 and other devices with limited shared memory to reduce shared memory usage during the reduce phase. Requires `block_shape_m == warp_shape_m`. |

### TMA (Tensor Memory Accelerator)

| Parameter | Description |
|-----------|-------------|
| `use_tma` | Whether to use TMA. Requires SM90+. When set to `True`, all parameters use TMA by default. Fine-grained control is available via the parameters below. |
| `use_tma_a` | Enable TMA for matrix A loading. |
| `use_tma_b` | Enable TMA for matrix B loading. |
| `use_tma_c` | Enable TMA for output matrix storing. |
| `use_tma_bs` | Enable TMA for weight scale loading. |
| `use_tma_bzp` | Enable TMA for zero point loading. |
| `use_tma_bias` | Enable TMA for bias loading. |
| `multi_cast_size_a` | When greater than 1, enables TMA MultiCast for matrix A. Currently only supports Dense GEMM. Only one of `multi_cast_size_a` and `multi_cast_size_b` can be greater than 1. |
| `multi_cast_size_b` | When greater than 1, enables TMA MultiCast for matrix B. Only one of `multi_cast_size_a` and `multi_cast_size_b` can be greater than 1. |
