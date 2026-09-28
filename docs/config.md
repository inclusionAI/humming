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
| `mma_type` | Can be `mma`, `wgmma`, `umma`, or `mxmma`. This selects the weight layout and preferred tensor-core backend. |

`umma` requires SM100-family GPUs, CUDA 12.9+, and FP16/BF16 inputs/outputs with FP32
accumulation. It shares the `mma` weight layout; tuning selects the backend per shape.

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

### Block and Warp Shapes

`block_shape` and `warp_shape` are 3D tuples representing the M/N/K dimensions, with the following constraints:

- `block_shape[i]` must be a power-of-2 multiple of `warp_shape[i]`.
- `block_shape_n` must be at least 64.
- When using WGMMA, `block_shape_n` must be at least 4x `warp_shape_n`.
- When using UMMA, block M/K must equal warp M/K, warp N is 32, M is a multiple of 8 in [8, 256], and K is a power of two of at least 32. Block N can be 64, 128, 256, or 512; the tile must fit SMEM and TMEM.
- For indexed GEMMs, align `sorted_ids` and `expert_ids` to each projection's `block_shape_m`.
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
| `num_stages` | Number of pipeline stages. Must be at least 2. Must be at least 3 when using `use_warp_spec` with WGMMA. |
| `use_warp_spec` | Whether to enable Warp Specialization. Requires SM90+. Required for UMMA. |
| `use_mbarrier` | Whether to use MBarrier. Requires SM80+. |
| `use_cp_async` | Whether to use CP Async. Requires SM80+. |
| `num_ctas_per_sm` | Number of CTAs (Cooperative Thread Arrays / Thread Blocks) launched per SM. |
| `umma_cta_group_size` | `1` (default) or `2`. With `2`, a cluster of two CTAs cooperatively executes UMMA for adjacent N tiles. This is independent of CTA residency and TMA multicast. |
| `umma_output_chunk_rows` | `0` (default) writes a full output tile. `32` uses two alternating 32-row shared-memory buffers and issues TMA stores as each chunk becomes ready. |
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

### UMMA pipeline and cooperative output

Both one-CTA and two-CTA execution use the same continuous stage ring and three
warp groups per CTA. WG0 contains two loading warps, an issuing warp (active only
in the leader CTA for cooperative execution), and an activation readiness warp.
For cp.async activation tiles of at least 12 KiB, WG0 instead uses three loading
warps and one issuing warp. Dequantization's combined load barrier supplies A
readiness in this case. The choice depends on bytes per tile, not token count.
WG1 writes output; WG2 converts the weights. Accumulator ready/free barriers
separate issuing from output. Indexed loading retires the preceding tile before
reusing its row-index buffer. With two CTAs, each loads half of A and its own N
tile of B. Both CTAs retain the existing compressed weight
layout and register-to-TMEM conversion. MMA completion releases operands in both
CTAs through multicast barrier commits; this does not enable TMA multicast.

Chunked output currently requires dense GEMM, TMA output, separate
output storage (`smem_reuse_mode="none"`), block N=128, and block M divisible by
32. Two-CTA execution additionally requires chunked output, N divisible by 256,
TMA stage loads, and `num_ctas_per_sm=1`. Activation scales are not supported in
this pipeline. Channel weight scales, channel secondary scales, bias, and
channel/group zero points reuse the existing loaders and output arithmetic.
Channel parameters are released once all consuming threads have read them into
registers, allowing the next tile's channel loads to overlap output. Chunked
output supports Stream-K for both one- and two-CTA execution: the first slice
stores each chunk, later slices use TMA reduce-add, and partial writes complete
before releasing the output lock. Bias is applied only by the first slice.

SM100 dense heuristics select two CTAs with six stages when the tile is suitable,
K is long enough to amortize the pipeline, and the estimated shared-memory
allocation fits. The existing Stream-K decision is preserved for CTA pairs.
Without Stream-K, underfilled output waves retain single-CTA execution. Chunked
output remains opt-in for single-CTA execution because it did not improve the
measured large dense cases by itself.

### SM100 MoE tile selection

UMMA MoE selection samples expert row counts from total routed rows (including
top-k), the expert count, and the configured probability CV (default 0.25).
It scores M/N tiles with Stream-K already included, balancing scheduled work
against padded rows. A small fixed per-tile cost accounts for activation loading
and synchronization shared by wider N tiles. Stream-K must predict at least a
50% reduction in work, including its startup allowance, before it is selected.
The candidate pipeline has at least three stages unless K has fewer than three
iterations. These are conservative heuristic choices, not kernel restrictions;
explicit configurations may still use two stages. No token-specific or
weight-dtype-specific tuning cases are used in this rule.
