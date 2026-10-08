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
| `bs_dtype` | Weight scale data type. Supports `float16` / `bfloat16` / `float8e8m0` / `float8e4m3` / `float8e5m2` / `float8e5m3`. |
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
| `use_fused_e8m0_scale` | Fuse E8M0 group scales into MXFP4-to-FP8/INT8 weight conversion. Weight preprocessing extracts a secondary scale. |
| `use_packed_k_layout` | Pack K slabs for WGMMA with 8-bit activations and even-bit weights. Can be explicitly enabled together with `use_fused_e8m0_scale`; transformed weights must use the same setting as the kernel. |

Ordinary E4M3/E5M2 scales use the software dequantization path before SM89 and
native `cvt` conversions on SM89 and newer (via FP16 when needed).
Unsigned E5M3 scales are stored as `torch.uint8` and use software dequantization.
Their exponent bits retain the existing conversion semantics: exponent 31 is
finite in BF16/FP32 and maps to Inf/NaN in FP16.

For ordinary FP16 layers, E8M0 scales are rebased from bias 127 to bias 15 and
packed as E5M0 in the high five bits of a byte; the low three bits are zero and
ignored. Scales within `[2^-14, 2^15]` need no extra secondary scale. Weight
preprocessing extracts a tensorwise `weight_scale_2` only when this range is
exceeded, merging it with an existing secondary scale. When both ends cannot
fit, it preserves the largest scales and lets the smallest scales underflow.
Native block-scaled MMA and fused E8M0 weight paths retain their own formats.

Weight preprocessing is independent of the tuning backend. `use_block_scaled_mma`
and `use_native_dequant` are derived from the architecture, data types, quantization
parameters, and (for native dequantization) compiler support.

Equal-bit-width A/B operands retain K-contiguous packed B rows, with padding and
integer encoding conversion where needed. MMA/MXMMA load these rows into swizzled
shared memory and use `ldmatrix`; WGMMA and UMMA use SS operands. Native E2M1,
E3M2, and E2M3 weights paired with FP8 inputs on SM10x/SM11x automatically use raw rows when
the data types and quantization parameters support UMMA. These weights require
the UMMA SS TMA expansion path. Other mixed-width operands use repacking.
`use_umma_ss` is now derived by the kernel and is not a layer configuration option.
UMMA SS requires `use_tma_b=True`.

Ordinary mixed-width weights use the same MMA repack layout across backends,
including SM90. WGMMA adapts the register order during dequantization, including
integer zero points. Weight transformation no longer takes `use_wgmma`.
Re-transform ordinary weights previously repacked with the WGMMA mini-block order.
Packed-K weights retain their existing layout and require WGMMA. Ordinary
per-group scales use the shared layout described below; block scales and fused
E8M0 scales have separate storage rules. A tuning override must be compatible
with all stored tensors, including the scales.

`umma` requires SM10x/SM11x GPUs and CUDA 12.9+, with FP16/BF16 outputs and FP32
accumulation. TS continues to handle repacked mixed-width operands. Native SS
uses 256 threads and retains dense/MoE, Stream-K and cooperative CTA scheduling.
SS keeps the input-scale global-memory layout unchanged. It rearranges scales
in the existing AS shared-memory storage when the M tile is 128-row aligned,
and uses scratch storage otherwise. Scale copies and MMA instructions share
one issuer and one TMEM scale buffer. With separate TMA loading warps, AS has
its own completion barrier. When scales can be prepared without reading B/BS,
two-CTA SS loads publish A/B completion to the issuer through cooperative TMA;
the scale warp can prepare AS before those operands finish loading. Indexed SS
keeps all activation-loading threads on cp.async and tracks TMA weight completion
separately, so scale preparation can overlap B/BS loading.

With 32-row chunked output, separate output storage and non-indexed scheduling, SS
uses the available TMEM capacity for an overlapping accumulator pair. The
epilogue reads overlapping rows first so the next tile can begin computing.
Native FP4 SS stages whose K size is a multiple of 256 use K64+96+96 issues
per 256 elements. Other stage sizes retain the standard instruction shape.
Indexed SS also overlaps 32-row chunked output with the next accumulator tile; row-index
buffers are released separately after output scatter finishes. MoE selection
accounts for SS's single scale buffer, chunked output, and cooperative CTA pairs.
It retains whole-tile output for small M tiles and uses sampled expert sizes and
available work to avoid underfilled tiles and short cooperative pipelines.
These optimizations are selected internally; TS keeps its existing schedule.
SS is selected from the fixed weight layout and the tuning backend.

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

`mma_type` selects `mma`, `wgmma`, `umma`, or `mxmma`. Heuristics resolve it per
shape; explicit tuning configurations can override it without changing LayerConfig
or transforming weights again, provided the selected backend supports the fixed
weight and scale layouts. Block-scaled layers require UMMA on SM10x/SM11x or MXMMA
on SM12x. Move `mma_type` from old layer dictionaries into tuning dictionaries and
re-transform weights created with the old packing convention.

On NVIDIA GPUs, ordinary per-group weight scales use the WGMMA layout for both
MMA and WGMMA. For low-bit activations, MMA loads even and odd N channels into
separate register sequences and applies the converted values to the corresponding
accumulators. FP32 and integer accumulation need no intermediate scale repacking;
FP16 accumulation assembles half2 pairs when applying scales. These MMA group-scale
configurations require block N >= 64, matching the scale packing block.

Re-transform group scales previously stored in the MMA layout. Fused E8M0, native
block-scaled, channel, tensor, and block-scale layouts are unchanged.

### Block and Warp Shapes

`block_shape` and `warp_shape` are 3D tuples representing the M/N/K dimensions, with the following constraints:

- `block_shape[i]` must be a power-of-2 multiple of `warp_shape[i]`.
- `block_shape_n` must be at least 64.
- When using WGMMA, `block_shape_n` must be at least 4x `warp_shape_n`.
- When using UMMA, block M/K must equal warp M/K, warp N is 32, M is a multiple of 8 in [8, 256], and K is a power of two of at least 32. Block N can be 128, 256, or 512; the tile must fit SMEM and TMEM.
- For indexed GEMMs, align `sorted_ids` and `expert_ids` to each projection's `block_shape_m`.
- `warp_shape_m` must be a multiple of MMA shape M.
- Valid values for `warp_shape_n` and `warp_shape_k` depend on the activation type:

| Activation Type | `warp_shape_n` | `warp_shape_k` |
|----------------|----------------|----------------|
| `float16` / `bfloat16` | 32, 64 | 32, 64 |
| `float8e4m3` / `float8e5m2` / `int8` | 16, 32, 64 | 64, 128 |
| `float4e2m1` / `int4` | 16, 32, 64 | 128, 256 |

With `use_packed_k_layout`, warp K must be 128 and warp N can start at 16. Activation scale groups,
when present, must cover warp K. Weight scale groups must also cover warp K unless
`use_fused_e8m0_scale` is enabled; fused conversion applies each K32 slab's weight
scale before WGMMA, so GS32 weights can use packed warp K128. Fused packed-K
remains opt-in; the default layout selection is unchanged.

`raster_group_m` controls M tile grouping for dense and grouped-contiguous GEMMs.
For grouped-contiguous GEMMs, values greater than 1 automatically use an expert
tile prefix table and binary lookup, allowing M tile IDs to move backwards as N
advances. A value of 1 uses the existing forward warp scan without the prefix table.

### Pipeline and Synchronization

| Parameter | Description |
|-----------|-------------|
| `num_stages` | Number of pipeline stages. Must be at least 2. Must be at least 3 when using `use_warp_spec` with WGMMA. |
| `use_warp_spec` | Whether to enable Warp Specialization. Requires SM90+. Required for UMMA. |
| `wgmma_use_late_as` | Delay per-group input-scale register loads until WGMMA accumulator promotion. Defaults to `False`; has no effect for per-token or per-tensor input scales. |
| `wgmma_split_issue_wait` | Prefetch the next fragment between WGMMA issue and wait. Defaults to `False`; independent of input-scale granularity and `wgmma_use_late_as`. |
| `use_mbarrier` | Whether to use MBarrier. Requires SM80+. |
| `use_cp_async` | Whether to use CP Async. Requires SM80+. |
| `num_ctas_per_sm` | Number of CTAs (Cooperative Thread Arrays / Thread Blocks) launched per SM. |
| `umma_cta_group_size` | `1` (default) or `2`. With `2`, a cluster of two CTAs cooperatively executes UMMA for adjacent N tiles. This is independent of CTA residency and TMA multicast. |
| `output_chunk_rows` | Output rows per shared-memory chunk for every MMA backend. `0` (default) writes a full tile; positive values must be multiples of 32 up to 256 and are clamped to tile M. Partial final chunks are supported. UMMA alternates two buffers; other backends reuse one buffer. Supports TMA and regular stores, Stream-K, and MoE scatter. Replaces `num_write_splits` (use half of tile M to reproduce two splits). |

The two `wgmma_*` options apply only to WGMMA and support both warp-specialized
and non-warp-specialized kernels. All four combinations are supported. They keep
the existing weight layout and weight-scale consumption order; scale prefetches
that would overwrite a live scale buffer wait until the current promotion finishes.
Tune these options together with tile shapes, pipeline stages, and Stream-K.

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

Chunked output supports dense, indexed, and grouped GEMMs, TMA or ordinary
stores, and all `smem_reuse_mode` values. Block N can be 128, 256, or 512;
block M is a multiple of 8 for one CTA, or 16 for two CTAs, subject to SMEM and
TMEM capacity. The two-CTA M alignment comes from the transposed `tcgen05.mma`
instruction's N dimension. Indexed output uses ordinary stores and preserves
row indices until all chunks have been written. Grouped TMA output uses the
existing per-CTA descriptor buffer and expert row offset.

Positive output chunk heights are multiples of 32, clamped to block M.
The TMA box retains that height. When it does not divide block M, the per-CTA
descriptor's global M boundary is updated for each tile so excess tail rows are
masked. Updates wait for outstanding stores and publish the descriptor through
the tensor-map proxy fences. Dense tiles with evenly dividing chunks need no updates.
When the actual output N (excluding padding) is divisible by 64, a 3D descriptor
combines 64-column SMEM slabs into one store. UMMA chunked output combines 128
columns per partition; other output paths combine the entire block N. Otherwise,
2D stores retain exact N bounds and mask padded columns. Stream-K uses matching
3D or 2D reduction stores.
Each N partition writes only its own columns; the accumulator is released only
after the last partition has been read. Reusing stage SMEM waits for output
completion before loading the next tile, reducing load/epilogue overlap.

Two-CTA execution requires N divisible by twice block N,
and `num_ctas_per_sm=1`. Both TMA and cp.async stage loads are supported,
including indexed A gathers. Each CTA loads only its own half of A; both CTAs
publish operand readiness before the leader issues UMMA.
Cooperative instructions support FP16/BF16, ordinary FP8 with FP8/FP6/FP4
weights, and MXFP8 with MXFP8/MXFP6/MXFP4 weights and group-32 E8M0 scales.
FP4 activations use native `mxf4nvf4` instructions with packed FP4 weights:
MXFP4 uses group-32 E8M0, while group-16 supports E8M0 or E4M3 (NVFP4).
Both one-CTA and two-CTA execution support these formats. The SM100 dispatcher
also selects UMMA for supported FP4 activation configurations.

TS also supports lower-bit integer weights, such as INT2 with MXFP4 or NVFP4
activations, through the existing register conversion and TMEM store path.
Hardware group scales must be nonnegative (E8M0 or unsigned E4M3).
Either operand may omit group scales: its hardware scales are filled with one,
while tensor/token activation scales and tensor/channel weight scales are applied
in the epilogue. This includes tokenwise FP4 with channelwise FP4 in TS and SS.

SM100-family UMMA also supports the undocumented `float8e3m4` and
`float4e0m3` formats. E3M4 supports ordinary FP8 and group-32 E8M0 scaling;
E0M3 requires group-16 E8M0 or E4M3 scales (group-32 faults in hardware).
Activation and weight formats may differ, including E3M4 with ordinary
FP8/FP6/FP4 weights and E0M3 with E2M1 weights, in either FP4 operand.
UMMA selects these formats directly in its instruction descriptor. Input
quantization uses the existing F2FP cubin patcher, extended to SM100/103;
SM120/121 MMA patching remains unchanged.

Tensor/token activation scales and MX activation scales use the existing
loaders. `static_tensor_dynamic_group` applies the secondary tensor scale in
the UMMA epilogue; NVFP4 `dynamic_group_token` similarly applies the secondary
per-token scale before output conversion. Input-scale GMEM layout is unchanged.
The TMEM scale allocation pads small M tiles to keep successive K scale words
aligned. The resource estimator accounts for the scale group size and padding.
Channel weight scales, channel secondary scales,
bias, and channel/group zero points reuse the existing loaders and arithmetic.
Channel parameters are released once all consuming threads have read them.
Both output paths support Stream-K: the first slice stores each chunk, later
slices reduce into it, and partial writes complete before releasing the output
lock. Bias is applied only by the first slice.

SM100 FP16/BF16 dense heuristics select two CTAs with six stages when the tile is suitable,
K is long enough to amortize the pipeline, and the estimated shared-memory
allocation fits. The existing Stream-K decision is preserved for CTA pairs.
Without Stream-K, underfilled output waves retain single-CTA execution. Chunked
output remains opt-in for single-CTA execution because it did not improve the
measured large dense cases by itself. FP8/FP4 cooperative execution is currently
explicitly configured with `umma_cta_group_size=2`; automatic cooperative selection
remains limited to FP16/BF16. All output chunk heights, including full-tile output,
are supported. The heuristics use `output_chunk_rows=32`, which also enables
overlapping accumulator reuse when the tile and scheduling support it.

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
