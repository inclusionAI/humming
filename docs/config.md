# Humming Configuration

Humming divides GEMM configuration into `LayerConfig`, `ComputeConfig`, and `TuningConfig`.

| Configuration | Describes | Main effects |
|---|---|---|
| `LayerConfig` | Layer shapes, data types, quantization, and weight layout | Data representation, weight preprocessing, and available compute paths |
| `ComputeConfig` | GEMM type, precision, and computation behavior | Input organization, accumulation, and batch invariant behavior |
| `TuningConfig` | Kernel tiling, pipelines, data transfers, and scheduling | Execution efficiency and resource usage |

`LayerConfig` is determined when weights are transformed. Changing an option that affects storage requires transforming the weights and associated scales again.
`ComputeConfig` and `TuningConfig` can be selected at invocation time, but must remain compatible with the existing data layouts.
For example, changing `mma_type` does not necessarily require another weight transformation, provided the selected backend supports the stored weight and scale layouts.

Throughout this guide, M is the number of input rows, N is the number of output features, and K is the reduction dimension.
Parameter definitions are in [config.py](../humming/config/config.py); additional compatibility checks run during kernel initialization.

## LayerConfig

`LayerConfig` describes how a layer's data is represented. The same transformed weights can serve inputs with different M dimensions, while their data types, N/K shapes, quantization parameters, and layouts remain fixed.

### Shapes and Target Device

#### `sm_version`

The target GPU's SM version, such as `90` for SM90. Defaults to the current device when omitted.

It affects weight layout, dequantization, and available instructions. A transformed layer must run on the same SM version it targets; prepare separate layer configurations and weights when deploying across architectures.

#### `shape_n`, `shape_k`, `pad_shape_n`, and `pad_shape_k`

`shape_n` and `shape_k` are the N/K dimensions **after padding**. `pad_shape_n` and `pad_shape_k` specify how many elements were added, both defaulting to `0`.

- The effective output feature count is `shape_n - pad_shape_n`.
- The effective input feature count is `shape_k - pad_shape_k`.
- Input and output tensors use the effective feature counts; transformed weights use the padded shapes.

For example, `shape_n=1024` and `pad_shape_n=24` represent 1000 output channels.
Padding satisfies layout and kernel alignment requirements, but also increases storage and some computation. Use the values established by weight transformation rather than changing the configuration without updating storage.

#### `num_experts`

The number of MoE experts. Use `0` for ordinary GEMM. This determines the expert dimension of weights and associated parameters.

It does not specify top-k or the actual token count for each expert; those come from the inputs and routing data supplied at invocation time.

### Data Types

#### `a_dtype`, `b_dtype`

The activation and weight types used by the GEMM kernel, respectively. They affect quantization precision, storage size, dequantization cost, and available backends.

- `a_dtype` is the activation type supplied to GEMM after quantization. The caller's original input may first pass through input quantization.
- `b_dtype` matches the transformed weight representation. Low-bit data is typically packed in storage.
- Supported combinations also depend on the SM version, scales, and zero points; see the [README](../README.md).

#### `c_dtype`

The output type: `float16` or `bfloat16`. This is separate from accumulator precision; FP16 output does not imply FP16 intermediate accumulation.

#### `as_dtype`, `bs_dtype`

The activation scale and weight scale types, respectively. For group quantization, these determine the precision and storage format of group scales.

- `as_dtype` is usually selected automatically: `None` when no input scale is present, typically FP32 for ordinary paths, and formats such as E8M0 or E4M3 for block-scaled paths, depending on the quantization format.
- `bs_dtype` defaults to `c_dtype`. Low-precision scales must satisfy both quantization format and backend requirements.
- Block-scaled paths that use both activation and weight group scales require matching scale data types.

Scale dtype does not directly determine output dtype. Reducing scale bit width saves scale storage, but also changes the representable range and quantization error.

### Input Quantization

#### `input_quant_mode`

The input quantization method and organization of input scales.

| Value | Scale organization |
|---|---|
| `none` | No input quantization; used for FP16/BF16 activations. |
| `static_tensor` | A supplied tensor scale. |
| `dynamic_token` | One dynamically computed scale per token. |
| `dynamic_group` | One dynamically computed scale per K group within each token. |
| `static_tensor_dynamic_group` | A static tensor scale combined with dynamic group scales. |
| `dynamic_group_token` | Dynamic group scales combined with a secondary token scale. |

When omitted, 16-bit activations use `none`. Lower-precision activations use `dynamic_group` when `input_scale_group_size > 0`, or `dynamic_token` otherwise.

The two combined modes with secondary input scales require block-scaled MMA.
`dynamic_group_token` is available for supported FP4 activations and requires group size `16` with E4M3 group scales.

#### `input_scale_group_size`

The quantization group size along K for each token. Use `0` when there are no group scales.
For example, K=4096 with group size 128 gives each token 32 K groups, each with its own group scale.

Smaller groups can usually adapt more closely to the value distribution, but increase the number of scales and processing overhead. Values must also satisfy quantization format, backend, and tile alignment requirements.
Block-scaled paths that use both activation and weight group scales require matching group sizes.

### Weight Quantization

#### `weight_scale_type`

The sharing granularity of primary weight scales. The table below describes a logical weight matrix `[N, K]`; each MoE expert uses its own corresponding scales.

| Value | Scale organization |
|---|---|
| `group` | One scale per K group within each output channel. |
| `block` | One scale shared by a block spanning N and K. |
| `channel` | One scale per output channel. |
| `tensor` | One scale for the entire weight matrix. |

When omitted, `weight_scale_group_size_n > 1` selects `block`. Otherwise, a K group size of `0` selects `channel`, and a positive size selects `group`.
These modes describe the logical meaning of scales; their transformed physical storage may also be packed or reordered.

#### `weight_scale_group_size`, `weight_scale_group_size_n`

The scale sharing granularity along K and N, respectively.

- `group` and `block` require `weight_scale_group_size > 0`.
- `channel` and `tensor` require `weight_scale_group_size=0`.
- `weight_scale_group_size_n` is used for block scales and specifies how many adjacent output channels share a scale.

For example, N/K group sizes of 128/128 in `block` mode assign one scale to each logical `[128, 128]` weight block.
Group sizes are part of the quantized weight format and cannot be changed like ordinary tuning parameters.

#### `weight_scale_2_type`

The secondary weight scale type: `none`, `channel`, or `tensor`, defaulting to `none`.
It combines with the primary scale to provide additional channel or tensor scaling.

Supported combinations include:

- A `group` primary scale with a `channel` or `tensor` secondary scale.
- A `channel` primary scale with a `tensor` secondary scale.

Some preprocessing paths automatically extract a secondary scale, so the resulting configuration may differ from the original input. Supply the complete transformed parameter set when invoking the kernel.

#### `has_zero_point`, `is_fp_zero_point`

`has_zero_point` enables zero-point correction during dequantization and defaults to `False`. When enabled, the corresponding zero-point tensor is required.

`is_fp_zero_point` selects the representation:

- `False`: integer zero points with the same bit width as the quantized weights; supported only for integer weights.
- `True`: FP16/BF16 zero points; requires FP16/BF16 activations.

The zero-point type and layout must match weight quantization and preprocessing. These options cannot be changed only at kernel invocation time.

### Preprocessing and Other Options

#### `use_int_weight_scale`

Converts group weight scales to an integer representation to accelerate some INT8/INT4 activation paths. This transformation can introduce additional scale rounding error and extracts a secondary scale.

Usually leave this to automatic selection. The main requirements are group weight scales, no input group scales, and no channel secondary weight scale.
The transformed scales use a special storage representation and should be generated by the corresponding preprocessing path.

#### `use_fused_e8m0_scale`

Fuses E8M0 group scales into FP4-to-FP8/INT8 weight conversion and extracts a secondary scale, reducing the work needed to apply group scales separately.

Usually selected automatically for supported 8-bit activation and E2M1 weight paths. It changes how scales are preprocessed and consumed, and must match the transformed weight parameters.

#### `use_packed_k_layout`

Uses a packed-K weight layout that organizes K data for WGMMA.

- Requires SM90, 8-bit activations, weights with an even bit width, and a path that does not use raw weights.
- The transformed weights require WGMMA and must satisfy its warp K and scale group constraints.
- When omitted, it is selected from the quantization parameters and layer shape. Changing it requires transforming weights again.

On SM90, fused E8M0 layers use packed-K by default when input scale groups cover K128 and N is a multiple of 64, for both dense and MoE GEMMs. Dense weights transformed with the previous unpacked default must be transformed again, or keep `use_packed_k_layout=False` explicitly.

Usually leave this to automatic selection. It determines data layout and cannot be freely switched as a backend tuning option on the same stored weights.

#### `has_bias`

Fuses bias addition into the epilogue. Defaults to `False`. When enabled, supply bias for the effective output channels; MoE uses the corresponding expert's bias.

## ComputeConfig

### `gemm_type`

The input organization and scheduling form of GEMM.

| Value | Usage |
|---|---|
| `dense` | Ordinary matrix multiplication, with input rows sharing the same weights. |
| `indexed` | Reads inputs through routing indices and writes results to the corresponding positions. |
| `grouped_contiguous` | Stores each expert's inputs contiguously along the row dimension, with expert boundaries describing the groups. |
| `grouped_masked` | Uses a fixed-capacity storage region per expert and supplies the actual valid row counts. |

Ordinary layers can automatically select `dense`. MoE requires an explicit type and the routing or expert layout data required by that type.
These forms differ in TMA support, tile alignment, and output handling; switching also requires matching input organization.

### `use_f16_accum`

Uses FP16 accumulators. Defaults to `False`. This controls intermediate accumulation precision, not the output tensor type.

- Can reduce accumulator storage and some computation overhead on supported paths.
- Has higher rounding error and overflow risk than FP32 accumulation. Check precision carefully for long K dimensions or wide value ranges.
- Supports only specific activation/backend combinations; the precision constraints of UMMA, MXMMA, and block-scaled paths also apply.

### `use_batch_invariant`

Enables batch invariant behavior by constraining reduction and tuning choices, preventing changes in computation organization across batch sizes from producing different numerical results. Defaults to `False`.

Requires Stream-K to be disabled, `warp_shape_k == block_shape_k`, and the same `mma_type` to be used across batch sizes. Heuristics adjust the configuration accordingly; manual tuning must also follow these constraints, which may reduce performance.

### `use_m_major_input_scale`

Uses an M-major input scale layout. Defaults to `False`. For group scales, this places scales from different tokens for the same K group contiguously along M.

Input preprocessing and GEMM must use the same layout; changing only the GEMM configuration does not convert existing scale tensors.
`indexed` GEMM does not support this option. Enabling `use_tma_as` requires it to be `True`, except for UMMA group scales (see `use_tma_as`).

## TuningConfig

Start from a configuration generated by the heuristics and adjust parameters that could benefit the target workload.
The suggested values below are starting points. Backend, quantization format, alignment, and resource constraints still apply; measure performance on the actual workload.

### Compute Backend

#### `mma_type`

Selects the MMA backend, determining the compute instructions and the associated thread, data-loading, and accumulator organization.

| Value | Main usage |
|---|---|
| `mma` | Ordinary MMA paths; dtype support depends on the architecture. |
| `wgmma` | WGMMA on SM90. |
| `umma` | Supported SM10x/SM11x configurations; requires CUDA 12.9+. |
| `mxmma` | Supported SM12x block-scaled configurations. |

Block-scaled layers require UMMA or MXMMA for the corresponding architecture; packed-K layouts require WGMMA.
Usually retain the heuristic choice. When multiple backends support the same layout, compare their performance using tile and pipeline configurations supported by each backend.

### Tiling and Thread Organization

#### `block_shape`

The CTA's `(M, N, K)` tile. Affects data reuse, parallelism, and shared-memory (SMEM) and register usage.

- **M**: prefer smaller tiles for small batches or few rows per expert to reduce padding and wasted computation. Larger tiles can improve weight reuse when enough rows are available.
- **N**: larger tiles improve activation reuse but require more resources for accumulators and weight tiles.
- **K**: larger tiles reduce stage iterations, but each stage consumes more storage, potentially limiting pipeline depth or concurrent CTAs.

Start with adjustments near the heuristic configuration. Larger tiles also reduce the number of independent output tiles, which can leave SMs underutilized for small matrices.

#### `warp_shape`

The warp's `(M, N, K)` compute tile. WGMMA uses four cooperating warps, while UMMA uses this field to express its specific thread organization constraints.
For ordinary MMA/WGMMA paths, the ratios between block and warp dimensions determine the number of compute threads, so a smaller warp tile does not necessarily reduce total CTA resource usage.

Usually retain the backend's recommended values. Try smaller warp M/N dimensions when register pressure is high. More warps along K can increase reduction parallelism, but require additional partial-result reduction.

Main shape constraints:

- For ordinary MMA/WGMMA, each block dimension must be a power-of-two multiple of the corresponding warp dimension, with the required instruction, quantization group, and layout alignment.
- WGMMA requires complete groups of four warps along N: `block_shape_n / warp_shape_n` must be a multiple of `4`.
- UMMA requires block M/K to equal warp M/K, warp N to be `32`, and block N to be `128`, `256`, or `512`. Each activation row must contain at least 64 bytes across block K.
- Packed-K requires warp K to be `128`. Input scale groups must cover warp K, as must weight scale groups unless fused E8M0 conversion is enabled.
- For `indexed` GEMM, align `sorted_ids` and `expert_ids` to the corresponding projection's block M.

### Pipelines and Concurrency

#### `num_stages`

The number of pipeline buffer stages, generally at least `2`; WGMMA requires at least `3`. More stages allow data to be prepared further ahead of computation, but each additional stage consumes SMEM.

For long K dimensions, try increasing from `2` / `3` to `4` or more to hide memory latency. Prefer fewer stages for short K dimensions or when SMEM limits concurrency. Consider `block_shape_k` when determining how many iterations can actually use these buffers.

#### `producer_stage_unroll`, `consumer_stage_unroll`

Control producer and consumer stage loop unrolling independently. `None` (default) resolves to `num_stages` when the tuning config is initialized; explicit values must be positive integers. A value of `1` disables unrolling.

The producer factor includes prefill and tail loops. Without warp specialization, the consumer factor also controls loads interleaved in the compute loop. Neither factor changes pipeline depth or fragment loop unrolling.

These factors apply to all MMA backends and do not need to divide `num_stages`. UMMA heuristics explicitly choose `4` for both, including when fewer than four stages are used. Other backends unroll within one stage cycle, so factors at least as large as that cycle fully unroll it.

For example, `num_stages=5, producer_stage_unroll=2, consumer_stage_unroll=1` keeps all five pipeline slots while requesting partial producer unrolling and no consumer stage unrolling. Smaller factors can reduce register pressure, but can add dynamic stage addressing; measure them together with tile shapes and pipeline depth.

#### `num_ctas_per_sm`

The number of CTAs scheduled per SM. It also affects launch bounds and resource budgets, but does not guarantee actual hardware occupancy.
Increasing it tightens each CTA's register and SMEM budgets, potentially requiring smaller tiles or fewer stages.

Usually start with `1`. Try `2` for small tiles with sufficient resources to improve concurrency; large tiles or deep pipelines usually favor `1`.

#### `use_warp_spec`

Uses warp specialization to assign loading and computation to different threads. Requires architecture and backend support; UMMA uses this organization.

On SM90, compare `True` and `False`: longer pipelines may benefit, but the extra threads consume resources and may not pay off for small workloads.

### Data Transfers and Synchronization

The kernel selects the following behavior automatically, without configuration switches:

- cp.async is enabled on SM80+ and disabled on earlier architectures.
- MBarrier is enabled for UMMA, warp specialization, or TMA, and disabled otherwise.

#### `use_tma`

The master switch for TMA (Tensor Memory Accelerator). Requires SM90+ and defaults to `False`.

- When `use_tma=False`, no `use_tma_*` option may be explicitly set to `True`.
- When `use_tma=True`, unspecified tensor-specific switches generally inherit it. `use_tma_as` is an exception and still defaults to `False`.
- The kernel also adjusts the TMA paths used according to tensor presence, scale granularity, and GEMM type.

Try `True` for large, regular tiles. Compare ordinary loads for small or irregular transfers. TMA has descriptor and synchronization overhead, so individual tensor switches are also worth comparing after enabling the master switch.

#### `use_tma_a`, `use_tma_b`

Control TMA loads for activations and weights, respectively.

Prefer enabling them for large, contiguous, aligned transfers. Disable the corresponding switch for unsupported paths such as indexed A gathers. UMMA SS requires `use_tma_b=True`.

#### `use_tma_c`

Controls TMA stores for the output.

Try `True` for regular, contiguous output; indexed scatter uses ordinary stores. For small outputs, compare the TMA setup and synchronization overhead.

#### `use_tma_as`, `use_tma_as2`

Control TMA loads for input scales and secondary input scales, respectively.
`use_tma_as` defaults to `False` and requires `use_m_major_input_scale=True` when enabled. UMMA can also load row-major group scales with TMA when a stage holds whole 4-byte scale words and each row of the scale tensor is 16-byte aligned. `indexed` GEMM does not support either TMA scale switch.
Try enabling them when the scale layout is supported and transfers are sufficiently large. Paths such as tensor scaling do not use the corresponding TMA loads.

#### `use_tma_bs`, `use_tma_bs2`

Control TMA loads for weight scales and secondary weight scales, respectively.

Try enabling them when there are many group/channel scales. The secondary scale TMA path handles channel scales; tensor scales do not need this transfer method.

#### `use_tma_bzp`, `use_tma_bias`

Control TMA loads for zero points and bias, respectively.

Usually follow `use_tma`. Try disabling them for small parameter transfers to reduce synchronization overhead. They have no effect when the corresponding tensor is absent.

### Output and Shared Memory

#### `output_chunk_rows`

The number of rows written through SMEM per output chunk. Defaults to `0`.

- `0`: write the entire tile.
- Positive values: must be multiples of `32`, no greater than `256`. The actual chunk height is capped at tile M; a partial final chunk is supported.

This changes output chunking, not the output tensor's logical shape.
Try `32` or `64` when SMEM pressure is high. Chunking can reduce output buffer requirements and improve overlap on some paths, but may increase synchronization and store overhead.

#### `smem_reuse_mode`

How the output buffer reuses pipeline SMEM.

| Value | Storage arrangement |
|---|---|
| `none` | Allocate output storage separately from stage storage. |
| `last_stage` | Reuse storage starting at the last stage's position. |
| `all_stages` | Reuse the entire stage storage region for the output buffer. |

Defaults to `none` for UMMA and `all_stages` for other backends.
Try `last_stage` or `all_stages` when SMEM is tight. With sufficient resources, try `none` to reduce waiting between loads and output. Combine this with `output_chunk_rows` to shrink the output buffer and reassess whether reuse is needed.

### Work Scheduling and Data Reuse

#### `use_stream_k`

Partitions work along K so multiple CTAs can share computation for an output tile, improving load distribution when there are too few output tiles or uneven waves. Partial results require additional reduction and synchronization.

Enabling it changes the loop iteration count from a compile-time constant to a runtime variable. This can limit compiler loop optimizations and make some cases slower.

Try `True` for small M/N and long K. Try `False` when there are already enough tiles or K is short to avoid partial-result reduction and synchronization overhead. Must be disabled for batch invariant behavior.

WGMMA output with multiple output warpgroups and `output_chunk_rows=0` uses one Stream-K lock per output warpgroup. Each group initializes and accumulates its own output region independently. K reduction, shared scratch reuse and indexed row buffer reuse still synchronize the math threads that share those resources.

Layer-owned lock buffers contain 2048 int32 elements. The launcher checks supplied lock capacity against the grid and kernel's lock count per tile.

#### `raster_group_m`

Controls grouped traversal of M tiles for dense and grouped-contiguous GEMM.

Defaults to `1`. Try `2`, `4`, or `8` to improve cache locality when adjacent M tiles can reuse weight data. Larger groups also change A reuse, and add expert tile lookup overhead in the grouped-contiguous path.

#### `multi_cast_size_a`, `multi_cast_size_b`

The number of CTAs sharing a TMA multicast. Defaults to `1`, which disables multicast.

- `multi_cast_size_a` shares activation data across different N tiles.
- `multi_cast_size_b` shares weight data across different M tiles. The current kernel requires dense GEMM for B multicast.
- Both cannot exceed `1` at the same time. Multicast requires the corresponding tensor's TMA load, warp specialization, and cluster/tile alignment. UMMA currently requires both to be `1`.

Try `2` on supported backends when multiple CTAs repeatedly read the same A/B tile. Keep `1` when reuse is limited to avoid cluster scheduling constraints.

### Backend-Specific Options

#### `wgmma_use_late_as`

Delays input group scale register loads until accumulator promotion. Defaults to `False` and applies only to WGMMA with input group scales.

Try `True` when register pressure is high, but note that delayed loads can also increase waiting for scales.

#### `wgmma_split_issue_wait`

Prefetches the next fragment between WGMMA issue and wait. Defaults to `False`. It can be selected independently of `wgmma_use_late_as`; both apply to WGMMA paths with or without warp specialization.

Try `True` to increase overlap between loading and computation. It extends the lifetime of some operands and accumulators. When register budgets are tight, prefer disabling it, or reduce tile sizes or the number of compute threads per CTA before comparing pipeline depths.

#### `umma_num_dequant_warpgroups`

The number of warp groups performing weight dequantization in UMMA TS. Supports `1` or `2`.

Usually use `1`. Try `2` when weight conversion is a bottleneck, at the cost of more threads and resource usage. The SS path for raw weights does not need this value increased.

#### `umma_cta_group_size`

The number of cooperating UMMA CTAs: `1` or `2`, defaulting to `1`. With `2`, two CTAs cooperate on adjacent N tiles. This is separate from TMA multicast and CTA residency.

Two-CTA mode requires:

- Block M to be a multiple of `16`.
- N to be divisible by twice block N.
- `num_ctas_per_sm=1`, along with the applicable data type and resource constraints.

Try `2` for large, regular matrices with long K. Prefer `1` for small workloads to avoid cooperation overhead. Also compare output chunk sizes to tune overlap between cooperative computation and output.

### Scheduling Between Kernels

#### `use_pdl`

Enables Programmatic Dependent Launch. Defaults to `False`.

Try `True` when the calling pipeline supports PDL and adjacent kernels have work that can overlap, measuring end-to-end latency. Usually keep `False` for isolated GEMM benchmarks.
