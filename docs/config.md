# Humming 配置

Humming 的 GEMM 配置分为 `LayerConfig`、`ComputeConfig` 和 `TuningConfig` 三部分。

| 配置 | 描述内容 | 主要影响 |
|---|---|---|
| `LayerConfig` | 层的形状、数据类型、量化方式与权重布局 | 数据表示、权重预处理，以及可用的计算路径 |
| `ComputeConfig` | GEMM 类型、精度和计算行为 | 输入组织、累加方式和 batch invariant |
| `TuningConfig` | kernel 的分块、流水线、数据搬运和调度 | 执行效率与资源占用 |

`LayerConfig` 在权重转换时确定；更改其中影响存储的选项后，需要重新转换权重及相关 scale。
`ComputeConfig` 和 `TuningConfig` 可以在调用时选择，但必须与已有数据布局兼容。
例如，切换 `mma_type` 不一定需要重新转换权重，但目标 backend 必须支持现有的权重和 scale 布局。

下文使用 GEMM 的 M/N/K 维度：M 表示输入行数，N 表示输出特征数，K 表示归约维度。
参数定义见 [config.py](../humming/config/config.py)，组合限制还会在 kernel 初始化时检查。

## LayerConfig

`LayerConfig` 描述一层数据的表示方式。同一组转换后的权重可以服务于不同 M 的输入，但其数据类型、N/K 形状、量化参数和布局保持固定。

### 形状与目标设备

#### `sm_version`

目标 GPU 的 SM 版本，例如 `90` 表示 SM90。省略时使用当前设备。

它参与决定 weight layout、反量化方式和可用指令。转换后的层需在相同 SM 版本上使用；跨架构部署时，应分别准备对应的层配置与权重。

#### `shape_n`、`shape_k` 与 `pad_shape_n`、`pad_shape_k`

`shape_n` 和 `shape_k` 是 **padding 后** 的 N/K；`pad_shape_n` 和 `pad_shape_k` 是补齐的元素数，默认均为 `0`。

- 实际输出特征数为 `shape_n - pad_shape_n`。
- 实际输入特征数为 `shape_k - pad_shape_k`。
- 输入和输出 tensor 使用实际特征数，转换后的权重使用 padding 后的形状。

例如，`shape_n=1024`、`pad_shape_n=24` 表示实际输出有 1000 个 channel。
Padding 用于满足布局和 kernel 的对齐要求，也会增加存储量及部分计算量；应使用权重转换流程确定的值，不能仅修改配置而不调整存储。

#### `num_experts`

MoE 的 expert 数量，普通 GEMM 使用 `0`。它决定权重及相关参数的 expert 维度。

该参数不表示 top-k，也不决定每个 expert 的实际 token 数；这些信息由调用时的输入和 routing 数据提供。

### 数据类型

#### `a_dtype`、`b_dtype`

分别指定 GEMM kernel 使用的 activation 和 weight 类型，影响量化精度、存储大小、反量化成本以及可用 backend。

- `a_dtype` 指量化后送入 GEMM 的 activation 类型，调用方原始输入可以先经过 input quantization。
- `b_dtype` 与转换后的 weight 表示匹配，低位宽数据通常以 packed 形式存储。
- 支持组合还取决于 SM 版本、scale 和 zero point；参见 [README](../README.md)。

#### `c_dtype`

输出类型，支持 `float16` / `bfloat16`。它与 accumulator 精度是两个不同的选项；输出为 FP16 不代表中间累加也使用 FP16。

#### `as_dtype`、`bs_dtype`

分别描述 activation scale 与 weight scale 的类型。在 group quantization 中，它们决定 group scale 的存储精度与格式。

- `as_dtype` 通常自动推导：没有 input scale 时为 `None`；普通路径通常使用 FP32；block-scaled 路径根据量化格式选择 E8M0 或 E4M3 等类型。
- `bs_dtype` 默认使用 `c_dtype`。指定低精度 scale 时，需要同时满足量化格式和 backend 的要求。
- 同时使用 activation 和 weight group scale 的 block-scaled 路径要求两者的 scale dtype 一致。

Scale dtype 不直接决定输出 dtype。减小 scale 的位宽可以降低 scale 存储量，但也会改变可表示的范围和量化误差。

### 输入量化

#### `input_quant_mode`

指定输入量化方式，以及 input scale 的组织形式。

| 值 | Scale 组织 |
|---|---|
| `none` | 不进行输入量化，用于 FP16/BF16 activation。 |
| `static_tensor` | 使用预先给定的 tensor scale。 |
| `dynamic_token` | 每个 token 动态计算一个 scale。 |
| `dynamic_group` | 每个 token 的每个 K group 动态计算 scale。 |
| `static_tensor_dynamic_group` | static tensor scale 与 dynamic group scale 组合。 |
| `dynamic_group_token` | dynamic group scale 与 secondary token scale 组合。 |

省略时，16-bit activation 使用 `none`；低精度 activation 在 `input_scale_group_size > 0` 时使用 `dynamic_group`，否则使用 `dynamic_token`。

带 secondary input scale 的两种组合模式要求 block-scaled MMA。
`dynamic_group_token` 用于支持的 FP4 activation，要求 group size 为 `16`，group scale 为 E4M3。

#### `input_scale_group_size`

每个 token 沿 K 的量化 group 大小；无 group scale 时为 `0`。
例如 K=4096、group size=128 时，每个 token 有 32 个 K group，各自使用一个 group scale。

较小的 group 通常能更细致地适应数值分布，但会增加 scale 数量和处理开销。取值还需满足量化格式、backend 和 tile 的对齐要求。
同时使用 activation 和 weight group scale 的 block-scaled 路径要求两者的 group size 一致。

### 权重量化

#### `weight_scale_type`

指定 primary weight scale 的共享范围。下表以逻辑 weight 矩阵 `[N, K]` 为例；MoE 中每个 expert 分别使用对应的 scale。

| 值 | Scale 组织 |
|---|---|
| `group` | 每个输出 channel 沿 K 分组，每组一个 scale。 |
| `block` | 沿 N/K 分块，一个 scale 由整个 block 共享。 |
| `channel` | 每个输出 channel 一个 scale。 |
| `tensor` | 整个 weight 矩阵一个 scale。 |

省略时，`weight_scale_group_size_n > 1` 推导为 `block`；否则根据 K group size 是否为 `0`，选择 `channel` 或 `group`。
这些模式描述 scale 的逻辑含义；转换后的物理存储还可能经过 packing 或重排。

#### `weight_scale_group_size`、`weight_scale_group_size_n`

分别指定 scale 沿 K 和 N 的共享范围。

- `group` / `block` 要求 `weight_scale_group_size > 0`。
- `channel` / `tensor` 要求 `weight_scale_group_size=0`。
- `weight_scale_group_size_n` 用于 block scale，描述有多少个相邻输出 channel 共享 scale。

例如 `block` 模式下 N/K group size 分别为 128/128，表示每个逻辑 `[128, 128]` weight block 使用一个 scale。
Group size 属于已有量化权重的格式，不能作为普通 tuning 参数直接修改。

#### `weight_scale_2_type`

Secondary weight scale 的类型，支持 `none`、`channel`、`tensor`，默认 `none`。
它与 primary scale 组合使用，表达额外的 channel 或 tensor 级缩放。

支持的组合包括：

- `group` primary scale，加 `channel` 或 `tensor` secondary scale。
- `channel` primary scale，加 `tensor` secondary scale。

部分预处理路径会自动提取 secondary scale，因此最终配置可能与最初传入值不同。调用时应使用转换后的完整参数集合。

#### `has_zero_point`、`is_fp_zero_point`

`has_zero_point` 控制是否在反量化中进行 zero-point 修正，默认 `False`。启用后需要提供相应的 zero-point tensor。

`is_fp_zero_point` 决定 zero point 的表示：

- `False`：使用与量化位宽一致的 integer zero point，仅用于 integer weight。
- `True`：使用 FP16/BF16 zero point，要求 FP16/BF16 activation。

Zero point 的类型和布局需要与 weight quantization 及预处理结果一致；这两个选项不能只在 kernel 调用时切换。

### 预处理与其他选项

#### `use_int_weight_scale`

将 group weight scale 转换为整数表示，以加速部分 INT8/INT4 activation 路径。该转换可能引入额外的 scale 舍入误差，并提取 secondary scale。

通常保留自动选择。主要限制是使用 group weight scale、没有 input group scale，且不能与 channel secondary weight scale 组合。
转换后的 scale 采用特殊存储表示，应由对应预处理流程生成。

#### `use_fused_e8m0_scale`

在 FP4 weight 转换为 FP8/INT8 的过程中融合 E8M0 group scale，并提取 secondary scale，减少单独处理 group scale 的工作。

通常自动选择，用于支持的 8-bit activation 与 E2M1 weight 路径。它会改变 scale 的预处理和消费方式，需要与转换后的权重参数保持一致。

#### `use_packed_k_layout`

使用 packed-K weight layout，将 K 方向的数据组织成适合 WGMMA 的形式。

- 要求 SM90、8-bit activation、even-bit weight，且不是 raw weight 路径。
- 转换后的权重要求使用 WGMMA，并满足对应的 warp K 与 scale group 限制。
- 省略时根据量化参数和层形状自动选择；改变它需要重新进行权重转换。

通常保留自动选择。它决定数据布局，不属于可以在同一份权重上随意切换的 backend tuning 开关。

#### `has_bias`

是否在 epilogue 中融合 bias addition，默认 `False`。启用时需提供与实际输出 channel 对应的 bias；MoE 中使用对应 expert 的 bias。

## ComputeConfig

### `gemm_type`

选择 GEMM 的输入组织与调度形式。

| 值 | 使用方式 |
|---|---|
| `dense` | 普通矩阵乘法，输入行共享同一组权重。 |
| `indexed` | 根据 routing 索引读取输入，并将结果写回对应位置。 |
| `grouped_contiguous` | 各 expert 的输入沿行方向连续存储，使用 expert 边界描述分组。 |
| `grouped_masked` | 各 expert 使用固定容量的存储区域，并提供实际有效行数。 |

普通层可以自动选择 `dense`；MoE 需明确指定类型，并提供该类型要求的 routing 或 expert layout 数据。
这些类型对 TMA、tile 对齐和输出方式的支持不同，切换时也需要调整输入组织。

### `use_f16_accum`

使用 FP16 accumulator，默认 `False`。它控制中间累加精度，而不是输出 tensor 的类型。

- 在支持的路径中可减少 accumulator 存储及部分计算开销。
- 相比 FP32 accumulation，舍入误差和溢出风险更高；K 较长或数据范围较大时尤其需要检查精度。
- 仅支持特定 activation/backend 组合；UMMA、MXMMA 及 block-scaled 路径的精度限制需同时满足。

### `use_batch_invariant`

启用 batch invariant 支持，约束归约与调优选择，避免 batch 大小改变计算组织后产生不同的数值结果。默认 `False`。

要求关闭 Stream-K，满足 `warp_shape_k == block_shape_k`，并在不同 batch 大小下固定使用同一种 `mma_type`。启用后 heuristic 会配合调整配置；手动 tuning 时也需要遵守这些限制，可能牺牲部分性能。

### `use_m_major_input_scale`

使用 M-major input scale 布局，默认 `False`。对于 group scale，可以将其理解为让同一 K group 的不同 token scale 沿 M 方向连续排列。

输入预处理和 GEMM 必须使用相同布局；仅修改 GEMM 配置不会自动转换已有 scale tensor。
`indexed` GEMM 不支持该选项；启用 `use_tma_as` 时要求它为 `True`。

## TuningConfig

建议从 heuristic 生成的配置开始，只调整目标 workload 中可能有收益的参数。
以下候选值是尝试方向，仍需满足 backend、量化格式、对齐和资源限制；性能以实际测量为准。

### 计算后端

#### `mma_type`

选择使用的 MMA backend。它决定计算指令以及对应的线程、数据加载和 accumulator 组织方式。

| 值 | 主要适用范围 |
|---|---|
| `mma` | 普通 MMA 路径，具体 dtype 支持取决于架构。 |
| `wgmma` | SM90 上的 WGMMA 路径。 |
| `umma` | 支持的 SM10x/SM11x 配置，要求 CUDA 12.9+。 |
| `mxmma` | 支持的 SM12x block-scaled 配置。 |

Block-scaled 层必须使用对应架构的 UMMA 或 MXMMA；packed-K 布局要求 WGMMA。
通常保留 heuristic 选择；同一布局支持多个 backend 时，可以比较其性能，但切换时需要同时使用该 backend 支持的 tile 和 pipeline 配置。

### 分块与线程组织

#### `block_shape`

CTA 的 `(M, N, K)` tile，影响数据复用、并行度与 SMEM/register 占用。

- **M**：小 batch 或每个 expert 行数较少时，优先使用较小的 tile M，减少 padding 和无效计算；行数充足时可增大以复用 weight。
- **N**：较大的 tile N 可以复用 activation，但会增加 accumulator 和 weight tile 的资源需求。
- **K**：增大后可减少 stage 迭代次数，但每个 stage 更大，可能压缩可用 stage 数或 CTA 并发。

调优时先在 heuristic 配置附近调整；较大的 tile 还会减少独立输出 tile 的数量，小矩阵可能因此无法充分利用所有 SM。

#### `warp_shape`

warp 的 `(M, N, K)` 计算分块；WGMMA 由四个 warp 协作，UMMA 则使用该字段表达其特定线程组织约束。
在普通 MMA/WGMMA 路径中，block 与 warp 各维度的比值共同决定计算线程数，因此减小 warp tile 不一定会减少整个 CTA 的资源占用。

通常保留 backend 推荐值。Register 压力较大时可尝试减小 warp M/N；增加 K 方向的 warp 数可提高归约并行度，但需要额外的部分结果归约。

主要 shape 限制：

- 普通 MMA/WGMMA 的 block 各维度需为 warp 对应维度的 power-of-two 倍数，并满足指令、量化 group 和布局的对齐要求。
- WGMMA 要求 N 方向组成完整的四-warp group，即 `block_shape_n / warp_shape_n` 为 `4` 的倍数。
- UMMA 要求 block M/K 等于 warp M/K，warp N 为 `32`，block N 为 `128`、`256` 或 `512`；每行 activation 的 block K 数据至少为 64 bytes。
- packed-K 要求 warp K 为 `128`；input group scale 必须覆盖 warp K，非 fused E8M0 的 weight group scale 也需覆盖 warp K。
- `indexed` GEMM 的 `sorted_ids`、`expert_ids` 需按对应 projection 的 block M 对齐。

### 流水线与并发

#### `num_stages`

流水线缓冲 stage 数，通常至少为 `2`，WGMMA 至少为 `3`。更多 stage 允许提前准备后续计算的数据，但每增加一个 stage 都需要额外的 SMEM。

K 较长时可尝试从 `2` / `3` 增至 `4` 或更多以隐藏访存延迟；K 较短或 SMEM 限制并发时，优先减少 stage。应结合 `block_shape_k` 判断实际有多少次迭代可以利用这些缓冲。

#### `num_ctas_per_sm`

每个 SM 的 CTA 调度数量，也参与 launch bounds 和资源预算；不代表硬件保证的实际 occupancy。
增大后，每个 CTA 可用的 register 和 SMEM 预算会更紧，可能需要配合减小 tile 或 stage 数。

通常从 `1` 开始，小 tile 且资源充足时可尝试 `2` 以提高并发；大 tile 或较深流水线通常保留 `1`。

#### `use_warp_spec`

使用 warp specialization，让加载与计算由不同线程分工；要求对应架构及 backend 支持，UMMA 使用该方式。

在 SM90 上可比较 `True` / `False`：较长流水线可能受益，但额外线程会占用资源，小 workload 未必划算。

### 数据搬运与同步

数据搬运与同步中，以下行为由 kernel 自动决定，不提供配置开关：

- SM80+ 自动启用 cp.async，其他架构关闭。
- UMMA、warp specialization 或 TMA 路径自动启用 MBarrier，其他情况关闭。

#### `use_tma`

TMA（Tensor Memory Accelerator）总开关，要求 SM90+，默认 `False`。

- `use_tma=False` 时，不能将任一 `use_tma_*` 显式设为 `True`。
- `use_tma=True` 时，未指定的张量级开关通常继承总开关；`use_tma_as` 是例外，默认仍为 `False`。
- Kernel 还会根据 tensor 是否存在、scale 粒度和 GEMM 类型调整实际使用的 TMA 路径。

规则且较大的 tile 可优先尝试 `True`；较小或不规则的数据搬运可比较普通加载方式。TMA 有 descriptor 和同步开销，开启总开关后也可以按 tensor 分别比较。

#### `use_tma_a`、`use_tma_b`

分别控制 activation 和 weight 的 TMA load。

连续且对齐的大块数据可优先开启；indexed A gather 等不支持的路径需关闭对应开关。UMMA SS 要求 `use_tma_b=True`。

#### `use_tma_c`

控制输出的 TMA store。

规则的连续输出可尝试 `True`；indexed scatter 使用普通 store。小输出需比较 TMA 设置与同步开销。

#### `use_tma_as`、`use_tma_as2`

分别控制 input scale 与 secondary input scale 的 TMA load。
`use_tma_as` 默认关闭，启用时要求 `use_m_major_input_scale=True`；`indexed` GEMM 不支持这两个 TMA scale 开关。
Scale 布局满足要求且搬运量较大时再尝试开启。Tensor scale 等路径不使用对应 TMA load。

#### `use_tma_bs`、`use_tma_bs2`

分别控制 weight scale 与 secondary weight scale 的 TMA load。

Group/channel scale 较多时可尝试开启；secondary scale 的 TMA 路径用于 channel scale，tensor scale 不需要这种搬运方式。

#### `use_tma_bzp`、`use_tma_bias`

分别控制 zero point 和 bias 的 TMA load。

通常跟随 `use_tma`；这些参数搬运量较小时可以尝试关闭，减少同步开销。没有对应张量时不生效。

### 输出与共享内存

#### `output_chunk_rows`

每次通过 SMEM 输出的行数，默认 `0`。

- `0`：按整个 tile 输出。
- 正值：必须是 `32` 的倍数且不超过 `256`，实际 chunk 高度不超过 tile M；末尾不足一个 chunk 的行也支持输出。

它改变输出的分块方式，不改变输出 tensor 的逻辑形状。
SMEM 压力较大时可尝试 `32` 或 `64`；分块可减少输出缓冲需求，并在部分路径中改善重叠，但也可能增加同步和 store 开销。

#### `smem_reuse_mode`

输出缓冲对流水线 SMEM 的复用方式。

| 值 | 存储安排 |
|---|---|
| `none` | 输出缓冲独立分配，不复用 stage 存储。 |
| `last_stage` | 输出缓冲从最后一个 stage 的存储位置开始复用。 |
| `all_stages` | 输出缓冲与整个 stage 存储区域复用。 |

默认 UMMA 使用 `none`，其他 backend 使用 `all_stages`。
SMEM 紧张时尝试 `last_stage` 或 `all_stages`；资源充足时尝试 `none`，减少加载与输出相互等待。可配合 `output_chunk_rows` 缩小输出缓冲，比较是否仍需要复用。

### 工作调度与数据复用

#### `use_stream_k`

沿 K 划分工作，让多个 CTA 分担同一输出 tile 的计算，改善输出 tile 不足或 wave 不均衡时的负载分配。部分结果需要通过额外的归约和同步合并。

启用后，循环迭代次数会从编译期常量变为运行时变量，可能限制编译器对循环的优化，因此在部分情况下反而会变慢。

M/N 较小而 K 较长时可尝试 `True`；tile 已足够多或 K 较短时可尝试 `False`，避免部分结果归约和同步开销。启用 batch invariant 时要求关闭。

#### `raster_group_m`

控制 dense / grouped-contiguous GEMM 的 M tile 分组遍历顺序。

默认 `1`；相邻 M tile 可复用 weight 数据时，可尝试 `2`、`4`、`8` 改善 cache locality。较大的分组也会改变 A 的复用，grouped-contiguous 路径还会增加 expert tile lookup 开销。

#### `multi_cast_size_a`、`multi_cast_size_b`

TMA multicast 的共享 CTA 数，默认 `1`，表示关闭 multicast。

- `multi_cast_size_a` 在不同 N tile 之间共享相同的 activation 数据。
- `multi_cast_size_b` 在不同 M tile 之间共享相同的 weight 数据；当前 kernel 要求 B multicast 使用 dense GEMM。
- 两者不能同时大于 `1`。需要启用对应 tensor 的 TMA load 和 warp specialization，并满足 cluster 与 tile 的对齐要求；UMMA 当前要求两者均为 `1`。

多个 CTA 重复读取相同 A/B tile 时，可在支持的 backend 上尝试 `2`；复用不足时保留 `1`，避免 cluster 调度约束。

### 后端专用选项

#### `wgmma_use_late_as`

将 input group scale 的 register load 延后至 accumulator promotion，默认 `False`，仅适用于带 input group scale 的 WGMMA。

Register 压力较高时可尝试 `True`，但延后加载也可能增加 scale 等待时间。

#### `wgmma_split_issue_wait`

在 WGMMA issue 与 wait 之间预取下一个 fragment，默认 `False`。它与 `wgmma_use_late_as` 可以独立选择，均适用于带或不带 warp specialization 的 WGMMA 路径。

希望增加加载与计算重叠时可尝试 `True`。它会延长部分 operand 和 accumulator 的存活时间；register 预算紧张时优先关闭，或配合减小 tile、减少 CTA 内的计算线程数，再与 stage 数一起比较。

#### `umma_num_dequant_warpgroups`

UMMA TS 中负责 weight dequantization 的 warp group 数，支持 `1` / `2`。

通常使用 `1`；weight 转换成为瓶颈时可尝试 `2`，代价是更多线程和资源占用。Raw weight 的 SS 路径无需增加此值。

#### `umma_cta_group_size`

UMMA 协作 CTA 数，支持 `1` / `2`，默认 `1`。设为 `2` 时，两个 CTA 协作计算相邻的 N tile；它与 TMA multicast 和 CTA residency 是不同概念。

两-CTA 模式要求：

- block M 为 `16` 的倍数。
- N 能被两倍 block N 整除。
- `num_ctas_per_sm=1`，并满足对应数据类型和资源限制。

较大的规则矩阵、较长 K 可尝试 `2`；小 workload 优先 `1`，避免协作开销。可以同时比较输出 chunk 大小，调整协作计算与输出的重叠。

### Kernel 间调度

#### `use_pdl`

启用 Programmatic Dependent Launch，默认 `False`。

在调用链支持 PDL、相邻 kernel 有可重叠工作时尝试 `True`，并测量端到端延迟；孤立的 GEMM benchmark 通常保留 `False`。
