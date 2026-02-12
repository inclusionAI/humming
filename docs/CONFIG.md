
# Config Options

The following config options are build-time config options. 

Note that config options tagged with "🔔" may affect the weight tensors and/or their layouts.


## Shape Config

Every entry within the "Shape Config" must be a three-integer tuple specifying the M/N/K dimensions.

- `problem_shape`:
  - In the problem shape, the M-dimension is consistently ignored because the actual value of M is dynamically determined based on the kernel input, rather than being set as a compile-time parameter.
  - The N and K dimensions represent the N and K of the weight matrix. In the case of MoE, these represent the N and K dimensions of an single expert's weight matrix.
  - `problem_shape.n` and `problem_shape.k` must be divisable by `block_shape.n` and `block_shape.k`, respectfully.

- `block_shape`: 
  - `m`: 
    - Must be divisable by 16.
    - `block_shape.m // warp_shape.m` must be powers of 2.
  - `n`:
    - Must be powers of 2.
    - Must be divisable by 64.
    - If use WGMMA, `block_shape.n // warp_shape.n` must be divisable by 4.
  - `k`:
    - Must be powers of 2.
    - Must be divisable by 32/64/128 for 16/8/4 bit activation, respectfully.

- `warp_shape`:
  - `m`: 
    - Must be divisable by 16.
    - `block_shape.m // warp_shape.m` must be powers of 2.
  - `n`:
    - Must be powers of 2.
    - 16 bit activation: must be 64. 8 bit activation: must be 64/32. 4 bit activation: must be 64/32/16.
    - If use WGMMA, `block_shape.n // warp_shape.n` must be divisable by 4.
  - `k`:
    - Must be powers of 2.
    - Must be divisable by 32/64/128 for 16/8/4 bit activation, respectfully.


## DataType Config 🔔

See support matrix in `README.md`.



## Quant Param Config


- `has_input_scale`:
  - For 16-bit activation (`float16` / `bfloat16`), this must be `False`.
  - For 8-bit/4-bit activation, we enable input scale by default. But you can also disable it.

- `has_weight_scale`🔔: By default, we enable weight scale for all cases. But you can also disable it. For example, if you also want tensor-wise scale, you can set `has_weight_scale` to `False` and set `has_global_scale` to `True`.

- `input_scale_group_size`: 
  - `0`: channelwise scale (also known as token-wise scale)
  - `>0`: must be powers of 2. The minimal value is 8/16/32 for 16/8/4-bit activation + MMA, and 16/32/64 for 16/8/4-bit activation + WGMMA.

- `weight_scale_group_size`🔔: 
  - `0`: channelwise scale
  - `>0`: must be powers of 2. The minimal value is 8/16/32 for 16/8/4-bit activation + MMA, and 16/32/64 for 16/8/4-bit activation + WGMMA.

- `has_global_scale`🔔: Enable global scale or not. Note that this is not conflict with `has_weight_scale`, you can set both to `True` (for example, NVFP4).

- `has_dynamic_zero_point`🔔: Enable dynamic zero point or not.
  - Must be `False` if `has_weight_scale` is False.
  - Share the same group size with `weight_scale_group_size`.


## Scheduler Config

- `use_stream_k`: Enable stream-k or not. It is recommended to be enabled in most case. If you want batch invariance mode, disable it.


## Pipeline Config

- `num_stages`: Num stages for multi-stages pipeline. Higher value requires larger shared memory requirement.

- `use_warp_spec`: Enable warp specialization or not. Requires SM90+.

- `use_mbarrier`: Enable mbarrier for pipeline or not. Must be enabled for warp spec and/or tma. Can also be enabled if `use_cp_async` is True. Requires SM80+.

- `use_cp_async`: Enable `cp.async` or not. Requires SM80+.

- `use_tma`: Enable TMA or not. Requires SM90+.
  - `use_tma_a`, `use_tma_b`, `use_tma_c`, `use_tma_bs`, `use_tma_bzp`, `use_tma_bias`: You can also configure TMA for only a subset of tensors. Note that if any specific tensor is set to use TMA, the global `use_tma` flag must also be `True`. When `use_tma` is True, all options will default to `True` unless explicitly set to False or if they conflict with other configs.

## Epilogue Config

- `has_bias`: Enable bias fusion or not.

- `activation_type`: Humming supports fused activation. The following activation types are supported:
  - `sigmoid`, `tanh`, `relu`, `gelu`, `fastgelu`, `quickgelu`, `silu`: identity activation
  - `silu_glu`: glu activation.
  - `custom`: custom identity activation
  - `custom_glu`: custom glu activation
  - Note that `use_tma_c` must be `False` to use activation fusion.

- `custom_activation_func_impl`: CUDA C++ code of custom activation function.
  - If custom identity activation, the input is `float` type variable `a`, you should return a `float`. Example `return a * 0.5 + 0.123;`
  - If custom glu activation, the input is `float2` type variable `a`, you should return a `float`. Example `return tanhf(a.y * 0.5) * (1 + a.x);`


Notes for GLU activation (`silu_glu` and `custom_glu`)

1. If you use glu activation and stream k is enabled. The `outputs` tensor still need to have shape `(m, n)`, but after kernel execution, you can reshape it to `(m * 2, n // 2)` and only take the `(m, n // 2)`.
2. The weight `up_proj` and `gate_proj` must be interleaved, that is, `weight[..., 1::2]` should be `up_proj`, `weight[..., 1::2]` should be `gate_proj`.

## MoE Config

- `is_moe`🔔: 
  - `True`: MoE GEMM. Note that humming currently only support triton-style indexed gemm, cublas-style grouped gemm is not ready now.
  - `False`: Dense GEMM.

- `top_k`: Must greater than 0 if `is_moe` is `True`.

- `is_moe_down`: The behavior of `is_moe_down`:

| `is_moe_down` | `True` | `False` |
| - | - | - |
| Read Index | `sorted_token_ids` | `sorted_token_ids // top_k` |
| Write Index | `sorted_token_ids` | `sorted_token_ids` |
| Multiply Topk Weights | YES | NO |



## MMA Config

- `mma_type`🔔: `mma` or `wgmma`
- `use_f16_accum`: Use float16 accumulation or not. This can have better performance since it requires less registers and mma with f16 accumulation have double tflops than mma with f32 accumulation on some devices (SM75/SM89). But it would increase the risk of numerical overflow. The f16 accum requires:
  - `c_dtype` must be `float16` (note that `bfloat16` is not supported)
  - `a_dtype` must be `float16` or `float8e4m3`. If `float8e4m3`, `b_dtype` must be integer type or floating type with exponent bits <= 3
