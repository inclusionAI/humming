
# Humming

Humming is a high-performance, lightweight, and highly flexible JIT (Just-In-Time) compiled GEMM kernel library specifically designed for quantized inference.


## Key Features

- **High Flexibility**
    - Supports inference for any weight type under 8-bit across **FP16 / BF16 / FP8 / FP4 / INT8 / INT4** activations (provided the activation's dynamic range covers the weight type).
    - Supports various quantization strategies.
    - Supports various scale types (BF16, FP16, E4M3, E5M2, and UE8M0).
    - Supports both **Dense GEMM** and **MoE GEMM**.
- **High Compatibility**: supports all NVIDIA GPUs from **SM75+** (Turing architecture) and beyond.
- **High Performance**
    * Delivers State-of-the-Art (SOTA) throughput and efficiency across a wide range of computational scenarios.
- **Ultra-Lightweight**
    * Minimal dependencies: Requires only **PyTorch** and **NVCC**.
    * Compact footprint: The package size is less than **100KB**.


## Support Matrix

| Activation Type | Supported Devices | Supported Weight Types |
| :--- | :--- | :--- |
| **FP16** (e5m10) | SM75+ | • Symmetric INT1-8<br>• INT1-8 with dynamic zero point<br>• Arbitrary signed FP (kBits ≤ 8, kExp ≤ 5) |
| **BF16** (e8m7) | SM80+ | • Symmetric INT1-8<br>• INT1-8 with dynamic zero point<br>• Arbitrary signed FP (kBits ≤ 8) |
| **FP8** (e4m3) | SM89+ | • Symmetric INT1-5<br>• INT1-4 with dynamic zero point<br>• Arbitrary signed FP (kExp ≤ 4, kMan ≤ 3) |
| **FP8** (e5m2) | SM89+ | • Symmetric INT1-4<br>• INT1-3 with dynamic zero point<br>• Arbitrary signed FP (kExp ≤ 5, kMan ≤ 2) |
| **FP4** (e2m1) | SM120+ | • Symmetric INT1-3<br>• INT1-2 with dynamic zero point<br>• Arbitrary signed FP (kExp ≤ 2, kMan ≤ 1) |
| **INT8** | SM75+ | • Symmetric INT1-8<br>• INT1-7 with dynamic zero point |
| **INT4** | SM80+ | • Symmetric INT1-4<br>• INT1-3 with dynamic zero point |


## Getting Started


### Installation

```
pip install git+https://github.com/inclusionAI/humming.git
```


### Usage Example


```python
import torch

from humming import dtypes
from humming.layer import HummingLayer
from humming.utils.test import generate_random_inputs, generate_random_weight

layer = HummingLayer(
    shape_n=1024,
    shape_k=1024,
    a_dtype=dtypes.float16,
    b_dtype=dtypes.uint4,
    c_dtype=dtypes.float16,
    bs_dtype=dtypes.float16,
    weight_scale_group_size=128,
).cuda()


random_weight_data = generate_random_weight(
    n=layer.shape_n,
    k=layer.shape_k,
    group_size=layer.weight_scale_group_size,
    dtype=layer.b_dtype,
    scale_dtype=layer.bs_dtype,
)

_, weight_ref, weight, weight_scale, _, _ = random_weight_data
_, inputs_ref, inputs, _ = generate_random_inputs(1234, layer.shape_k, dtype=dtypes.float16)

# Tensors can be loaded simultaneously or sequentially.
# For MoE models, you have the flexibility to load only a specific expert
layer.load_weight(weight=weight, weight_scale=weight_scale)
# Run `layer.finish_load()` after all weights are loaded, this would do some preprocessing.
# Note that you shouldn't load weight again after `finish_load`
layer.finish_load()

# Currently, you need to manually input block_shape and warp_shape to run.
# Auto-tuning features is coming soon.
outputs = layer(inputs=inputs, block_shape=(64, 256, 64), warp_shape=(64, 64, 64))
outputs_ref = inputs_ref.matmul(weight_ref.T).to(torch.float16)
torch.testing.assert_close(outputs, outputs_ref, atol=0.1, rtol=0.01)
```

For more config options, see [Config Options](./docs/CONFIG.md).

For performance tuning example, see `examples` dir.

## Roadmap

- [ ] Technical Analysis
- [ ] Config Tuning
- [ ] Kernel Bench
- [ ] NVCC-free Runtime
- [ ] UMMA Support
- [ ] MMA with Block Scaling Support

## Acknowledgement

This project is highly inspired by

- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM/)
- [Marlin Kernel](https://github.com/IST-DASLab/marlin/) and [vLLM](https://github.com/vllm-project/vllm) Marlin Kernel
- [lmdeploy](https://github.com/InternLM/lmdeploy/) GEMM kernel
- [CUTLASS](https://github.com/nvidia/cutlass)
