import dataclasses
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from filelock import FileLock

import humming.utils.jit as jit_utils
from humming import dtypes, ops
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType, TuningConfig
from humming.config.config import _cuda_compiler_version
from humming.config.mma import get_default_mma_type, get_mxmma_compiler_error
from humming.device import current_device
from humming.kernel.humming import HummingKernel
from humming.schema import HummingWeightSchema
from humming.testing.data import (
    generate_moe_tensors,
    generate_random_tensor,
    generate_random_topk_ids,
)
from humming.testing.tuning import (
    NUM_SAMPLED_TUNING_CONFIGS,
    create_tuning_config,
    generate_heuristics_configs,
    sample_test_tuning_configs,
)
from humming.transform import transform_humming_tensors

_DEFAULT_SHAPE_MS = (1, 30, 257, 1024, 4090)
TEST_TUNING_SOURCE_ENV = "HUMMING_TEST_TUNING_SOURCE"
NUMERICAL_ERROR_LOG_ENV = "HUMMING_TEST_NUMERICAL_ERROR_LOG"


def _get_sample_size(tuning_source: str) -> int | None:
    match = re.fullmatch(r"sampled([1-9][0-9]*)?", tuning_source)
    if match is None:
        return None
    return int(match[1]) if match[1] else NUM_SAMPLED_TUNING_CONFIGS


def _get_test_tuning_sources() -> tuple[str, ...]:
    value = os.environ.get(TEST_TUNING_SOURCE_ENV, "heuristic+batch_invariant")
    sources = tuple(dict.fromkeys(source.strip() for source in value.split("+")))
    for source in sources:
        if source not in ("heuristic", "batch_invariant") and _get_sample_size(source) is None:
            raise ValueError(f"invalid tuning source: {value}")
    return sources


def assert_kernel_test_shape_coverage(
    results: list["KernelTestResult"],
    shape_ms: list[int] | tuple[int, ...] | None = None,
) -> None:
    if shape_ms is None:
        shape_ms = _DEFAULT_SHAPE_MS
    tuning_sources = _get_test_tuning_sources()
    assert {result.tuning_source for result in results} == set(tuning_sources)
    for tuning_source in tuning_sources:
        counts = Counter(result.shape_m for result in results if result.tuning_source == tuning_source)
        assert set(counts) == set(shape_ms)
        assert len(set(counts.values())) == 1
        expected_minimum = _get_sample_size(tuning_source) or 1
        assert next(iter(counts.values())) >= expected_minimum


@dataclasses.dataclass(frozen=True, kw_only=True)
class KernelTestCase:
    name: str
    layer_config: LayerConfig
    compute_config: ComputeConfig
    top_k: int = 1
    expert_max_tokens: int | None = None
    seed: int = 0
    input_std_scale: float = 1.0
    weight_std_scale: float = 1.0
    bias_std_scale: float = 1.0
    rtol: float = 0.01
    atol: float = 0.05

    def __str__(self) -> str:
        return self.name

    def effective_shape_m(self, shape_m: int) -> int:
        if self.compute_config.gemm_type in [GemmType.INDEXED, GemmType.GROUPED_CONTIGUOUS]:
            return shape_m * self.top_k
        if self.compute_config.gemm_type == GemmType.GROUPED_MASKED:
            expert_max_tokens = self.resolve_expert_max_tokens(shape_m)
            return self.layer_config.num_experts * expert_max_tokens
        return shape_m

    def resolve_expert_max_tokens(self, shape_m: int) -> int:
        if self.expert_max_tokens is not None:
            return self.expert_max_tokens
        return shape_m * self.top_k

    @property
    def uses_m_major_input_scale(self) -> bool:
        return self.compute_config.use_m_major_input_scale


@dataclasses.dataclass(frozen=True, kw_only=True)
class KernelTestResult:
    tuning_source: str
    shape_m: int
    tuning_config: TuningConfig
    tuning_values: dict
    outputs: torch.Tensor
    outputs_ref: torch.Tensor


class KernelTestRunner:
    def __init__(self, test_case: KernelTestCase):
        self.test_case = test_case
        self.layer_config = test_case.layer_config
        self.compute_config = test_case.compute_config
        self.tuning_source = "heuristic"
        self.device = torch.cuda.current_device()
        self.weight_ref: torch.Tensor
        self.bias_ref: torch.Tensor | None = None
        self.kernel_tensors: dict[str, torch.Tensor]
        self.prepare_weight()

    def prepare_kernels(
        self, shape_ms: tuple[int, ...]
    ) -> list[tuple[str, ComputeConfig, dict[int, list[tuple[torch.Tensor, dict, int]]]]]:
        if self.layer_config.mxmma_supported:
            compiler_version = _cuda_compiler_version(HummingKernel._get_compiler())
            compiler_error = get_mxmma_compiler_error(self.layer_config, compiler_version)
            if compiler_error is not None:
                import pytest

                pytest.skip(compiler_error)

        tuning_sources = _get_test_tuning_sources()
        if any(_get_sample_size(source) is None for source in tuning_sources):
            if get_default_mma_type(self.layer_config) == MmaType.WGMMA:
                min_warp_shape_n = 32 if self.layer_config.a_dtype.num_bits == 16 else 16
                if self.layer_config.shape_n % (min_warp_shape_n * 4):
                    import pytest

                    pytest.skip("shape_n cannot form four WGMMA warp tiles")

        groups = []
        compile_configs = []
        effective_shape_ms = [self.test_case.effective_shape_m(shape_m) for shape_m in shape_ms]
        for tuning_source in tuning_sources:
            compute_config = self.test_case.compute_config
            if tuning_source == "batch_invariant":
                compute_config = dataclasses.replace(compute_config, use_batch_invariant=True)
            sample_size = _get_sample_size(tuning_source)
            if sample_size is not None:
                tuning_configs = sample_test_tuning_configs(
                    self.layer_config, compute_config, sample_size=sample_size
                )
            else:
                tuning_configs = generate_heuristics_configs(
                    self.layer_config, compute_config, effective_shape_ms
                )
            if not tuning_configs:
                import pytest

                pytest.skip(f"no legal tuning configs for {tuning_source}")
            groups.append((tuning_source, compute_config, tuning_configs))
            compile_configs.extend(
                (0, 1 << 30, values | {"use_batch_invariant": compute_config.use_batch_invariant})
                for values in tuning_configs
            )

        kernel_configs = HummingKernel.prepare_kernels(
            self.layer_config.to_str(),
            self.compute_config.to_str(),
            compile_configs,
            device=self.device,
        ).reshape(-1, 4)
        assert kernel_configs.shape[0] == len(compile_configs)
        for kernel_id in set(kernel_configs[:, 2].tolist()):
            HummingKernel._id2kernel[kernel_id].assert_smem_size_matches_estimate()

        prepared_groups = []
        offset = 0
        for tuning_source, compute_config, tuning_configs in groups:
            group_configs = kernel_configs[offset : offset + len(tuning_configs)]
            kernels = [
                (kernel_config, values, index)
                for index, (kernel_config, values) in enumerate(
                    zip(group_configs, tuning_configs, strict=True)
                )
            ]
            if _get_sample_size(tuning_source) is not None:
                shape_kernels = dict.fromkeys(shape_ms, kernels)
            else:
                shape_kernels = {shape_m: [kernel] for shape_m, kernel in zip(shape_ms, kernels, strict=True)}
            prepared_groups.append((tuning_source, compute_config, shape_kernels))
            offset += len(tuning_configs)
        return prepared_groups

    def prepare_weight(self) -> None:
        torch.manual_seed(self.test_case.seed + 123)
        config = self.layer_config
        shape_n = config.shape_n - config.pad_shape_n
        shape_k = config.shape_k - config.pad_shape_k
        shape = (shape_n, shape_k)
        if config.num_experts:
            shape = (config.num_experts,) + shape
        weight_orig = generate_random_tensor(
            shape,
            dtype=config.param_dtype,
            std_scale=self.test_case.weight_std_scale,
            group_size=config.weight_scale_group_size,
            device=self.device,
        )

        schema = HummingWeightSchema(
            b_dtype=config.b_dtype,
            bs_dtype=config.bs_dtype,
            weight_scale_group_size=config.weight_scale_group_size,
            weight_scale_group_size_n=config.weight_scale_group_size_n,
            weight_scale_type=config.weight_scale_type,
            weight_scale_2_type=config.weight_scale_2_type,
            has_zero_point=config.has_zero_point,
            is_fp_zero_point=config.is_fp_zero_point,
        )
        uses_unsigned_weight_scale = config.use_block_scaled_mma and config.is_group_weight_scale
        allow_negative_scale = not uses_unsigned_weight_scale
        dtype = config.param_dtype
        tensors = schema.quant_tensor(weight_orig, schema, dtype, allow_negative_scale=allow_negative_scale)
        self.weight_ref = schema.dequant_tensors(tensors)
        if config.has_bias:
            bias_shape = (shape_n,) if not config.num_experts else (config.num_experts, shape_n)
            self.bias_ref = generate_random_tensor(
                bias_shape,
                dtype=config.param_dtype,
                std_scale=self.test_case.bias_std_scale,
                device=self.device,
            )
            tensors["bias"] = self.bias_ref
        self.kernel_tensors = transform_humming_tensors(config, tensors)

    def prepare_inputs(
        self,
        inputs_orig: torch.Tensor,
        static_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        config = self.layer_config
        shape_k = config.shape_k - config.pad_shape_k
        assert inputs_orig.shape[1] == shape_k

        if config.a_dtype.num_bits == 16:
            inputs = inputs_orig.to(dtypes.torch_dtype_map[config.a_dtype])
            return inputs.float(), inputs, None, None

        if config.input_quant_mode.has_tensor_scale and static_scale is None:
            target_maximum = 448.0 if config.a_dtype == dtypes.float8e4m3 else 127.0
            static_scale = (inputs_orig.abs().amax() / target_maximum).reshape(1).float()

        def process(m_major_scale: bool = False):
            result = ops.process_input(
                inputs_orig,
                quant_mode=config.input_quant_mode.value,
                quant_dtype=str(config.a_dtype),
                quant_group_size=config.input_scale_group_size or None,
                group_scale_dtype=str(config.as_dtype),
                use_m_major_input_scale=m_major_scale,
                token_scales=static_scale,
            )
            return result

        inputs, group_scale_ref, token_scale_ref = process()
        use_m_major_input_layout = self.test_case.uses_m_major_input_scale
        if use_m_major_input_layout:
            _, major_groups, major_tokens = process(m_major_scale=True)
            input_scale = major_groups if config.input_quant_mode.has_group_scale else major_tokens
            input_scale_2 = major_tokens if config.input_quant_mode.has_secondary_scale else None
        else:
            input_scale = group_scale_ref if group_scale_ref is not None else token_scale_ref
            input_scale_2 = token_scale_ref if config.input_quant_mode.has_secondary_scale else None

        if config.input_quant_mode.has_tensor_scale:
            tensor_scale = static_scale
            if config.input_quant_mode.has_secondary_scale:
                input_scale_2 = tensor_scale
            else:
                input_scale = tensor_scale

        if config.a_dtype.num_bits == 4:
            codes = ops.unpack_weight(inputs.view(torch.int32), 4)
            if config.a_dtype == dtypes.int4:
                dequant_inputs = ((codes + 8) & 0xF) - 8
            elif config.a_dtype == dtypes.float4e0m3:
                magnitude = (codes & 0x7).float()
                dequant_inputs = torch.where((codes & 0x8) != 0, -magnitude, magnitude)
            else:
                dequant_inputs = ops.dequant_weight(
                    codes,
                    config.a_dtype.exponent_bits,
                    config.a_dtype.mantissa_bits,
                    config.a_dtype.is_signed,
                )
        elif config.a_dtype == dtypes.float8e3m4:
            codes = inputs.view(torch.uint8).to(torch.int32).contiguous()
            dequant_inputs = ops.dequant_weight(codes, 3, 4, True)
        else:
            dequant_inputs = inputs.float()

        if group_scale_ref is not None:
            scale_ref = group_scale_ref.view(dtypes.torch_dtype_map[config.as_dtype])
            scale_ref = scale_ref[:, : shape_k // config.input_scale_group_size].float()
            if token_scale_ref is not None:
                scale_ref = scale_ref * token_scale_ref.float().reshape(-1, 1)
            group_size = config.input_scale_group_size
        else:
            assert token_scale_ref is not None
            scale_ref = token_scale_ref.float().reshape(-1, 1)
            group_size = shape_k
        inputs_ref = dequant_inputs.float() * scale_ref.repeat_interleave(group_size, 1)
        return inputs_ref, inputs, input_scale, input_scale_2

    def _sample_inputs(
        self,
        base_inputs: torch.Tensor,
        base_topk_ids: torch.Tensor | None,
        shape_m: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        permute_idx = torch.randperm(base_inputs.shape[0], device=base_inputs.device)[:shape_m]
        unpermute_idx = torch.argsort(permute_idx)
        inputs = base_inputs[permute_idx].contiguous()
        topk_ids = None
        if base_topk_ids is not None:
            topk_ids = base_topk_ids[permute_idx].contiguous()
        return inputs, topk_ids, permute_idx, unpermute_idx

    def _prepare_problem(
        self,
        shape_m: int,
        inputs: torch.Tensor,
        topk_ids: torch.Tensor | None,
        block_shape_m: int,
    ) -> tuple[torch.Tensor, dict, torch.Tensor]:
        config = self.layer_config
        gemm_type = self.compute_config.gemm_type
        assert (config.num_experts == 0) == (gemm_type == GemmType.DENSE)

        if gemm_type == GemmType.DENSE:
            output_ids = torch.arange(shape_m, device=inputs.device)
            launch_tensors = {
                "sorted_ids": None,
                "expert_ids": None,
                "num_tokens_padded": None,
                "expert_layout": None,
            }
            return inputs, launch_tensors, output_ids

        assert topk_ids is not None
        expert_alignment = 1
        moe_tensors = generate_moe_tensors(
            topk_ids,
            config.num_experts,
            gemm_type=gemm_type,
            block_size_config=block_shape_m,
            expert_max_tokens=self.test_case.resolve_expert_max_tokens(shape_m),
            expert_alignment=expert_alignment,
        )
        _, expert_layout, sorted_ids, expert_ids, num_tokens_padded = moe_tensors

        launch_tensors = {
            "sorted_ids": sorted_ids,
            "expert_ids": expert_ids,
            "num_tokens_padded": num_tokens_padded,
            "expert_layout": expert_layout,
        }
        if gemm_type == GemmType.INDEXED:
            output_ids = torch.arange(topk_ids.numel(), device=inputs.device)
            return inputs, launch_tensors, output_ids

        flat_expert_ids = topk_ids.reshape(-1)
        flat_inputs = inputs.repeat_interleave(self.test_case.top_k, dim=0)
        if gemm_type == GemmType.GROUPED_CONTIGUOUS:
            assert expert_layout is not None
            expert_offsets = expert_layout[:-1]
            grouped_shape_m = expert_layout[-1].item()
        else:
            assert gemm_type == GemmType.GROUPED_MASKED
            expert_max_tokens = self.test_case.resolve_expert_max_tokens(shape_m)
            expert_offsets = torch.arange(config.num_experts, device=inputs.device)
            expert_offsets *= expert_max_tokens
            grouped_shape_m = self.test_case.effective_shape_m(shape_m)

        grouped_inputs = inputs[:1].expand(grouped_shape_m, -1).clone()
        output_ids = torch.empty(topk_ids.numel(), dtype=torch.long, device=inputs.device)
        for expert_id in range(config.num_experts):
            token_ids = torch.where(flat_expert_ids == expert_id)[0]
            grouped_ids = expert_offsets[expert_id] + torch.arange(
                token_ids.numel(),
                device=inputs.device,
            )
            grouped_inputs[grouped_ids] = flat_inputs[token_ids]
            output_ids[token_ids] = grouped_ids
        return grouped_inputs, launch_tensors, output_ids

    def _matmul_reference(
        self,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.compute_config.use_f16_accum:
            values = inputs.matmul(weight.T)
        else:
            previous = torch.backends.cuda.matmul.allow_fp16_accumulation
            # PPU's FP16-accumulation BLAS path returns zeros for GEMV (M=1).
            # Keep half inputs, but use its working FP32 accumulation reference.
            torch.backends.cuda.matmul.allow_fp16_accumulation = not current_device.is_ppu
            try:
                values = inputs.half().matmul(weight.half().T)
            finally:
                torch.backends.cuda.matmul.allow_fp16_accumulation = previous
        if bias is not None:
            values += bias
        return values

    def make_reference(
        self,
        inputs_ref: torch.Tensor,
        topk_ids: torch.Tensor | None,
        output_ids: torch.Tensor,
    ) -> torch.Tensor:
        config = self.layer_config
        gemm_type = self.compute_config.gemm_type
        output_dtype = dtypes.torch_dtype_map[config.c_dtype]

        if gemm_type == GemmType.DENSE:
            values = self._matmul_reference(inputs_ref, self.weight_ref, self.bias_ref)
            return values.to(output_dtype)

        assert topk_ids is not None
        flat_expert_ids = topk_ids.reshape(-1)
        if gemm_type == GemmType.INDEXED:
            token_inputs = inputs_ref.repeat_interleave(topk_ids.shape[1], dim=0)
        else:
            token_inputs = inputs_ref[output_ids]

        output_shape = (topk_ids.numel(), config.shape_n - config.pad_shape_n)
        outputs_ref = torch.zeros(output_shape, dtype=output_dtype, device=inputs_ref.device)
        for expert_id in range(config.num_experts):
            token_ids = torch.where(flat_expert_ids == expert_id)[0]
            if not token_ids.numel():
                continue
            bias_ref = None if self.bias_ref is None else self.bias_ref[expert_id]
            weight_ref = self.weight_ref[expert_id]
            values = self._matmul_reference(token_inputs[token_ids], weight_ref, bias_ref)
            outputs_ref[token_ids] = values.to(output_dtype)
        return outputs_ref

    def _launch_kernel(
        self,
        shape_m: int,
        launch_tensors: dict,
        kernel_config: torch.Tensor,
    ) -> torch.Tensor:
        inputs = launch_tensors["inputs"]
        output_shape_m = self.test_case.effective_shape_m(shape_m)
        if self.compute_config.gemm_type in [GemmType.GROUPED_CONTIGUOUS, GemmType.GROUPED_MASKED]:
            output_shape_m = inputs.shape[0]

        outputs = torch.zeros(
            (output_shape_m, self.layer_config.shape_n - self.layer_config.pad_shape_n),
            dtype=dtypes.torch_dtype_map[self.layer_config.c_dtype],
            device=inputs.device,
        )
        return ops.launch_kernel(
            configs=kernel_config,
            outputs=outputs,
            locks=torch.zeros((1024,), dtype=torch.int32, device=inputs.device),
            top_k=self.test_case.top_k,
            **launch_tensors,
            **self.kernel_tensors,
        )

    def _assert_close(self, outputs: torch.Tensor, outputs_ref: torch.Tensor, use_stream_k: bool) -> None:
        rtol, atol = self.test_case.rtol, self.test_case.atol
        try:
            torch.testing.assert_close(outputs, outputs_ref, rtol=rtol, atol=atol)
        except AssertionError:
            if outputs.dtype != torch.bfloat16 or not use_stream_k:
                raise
            # Keep rtol unchanged and validate tensor metadata and non-finite values.
            torch.testing.assert_close(outputs, outputs_ref, rtol=rtol, atol=2 * atol)
            outliers = ~torch.isclose(outputs, outputs_ref, rtol=rtol, atol=atol).reshape(-1)
            # Allow at most 0.01% isolated outliers, with a minimum allowance of two.
            max_outliers = max(2, outputs.numel() // 10000)
            if outliers.sum().item() > max_outliers:
                raise
            if torch.any(outliers[:-1] & outliers[1:]):
                raise

    def _run_kernel(
        self,
        shape_m: int,
        launch_tensors: dict,
        kernel: tuple[torch.Tensor, dict, int],
        outputs_ref: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> KernelTestResult:
        kernel_config, tuning_values, tuning_index = kernel
        try:
            outputs = self._launch_kernel(shape_m, launch_tensors, kernel_config)[output_ids]
        except Exception as error:
            raise RuntimeError(
                f"kernel launch failed for shape_m={shape_m}, "
                f"tuning_index={tuning_index}, tuning_config={tuning_values}"
            ) from error
        try:
            compiled_kernel = HummingKernel._id2kernel[int(kernel_config[2])]
            self._assert_close(outputs, outputs_ref, compiled_kernel.use_stream_k)
        except AssertionError as error:
            self._record_numerical_error(error, shape_m, tuning_values, tuning_index)
            raise AssertionError(
                f"kernel result mismatch for shape_m={shape_m}, "
                f"tuning_index={tuning_index}, tuning_config={tuning_values}"
            ) from error
        except Exception as error:
            raise RuntimeError(
                f"kernel result check failed for shape_m={shape_m}, "
                f"tuning_index={tuning_index}, tuning_config={tuning_values}"
            ) from error

        return KernelTestResult(
            tuning_source=self.tuning_source,
            shape_m=shape_m,
            tuning_config=create_tuning_config(tuning_values),
            tuning_values=tuning_values,
            outputs=outputs,
            outputs_ref=outputs_ref,
        )

    def run(
        self,
        shape_ms: list[int] | tuple[int, ...] | None = None,
    ) -> list[KernelTestResult]:
        if shape_ms is None:
            shape_ms = _DEFAULT_SHAPE_MS
        shape_ms = tuple(dict.fromkeys(shape_ms))
        if not shape_ms:
            return []

        prepared_groups = self.prepare_kernels(shape_ms)
        results = []
        original_compute_config = self.compute_config
        original_tuning_source = self.tuning_source
        try:
            for tuning_source, compute_config, kernels in prepared_groups:
                self.tuning_source = tuning_source
                self.compute_config = compute_config
                results.extend(self._run_prepared_kernels(shape_ms, kernels))
        finally:
            self.compute_config = original_compute_config
            self.tuning_source = original_tuning_source
        return results

    def _run_prepared_kernels(
        self,
        shape_ms: tuple[int, ...],
        kernels: dict[int, list[tuple[torch.Tensor, dict, int]]],
    ) -> list[KernelTestResult]:
        max_shape_m = max(shape_ms)
        shape_k = self.layer_config.shape_k - self.layer_config.pad_shape_k
        output_top_k = 1 if self.compute_config.gemm_type == GemmType.DENSE else self.test_case.top_k
        torch.manual_seed(self.test_case.seed + max_shape_m)
        base_inputs = generate_random_tensor(
            (max_shape_m, shape_k),
            dtype=self.layer_config.param_dtype,
            std_scale=self.test_case.input_std_scale,
            group_size=self.layer_config.input_scale_group_size,
            device=self.device,
        )
        static_scale = None
        if self.layer_config.input_quant_mode.has_tensor_scale:
            target_maximum = 448.0 if self.layer_config.a_dtype == dtypes.float8e4m3 else 127.0
            static_scale = (base_inputs.abs().amax() / target_maximum).reshape(1).float()
        base_topk_ids = None
        if self.compute_config.gemm_type != GemmType.DENSE:
            num_experts, top_k = self.layer_config.num_experts, self.test_case.top_k
            base_topk_ids = generate_random_topk_ids(max_shape_m, num_experts, top_k, device=self.device)

        max_kernel = kernels[max_shape_m][0]
        base_outputs = None
        if self.compute_config.use_batch_invariant:
            block_shape_m = max_kernel[1]["block_shape"][0]
            base_problem = self._prepare_problem(max_shape_m, base_inputs, base_topk_ids, block_shape_m)
            base_problem_inputs, base_launch_tensors, base_output_ids = base_problem
            _, inputs, input_scale, input_scale_2 = self.prepare_inputs(base_problem_inputs, static_scale)
            base_launch_tensors |= {
                "inputs": inputs,
                "input_scale": input_scale,
                "input_scale_2": input_scale_2,
            }
            base_outputs = self._launch_kernel(max_shape_m, base_launch_tensors, max_kernel[0])
            base_outputs = base_outputs[base_output_ids].view(max_shape_m, output_top_k, -1)

        results = []
        for shape_m in shape_ms:
            torch.manual_seed(self.test_case.seed + shape_m)
            sampled_inputs = self._sample_inputs(base_inputs, base_topk_ids, shape_m)
            inputs, topk_ids, permute_idx, unpermute_idx = sampled_inputs
            test_kernel = kernels[shape_m][0]
            moe_block_size = test_kernel[1]["block_shape"][0]
            problem = self._prepare_problem(shape_m, inputs, topk_ids, moe_block_size)
            problem_inputs, launch_tensors, output_ids = problem
            inputs_ref, inputs, input_scale, input_scale_2 = self.prepare_inputs(problem_inputs, static_scale)
            launch_tensors |= {
                "inputs": inputs,
                "input_scale": input_scale,
                "input_scale_2": input_scale_2,
            }
            outputs_ref = self.make_reference(inputs_ref, topk_ids, output_ids)

            for kernel in kernels[shape_m]:
                kernel_launch_tensors = launch_tensors
                block_shape_m = kernel[1]["block_shape"][0]
                if self.compute_config.gemm_type == GemmType.INDEXED and block_shape_m != moe_block_size:
                    kernel_problem = self._prepare_problem(shape_m, problem_inputs, topk_ids, block_shape_m)
                    _, kernel_launch_tensors, kernel_output_ids = kernel_problem
                    assert torch.equal(kernel_output_ids, output_ids)
                    kernel_launch_tensors |= {
                        "inputs": inputs,
                        "input_scale": input_scale,
                        "input_scale_2": input_scale_2,
                    }
                result = self._run_kernel(shape_m, kernel_launch_tensors, kernel, outputs_ref, output_ids)

                if base_outputs is not None:
                    outputs = result.outputs.view(shape_m, output_top_k, -1)[unpermute_idx]
                    outputs0 = base_outputs[permute_idx.sort().values]
                    assert torch.equal(
                        outputs.contiguous().view(torch.uint8),
                        outputs0.contiguous().view(torch.uint8),
                    ), f"batch-invariant shape_m={shape_m}, tuning_index={kernel[2]}"
                results.append(result)
        return results

    def _record_numerical_error(
        self,
        error: AssertionError,
        shape_m: int,
        tuning_values: dict,
        tuning_index: int,
    ) -> None:
        log_path = os.environ.get(NUMERICAL_ERROR_LOG_ENV)
        if not log_path:
            return

        path = Path(log_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pytest_test": os.environ.get("PYTEST_CURRENT_TEST"),
            "test_case": dataclasses.asdict(self.test_case),
            "shape_m": shape_m,
            "tuning_source": self.tuning_source,
            "tuning_index": tuning_index,
            "tuning_config": tuning_values,
            "device": current_device.name,
            "error": str(error),
        }
        line = json.dumps(record, default=str, sort_keys=True)
        lock_name = "numerical-errors-" + jit_utils.hash_to_hex(path.as_posix())
        with FileLock(jit_utils.get_humming_lock_filename(lock_name)):
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    def failure_details(self, test_case: KernelTestCase, tuning_values: dict) -> str:
        return (
            f"\ncase={test_case}"
            f"\nseed={test_case.seed}"
            f"\nlayer_config={self.layer_config.to_str()}"
            f"\ncompute_config={self.compute_config.to_str()}"
            f"\ntuning_config={json.dumps(tuning_values)}"
            f"\ndevice={current_device.name}"
        )
