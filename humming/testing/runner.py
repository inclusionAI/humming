import dataclasses
import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import torch
from filelock import FileLock

import humming.utils.jit as jit_utils
from humming import dtypes, ops
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType, TuningConfig
from humming.device import current_device
from humming.kernel.humming import HummingKernel
from humming.schema import HummingWeightSchema
from humming.testing.data import (
    generate_moe_tensors,
    generate_random_tensor,
    generate_random_topk_ids,
)
from humming.testing.tuning import (
    create_tuning_config,
    generate_heuristics_configs,
    sample_test_tuning_configs,
)
from humming.transform import transform_humming_tensors

_DEFAULT_SHAPE_MS = (1, 17, 64, 257, 1024, 4096)
TEST_TUNING_SOURCE_ENV = "HUMMING_TEST_TUNING_SOURCE"
NUMERICAL_ERROR_LOG_ENV = "HUMMING_TEST_NUMERICAL_ERROR_LOG"
M_MAJOR_INPUT_SCALE_M_ALIGNMENT = 4


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def assert_kernel_test_shape_coverage(
    results: list["KernelTestResult"],
    shape_ms: list[int] | tuple[int, ...] | None = None,
) -> None:
    if shape_ms is None:
        shape_ms = _DEFAULT_SHAPE_MS
    counts = Counter(result.shape_m for result in results)
    assert set(counts) == set(shape_ms)
    assert len(set(counts.values())) == 1
    expected_minimum = 100 if os.environ.get(TEST_TUNING_SOURCE_ENV) == "sampled" else 1
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
        expert_max_tokens = shape_m * self.top_k
        if self.uses_m_major_input_scale:
            expert_max_tokens = _round_up(expert_max_tokens, M_MAJOR_INPUT_SCALE_M_ALIGNMENT)
        return expert_max_tokens

    @property
    def uses_m_major_input_scale(self) -> bool:
        return self.compute_config.use_m_major_input_scale or (
            self.layer_config.has_input_scale
            and self.layer_config.input_scale_group_size == 0
            and self.layer_config.mma_type != MmaType.MXMMA
        )


@dataclasses.dataclass(frozen=True, kw_only=True)
class KernelTestResult:
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
        self.device = torch.cuda.current_device()
        self.weight_ref: torch.Tensor
        self.bias_ref: torch.Tensor | None = None
        self.kernel_tensors: dict[str, torch.Tensor]
        self.prepare_weight()

    def prepare_kernels(self, shape_ms: tuple[int, ...]) -> dict[int, list[tuple[torch.Tensor, dict, int]]]:
        tuning_source = os.environ.get(TEST_TUNING_SOURCE_ENV, "heuristic")
        if tuning_source == "batch_invariant":
            self.compute_config = dataclasses.replace(self.compute_config, use_batch_invariant=True)

        if self.layer_config.mma_type == MmaType.WGMMA:
            min_warp_shape_n = (
                32
                if self.layer_config.a_dtype.num_bits == 16 or self.layer_config.use_packed_k_layout
                else 16
            )
            if self.layer_config.shape_n % (min_warp_shape_n * 4):
                import pytest

                pytest.skip("shape_n cannot form four WGMMA warp tiles")

        if tuning_source == "sampled":
            tuning_configs = sample_test_tuning_configs(self.layer_config, self.compute_config)
        elif tuning_source in ("heuristic", "batch_invariant"):
            effective_shape_ms = [self.test_case.effective_shape_m(shape_m) for shape_m in shape_ms]
            tuning_configs = generate_heuristics_configs(
                self.layer_config,
                self.compute_config,
                effective_shape_ms,
            )
        else:
            raise ValueError(f"invalid tuning source: {tuning_source}")

        if not tuning_configs:
            import pytest

            pytest.skip("no legal tuning configs for this layer and compute config")

        kernel_configs = HummingKernel.prepare_kernels(
            self.layer_config.to_str(),
            self.compute_config.to_str(),
            [(0, 1 << 30, values) for values in tuning_configs],
            device=self.device,
        ).reshape(-1, 4)
        assert kernel_configs.shape[0] == len(tuning_configs)
        for kernel_id in set(kernel_configs[:, 2].tolist()):
            HummingKernel._id2kernel[kernel_id].assert_smem_size_matches_estimate()

        enum_iter_objs = enumerate(zip(kernel_configs, tuning_configs, strict=True))
        kernels = [(*values, index) for index, values in enum_iter_objs]
        if tuning_source == "sampled":
            return dict.fromkeys(shape_ms, kernels)
        return {shape_m: [kernel] for shape_m, kernel in zip(shape_ms, kernels, strict=True)}

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
        allow_negative_scale = config.mma_type != MmaType.MXMMA
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        config = self.layer_config
        shape_k = config.shape_k - config.pad_shape_k
        assert inputs_orig.shape[1] == shape_k

        if config.a_dtype.num_bits == 16:
            inputs = inputs_orig.to(dtypes.torch_dtype_map[config.a_dtype])
            return inputs.float(), inputs, None

        def process(m_major_scale: bool = False):
            scale_layout = "row_major"
            if m_major_scale and config.input_scale_group_size > 0:
                scale_layout = "m_major"
                if str(config.as_dtype) in ("float8e4m3", "float8e8m0"):
                    scale_layout = "mx_packed"
            quant_mode = "dynamic_group" if config.input_scale_group_size > 0 else "dynamic_token"
            result = ops.process_input(
                inputs_orig,
                quant_mode=quant_mode,
                quant_dtype=str(config.a_dtype),
                quant_group_size=config.input_scale_group_size or None,
                group_scale_dtype=str(config.as_dtype),
                group_scale_layout=scale_layout,
            )
            scale = result[1]
            if scale is None:
                assert result[2] is not None
                scale = result[2].unsqueeze(-1)
            return result[0], scale

        inputs, scale_ref = process()
        use_m_major_input_layout = self.test_case.uses_m_major_input_scale and (
            config.input_scale_group_size > 0 or config.mma_type == MmaType.MXMMA
        )
        if use_m_major_input_layout:
            _, input_scale = process(m_major_scale=True)
        elif config.mma_type == MmaType.MXMMA and config.input_scale_group_size > 0:
            input_scale = scale_ref.view(torch.int32).contiguous()
        else:
            input_scale = scale_ref

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

        group_size = config.input_scale_group_size or shape_k
        inputs_ref = dequant_inputs.float() * scale_ref.float().repeat_interleave(group_size, 1)
        return inputs_ref, inputs, input_scale

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
        expert_max_tokens = None
        if self.test_case.uses_m_major_input_scale:
            if gemm_type == GemmType.GROUPED_MASKED:
                expert_max_tokens = self.test_case.resolve_expert_max_tokens(shape_m)
                if expert_max_tokens % M_MAJOR_INPUT_SCALE_M_ALIGNMENT:
                    raise RuntimeError("input-scale M dimension 16-byte aligned requirement is not met")
            if gemm_type == GemmType.GROUPED_CONTIGUOUS:
                expert_alignment = M_MAJOR_INPUT_SCALE_M_ALIGNMENT

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
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
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
            torch.testing.assert_close(
                outputs,
                outputs_ref,
                rtol=self.test_case.rtol,
                atol=self.test_case.atol,
            )
        except AssertionError as error:
            self._record_numerical_error(error, shape_m, tuning_values, tuning_index)
        except Exception as error:
            raise RuntimeError(
                f"kernel result check failed for shape_m={shape_m}, "
                f"tuning_index={tuning_index}, tuning_config={tuning_values}"
            ) from error

        return KernelTestResult(
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

        kernels = self.prepare_kernels(shape_ms)
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
            _, inputs, input_scale = self.prepare_inputs(base_problem_inputs)
            base_launch_tensors |= {"inputs": inputs, "input_scale": input_scale}
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
            inputs_ref, inputs, input_scale = self.prepare_inputs(problem_inputs)
            launch_tensors |= {"inputs": inputs, "input_scale": input_scale}
            outputs_ref = self.make_reference(inputs_ref, topk_ids, output_ids)

            for kernel in kernels[shape_m]:
                kernel_launch_tensors = launch_tensors
                block_shape_m = kernel[1]["block_shape"][0]
                if self.compute_config.gemm_type == GemmType.INDEXED and block_shape_m != moe_block_size:
                    kernel_problem = self._prepare_problem(shape_m, problem_inputs, topk_ids, block_shape_m)
                    _, kernel_launch_tensors, kernel_output_ids = kernel_problem
                    assert torch.equal(kernel_output_ids, output_ids)
                    kernel_launch_tensors |= {"inputs": inputs, "input_scale": input_scale}

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
            "timestamp": datetime.now(UTC).isoformat(),
            "pytest_test": os.environ.get("PYTEST_CURRENT_TEST"),
            "test_case": dataclasses.asdict(self.test_case),
            "shape_m": shape_m,
            "tuning_source": os.environ.get(TEST_TUNING_SOURCE_ENV, "heuristic"),
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
