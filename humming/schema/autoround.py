import dataclasses
from typing import Any

import torch

from humming.schema.awq import AWQWeightSchema
from humming.schema.base import BaseWeightSchema
from humming.schema.gptq import GPTQWeightSchema
from humming.schema.humming import HummingWeightSchema


@dataclasses.dataclass(kw_only=True)
class AutoRoundWeightSchema(BaseWeightSchema):
    quant_method: str = "auto-round"
    bits: int
    group_size: int
    sym: bool = True
    data_type: str = "int"
    packing_format: str = "auto_round:auto_gptq"
    act_bits: int = 16

    def __post_init__(self):
        fmt = self.packing_format

        # fp formats (fp8 / nvfp / mxfp) are rejected outright.
        is_fp = "fp" in self.data_type or any(tag in fmt for tag in ("fp8", "nvfp", "mxfp", "nv_fp", "mx_fp"))
        assert not is_fp and self.data_type == "int", (
            f"autoround fp quantization is not supported by humming "
            f"(data_type={self.data_type!r}, packing_format={fmt!r}); "
            "only weight-only int quantization is supported"
        )
        assert self.act_bits >= 16, (
            f"autoround act_bits={self.act_bits} (activation quantization) is not supported by humming"
        )

        if "awq" in fmt:
            self._delegate: BaseWeightSchema = AWQWeightSchema(
                bits=self.bits,
                group_size=self.group_size,
                zero_point=not self.sym,
            )
        elif "gptq" in fmt or "auto_round" in fmt:
            # Covers auto_gptq / gptqmodel (gptqv2) and the native auto_round format,
            # all of which serialize weights with the GPTQ packing layout.
            self._delegate = GPTQWeightSchema(
                bits=self.bits,
                group_size=self.group_size,
                sym=self.sym,
            )
        else:
            raise ValueError(f"unsupported autoround packing_format: {fmt!r}")

    def get_tensors_attrs(
        self,
        shape_n: int,
        shape_k: int,
        param_dtype: torch.dtype,
        num_experts: int | None = None,
        has_bias: bool = False,
        stack_size: int = 1,
    ) -> dict[str, dict[str, Any]]:
        return self._delegate.get_tensors_attrs(
            shape_n=shape_n,
            shape_k=shape_k,
            param_dtype=param_dtype,
            num_experts=num_experts,
            has_bias=has_bias,
            stack_size=stack_size,
        )

    def infer_shape(self, tensors: dict[str, torch.Tensor]) -> tuple[int, int, int | None, bool]:
        return self._delegate.infer_shape(tensors)

    def process_loaded_weight(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        return self._delegate.process_loaded_weight(tensor, name)

    def _convert_humming(
        self,
        tensors: dict[str, torch.Tensor],
        shape_n_stacks: list[int],
        shape_k_stacks: list[int],
        param_dtype: torch.dtype,
        num_experts: int | None = None,
    ) -> tuple[HummingWeightSchema, dict[str, torch.Tensor]]:
        return self._delegate.convert_humming(
            tensors=tensors,
            shape_n_stacks=shape_n_stacks,
            shape_k_stacks=shape_k_stacks,
            param_dtype=param_dtype,
            num_experts=num_experts,
        )
