import torch
from humming import dtypes
from typing import Optional
from humming.kernel.humming import HummingKernel
from humming.kernel.pack_weight import PackWeightKernel
from humming.utils.weight import (
    quantize_weight,
    prepare_humming_weight,
    prepare_humming_weight_scale,
    prepare_humming_zero_point,
    prepare_humming_bias,
)


class HummingLayer(torch.nn.Module):
    def __init__(
        self,
        shape_n: int,
        shape_k: int,
        a_dtype: dtypes.DataType,
        b_dtype: dtypes.DataType,
        c_dtype: dtypes.DataType,
        bs_dtype: dtypes.DataType,
        has_input_scale: Optional[bool] = None,
        has_weight_scale: bool = True,
        input_scale_group_size: int = 0,
        weight_scale_group_size: int = 0,
        has_dynamic_zp: bool = False,
        has_bias: bool = False,
        has_global_scale: bool = False,
        num_experts: Optional[int] = None,
        mma_type: Optional[str] = None,
        weight_name: str = "weight",
        weight_scale_name: str = "weight_scale",
        zero_point_name: str = "zero_point",
        bias_name: str = "bias",
        global_scale_name: str = "global_scale",
    ):
        super().__init__()
        packed_size_k = 256 // a_dtype.num_bits
        assert shape_k % (2 * packed_size_k) == 0
        assert shape_n % 64 == 0
        self.shape_n = shape_n
        self.shape_k = shape_k
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.c_dtype = c_dtype
        self.bs_dtype = bs_dtype
        self.num_experts = num_experts

        if has_input_scale is None:
            has_input_scale = self.a_dtype.num_bits != 16
        self.has_input_scale = has_input_scale
        self.has_weight_scale = has_weight_scale
        self.input_scale_group_size = input_scale_group_size
        self.weight_scale_group_size = weight_scale_group_size

        self.has_dynamic_zp = has_dynamic_zp
        self.has_bias = has_bias
        self.has_global_scale = has_global_scale

        self.weight_name = weight_name
        self.weight_scale_name = weight_scale_name
        self.zero_point_name = zero_point_name
        self.bias_name = bias_name
        self.global_scale_name = global_scale_name

        repacked_shape_k = shape_k // packed_size_k
        repacked_shape_n = shape_n * packed_size_k * b_dtype.num_bits // 32
        weight_shape = (repacked_shape_k, repacked_shape_n)
        group_size = weight_scale_group_size if weight_scale_group_size > 0 else shape_k
        num_groups = shape_k // group_size
        weight_scale_shape = (num_groups, shape_n)
        num_zp_bits = 4 if b_dtype.num_bits <= 4 else 8
        zero_point_shape = (num_groups, shape_n * num_zp_bits // 32)
        bias_shape = (shape_n,)
        global_scale_shape = (1,)

        if num_experts is not None:
            assert num_experts > 0
            weight_shape = (num_experts,) + weight_shape
            weight_scale_shape = (num_experts,) + weight_scale_shape
            zero_point_shape = (num_experts,) + zero_point_shape
            bias_shape = (num_experts,) + bias_shape
            global_scale_shape = (num_experts,) + global_scale_shape

        weight = torch.empty(weight_shape, dtype=torch.int32)
        setattr(self, weight_name, torch.nn.Parameter(weight, requires_grad=False))
        if bs_dtype is not None:
            torch_dtype = dtypes.torch_dtype_map[bs_dtype]
            weight_scale = torch.empty(weight_scale_shape, dtype=torch_dtype)
            setattr(
                self,
                weight_scale_name,
                torch.nn.Parameter(weight_scale, requires_grad=False),
            )
            if has_dynamic_zp:
                zero_point = torch.empty(zero_point_shape, dtype=torch.int32)
                setattr(
                    self,
                    zero_point_name,
                    torch.nn.Parameter(zero_point, requires_grad=False),
                )
        if has_global_scale:
            global_scale = torch.empty(global_scale_shape, dtype=torch.float32)
            setattr(
                self,
                global_scale_name,
                torch.nn.Parameter(global_scale, requires_grad=False),
            )
        if has_bias:
            bias = torch.empty(bias_shape, dtype=torch.float32)
            setattr(self, bias_name, torch.nn.Parameter(bias, requires_grad=False))

        self.locks = torch.nn.Buffer(torch.zeros(1024, dtype=torch.int32))

        mma_type = "mma"
        if mma_type is None and torch.cuda.get_device_capability()[0] == 9:
            mma_type = "wgmma"
        mma_type = mma_type.lower()
        self.mma_type = mma_type

    def forward(
        self,
        inputs: torch.Tensor,
        input_scale: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        block_shape: Optional[tuple] = None,
        warp_shape: Optional[tuple] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.has_input_scale:
            assert input_scale is not None
        humming_kernel = HummingKernel(
            problem_shape=(0, self.shape_n, self.shape_k),
            block_shape=block_shape,
            warp_shape=warp_shape,
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
            c_dtype=self.c_dtype,
            bs_dtype=self.bs_dtype,
            has_input_scale=self.has_input_scale,
            has_weight_scale=self.has_weight_scale,
            input_scale_group_size=self.input_scale_group_size,
            weight_scale_group_size=self.weight_scale_group_size,
            has_bias=self.has_bias,
            has_global_scale=self.has_global_scale,
            has_dynamic_zero_point=self.has_dynamic_zp,
            is_moe=self.num_experts is not None,
            device_index=inputs.device.index,
            mma_type=self.mma_type,
            **kwargs,
        )

        return humming_kernel(
            inputs=inputs,
            weight=getattr(self, self.weight_name),
            outputs=outputs,
            input_scale=input_scale,
            weight_scale=getattr(self, self.weight_scale_name, None),
            zero_point=getattr(self, self.zero_point_name, None),
            bias=getattr(self, self.bias_name, None),
            global_scale=getattr(self, self.global_scale_name, None),
            locks=self.locks,
        )

    def set_param_data(
        self,
        param_name: str,
        data: torch.Tensor,
        expert_id: Optional[int] = None,
        pad_data: bool = False,
    ):
        param = getattr(self, param_name, None)
        if param is None:
            return
        if pad_data:
            data = data.view(-1)
            if expert_id is None:
                target_size = param.nelement()
            else:
                target_size = param[expert_id].nelement()
            data = torch.nn.functional.pad(data, pad=(0, target_size - data.size(0)))
        if expert_id is None:
            data = data.view(*param.shape)
            param.data = data
        else:
            data = data.view(*param[expert_id].shape)
            param.data[expert_id] = data

    def load_weight(
        self,
        weight: Optional[torch.Tensor] = None,
        weight_scale: Optional[torch.Tensor] = None,
        zero_point: Optional[torch.Tensor] = None,
        global_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        expert_id: Optional[int] = None,
    ):
        if self.num_experts is None:
            assert expert_id is None
        else:
            assert expert_id < self.num_experts
            assert expert_id >= 0

        if weight is not None:
            weight_param = getattr(self, self.weight_name)
            weight = weight.to(weight_param.device)
            assert weight.dtype == torch.int32

            shape = (self.shape_n, self.shape_k)
            if self.num_experts is not None and expert_id is None:
                shape = (self.num_experts,) + shape

            packed_shape = shape[:-1] + (self.shape_k * self.b_dtype.num_bits // 32,)

            if weight.shape == shape:
                kernel = PackWeightKernel(self.b_dtype.num_bits)
                weight = kernel(weight)

            assert weight.shape == packed_shape

            self.set_param_data(self.weight_name, weight, expert_id)

        if weight_scale is not None:
            weight_scale_param = getattr(self, self.weight_scale_name)
            weight_scale = weight_scale.to(
                device=weight_scale_param.device,
                dtype=weight_scale_param.dtype,
            )
            self.set_param_data(self.weight_scale_name, weight_scale, expert_id)

        if zero_point is not None:
            zero_point_param = getattr(self, self.zero_point_name)
            zero_point = zero_point.to(device=zero_point_param.device)

            num_groups = zero_point_param.size(-2)
            shape = (self.shape_n, num_groups)
            packed_shape = (self.shape_n * self.b_dtype.num_bits // 32, num_groups)
            if self.num_experts is not None and expert_id is None:
                shape = (self.num_experts,) + shape
                packed_shape = (self.num_experts,) + packed_shape

            if zero_point.shape == shape:
                zero_point = zero_point.transpose(-1, -2).contiguous()
                zero_point = zero_point.squeeze().view(*zero_point.shape)

                kernel = PackWeightKernel(self.b_dtype.num_bits)
                zero_point = kernel(zero_point)

                zero_point = zero_point.transpose(-1, -2).contiguous()
                zero_point = zero_point.squeeze().view(*zero_point.shape)

            assert zero_point.shape == packed_shape
            self.set_param_data(self.zero_point_name, zero_point, expert_id, pad_data=True)

        if global_scale is not None:
            global_scale_param = getattr(self, self.global_scale_name)
            global_scale = global_scale.to(
                device=global_scale_param.device,
                dtype=global_scale_param.dtype,
            )
            self.set_param_data(self.global_scale_name, global_scale, expert_id)

        if bias is not None:
            bias_param = getattr(self, self.bias_name, None)
            bias = bias.to(device=bias_param.device, dtype=bias_param.dtype)
            self.set_param_data(self.bias_name, bias, expert_id)

    def load_from_unquantized_weight(self, weight: torch.Tensor):
        if self.num_experts is not None:
            assert weight.shape == (self.num_experts, self.shape_n, self.shape_k)
        else:
            assert weight.shape == (self.shape_n, self.shape_k)

        quanted_weight, weight_scale, zero_point, global_scale = quantize_weight(
            weight,
            dtype=self.b_dtype,
            scale_dtype=self.bs_dtype,
            group_size=self.weight_scale_group_size,
            has_dynamic_zp=self.has_dynamic_zp,
            has_global_scale=self.has_global_scale,
        )

        return self.load_weight(
            weight=quanted_weight,
            weight_scale=weight_scale,
            zero_point=zero_point,
            global_scale=global_scale,
        )

    def finish_load(self):
        weight = getattr(self, self.weight_name)
        weight_scale = getattr(self, self.weight_scale_name, None)
        zero_point = getattr(self, self.zero_point_name, None)
        bias = getattr(self, self.bias_name, None)

        num_experts = self.num_experts if self.num_experts is not None else 1
        weight = weight.view(num_experts, self.shape_n, -1)
        if zero_point is not None:
            num_groups = zero_point.size(-2)
            padded_bits = 4 if self.b_dtype.num_bits <= 4 else 8
            zero_point = zero_point.view(padded_bits, -1)[: self.b_dtype.num_bits]
            zero_point = zero_point.view(num_experts, -1, num_groups)
        if weight_scale is not None:
            weight_scale = weight_scale.view(num_experts, self.shape_n, -1)

        weight = prepare_humming_weight(
            weight=weight,
            b_dtype=self.b_dtype,
            a_dtype=self.a_dtype,
            zero_point=zero_point,
            packed=True,
        )

        self.set_param_data(self.weight_name, weight)

        if weight_scale is not None:
            mma_type = self.mma_type
            weight_scale_group_size = self.weight_scale_group_size

            if mma_type == "mma":
                to_apply_on_c = weight_scale_group_size == 0 or self.a_dtype.num_bits != 16
            elif mma_type == "wgmma":
                to_apply_on_c = weight_scale_group_size == 0

            weight_scale = prepare_humming_weight_scale(
                weight_scale=weight_scale,
                to_apply_on_c=to_apply_on_c,
            )

            self.set_param_data(self.weight_scale_name, weight_scale)

        if zero_point is not None:
            b_dtype = self.b_dtype
            zero_point = prepare_humming_zero_point(
                zero_point,
                dtype=b_dtype,
                packed=True,
            )
            self.set_param_data(self.zero_point_name, zero_point)

        if bias is not None:
            bias = prepare_humming_bias(bias)
            self.set_param_data(self.bias_name, bias)
