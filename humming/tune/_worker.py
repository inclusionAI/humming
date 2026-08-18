import json
import math
import os
from types import SimpleNamespace


def _format_short_error(exc):
    return f"{type(exc).__name__}: {exc}"[:500]


def create_layer(args, gemm_type):
    import torch

    from humming import dtypes
    from humming.layer import HummingLayer
    from humming.testing.data import random_fill_tensor
    from humming.tune._testdata import generate_random_weight

    def _fill_layer_params(layer):
        # Preserve the well-conditioned fallback for any parameter that is not
        # replaced by generate_random_weight below.
        with torch.no_grad():
            for name, tensor in layer.named_parameters():
                if "scale" in name and tensor.dtype == torch.uint8:
                    tensor.random_(125, 130)
                elif "scale" in name and str(tensor.dtype).startswith(
                    "torch.float8_e8m0"
                ):
                    exponents = torch.randint(
                        125, 130, tensor.shape, device=tensor.device, dtype=torch.uint8
                    )
                    tensor.copy_(exponents.view(tensor.dtype))
                elif "scale" in name and tensor.dtype in (
                    torch.float16,
                    torch.bfloat16,
                    torch.float32,
                ):
                    tensor.uniform_(0.5, 2.0)
                else:
                    random_fill_tensor(tensor)

    torch_dtype = dtypes.torch_dtype_map[dtypes.DataType.from_str(args.c_dtype)]
    layer = HummingLayer(
        shape_n=args.shape_n,
        shape_k=args.shape_k,
        num_experts=args.num_experts if gemm_type.value != "dense" else 0,
        weight_config={
            "dtype": args.b_dtype,
            "group_size": args.weight_scale_group_size,
            "scale_dtype": args.bs_dtype,
            "has_zero_point": False,
            "is_fp_zero_point": False,
        },
        input_config={"dtype": args.a_dtype, "group_size": args.input_scale_group_size},
        torch_dtype=torch_dtype,
    ).to("cuda:0")

    _fill_layer_params(layer)
    num_experts = None if gemm_type.value == "dense" else args.num_experts
    _, weight_ref, weight, weight_scale, _, weight_scale_2 = generate_random_weight(
        n=args.shape_n,
        k=args.shape_k,
        group_size=args.weight_scale_group_size,
        dtype=dtypes.DataType.from_str(args.b_dtype),
        scale_dtype=dtypes.DataType.from_str(args.bs_dtype),
        num_experts=num_experts,
    )

    # Copy the untransformed representation; layer.transform() owns fused-E8M0
    # processing and all kernel-specific weight/scale layout changes. The layer
    # param stores sub-byte weights packed (e.g. 8 x fp4 per int32) while
    # generate_random_weight returns them unpacked, so pack before copying.
    from humming.ops.weight import pack_weight

    b_num_bits = dtypes.DataType.from_str(args.b_dtype).num_bits
    with torch.no_grad():
        for name, value in (
            ("weight", weight),
            ("weight_scale", weight_scale),
            ("weight_scale_2", weight_scale_2),
        ):
            if value is None:
                continue
            target = getattr(layer, name)
            if name == "weight" and target.shape != value.shape and b_num_bits < 8:
                value = pack_weight(value.contiguous().to(torch.int32), b_num_bits)
                if value.dtype != target.dtype:
                    value = value.view(target.dtype)
            if target.shape != value.shape:
                raise ValueError(
                    f"generated {name} shape {tuple(value.shape)} does not match "
                    f"layer shape {tuple(target.shape)} before transform"
                )
            target.copy_(value)
    layer.transform()
    return layer, torch_dtype, weight_ref


def default_expert_max_tokens(args, routed_rows):
    if args.expert_max_tokens is not None:
        return args.expert_max_tokens
    tokens_per_expert = math.ceil(routed_rows / args.num_experts / 64) * 64
    multiplier = 1 if args.balanced else 2
    return max(64, tokens_per_expert * multiplier)


def indexed_input_rows(shape_m, top_k, is_moe_down):
    """Activation rows and routing token count for indexed gemm.

    ``shape_m`` is routed rows (tokens * top_k), matching the serving bucket key
    ``estimate_local_valid_shape_m() = topk_ids.nelement()``. w13 consumes
    per-token activations (``shape_m // top_k`` rows); w2 consumes the routed
    rows themselves (``shape_m`` rows). Routing metadata is built from the token
    count for both.
    """
    if shape_m % top_k != 0:
        raise ValueError(f"indexed shape_m={shape_m} must be divisible by top_k={top_k}")
    routing_tokens = shape_m // top_k
    activation_rows = shape_m if is_moe_down else routing_tokens
    return activation_rows, routing_tokens


def masked_input_tokens(shape_m, top_k):
    """Routing token count for grouped masked gemm.

    ``shape_m`` is the serving bucket key (``expected_m * num_local_experts``,
    see sglang ``estimate_local_valid_shape_m``). Unlike indexed routed rows it
    is NOT guaranteed to be divisible by top_k (e.g. DSV4-Flash EP8 has
    E_local=32 with top_k=6), so round the token count up; the generated load
    carries at most top_k-1 extra routed rows, immaterial for timing.
    """
    assert shape_m > 0 and top_k > 0, f"{shape_m=} {top_k=}"
    return -(-shape_m // top_k)


def make_inputs(args, gemm_type, torch_dtype, shape_m, block_size_config=None):
    from humming import dtypes
    from humming.testing.data import generate_random_moe_tensors
    from humming.tune._testdata import generate_random_inputs

    # Number of token rows fed to the routing generator (topk_ids is
    # [routing_tokens, top_k]). Serving shape_m keys are routed rows for both
    # indexed and grouped masked gemm.
    routing_tokens = shape_m
    if gemm_type.value == "dense":
        actual_shape_m = shape_m
        expert_max_tokens = None
    elif gemm_type.value == "indexed":
        actual_shape_m, routing_tokens = indexed_input_rows(
            shape_m, args.top_k, args.is_moe_down
        )
        expert_max_tokens = None
    else:
        routing_tokens = masked_input_tokens(shape_m, args.top_k)
        expert_max_tokens = default_expert_max_tokens(args, shape_m)
        actual_shape_m = args.num_experts * expert_max_tokens

    # Metadata is cached per block_m; keep activations identical across configs.
    import torch

    torch.cuda.manual_seed(shape_m)
    _, inputs_ref, inputs, input_scale = generate_random_inputs(
        m=actual_shape_m,
        k=args.shape_k,
        group_size=args.input_scale_group_size,
        dtype=dtypes.DataType.from_str(args.a_dtype),
    )

    torch.cuda.manual_seed(shape_m)
    if gemm_type.value == "dense":
        expert_layout = None
        sorted_ids = None
        expert_ids = None
        num_tokens_padded = None
    else:
        moe_tensors = generate_random_moe_tensors(
            shape_m=routing_tokens,
            num_experts=args.num_experts,
            top_k=args.top_k,
            gemm_type=gemm_type,
            balanced=args.balanced,
            block_size_config=block_size_config,
            expert_max_tokens=expert_max_tokens,
        )
        topk_ids, expert_layout, sorted_ids, expert_ids, num_tokens_padded = moe_tensors

    return {
        "inputs": inputs,
        "inputs_ref": inputs_ref,
        "input_scale": input_scale,
        "topk_ids": None if gemm_type.value == "dense" else topk_ids,
        "expert_layout": expert_layout,
        "sorted_ids": sorted_ids,
        "expert_ids": expert_ids,
        "num_tokens_padded": num_tokens_padded,
        "actual_shape_m": actual_shape_m,
        "expert_max_tokens": expert_max_tokens,
    }


def compute_golden(batch, weight_ref, gemm_type, output_dtype, top_k):
    """Compute the config-independent quantized-input/weight reference."""
    import torch

    inputs_ref = batch["inputs_ref"].float()
    weight_ref = weight_ref.float()

    if gemm_type.value == "dense":
        return inputs_ref.matmul(weight_ref.T).to(output_dtype)

    shape_n = weight_ref.size(-2)
    if gemm_type.value == "grouped_masked":
        golden = torch.zeros(
            (batch["actual_shape_m"], shape_n),
            dtype=output_dtype,
            device=inputs_ref.device,
        )
        for expert_id, count in enumerate(batch["expert_layout"].cpu().tolist()):
            start = expert_id * batch["expert_max_tokens"]
            end = start + count
            if end > start:
                golden[start:end] = inputs_ref[start:end].matmul(
                    weight_ref[expert_id].T
                ).to(output_dtype)
        return golden

    assert gemm_type.value == "indexed"
    sorted_ids = batch["sorted_ids"]
    expert_ids = batch["expert_ids"]
    routed_rows = batch["topk_ids"].numel()
    if expert_ids.numel() == 0 or sorted_ids.numel() % expert_ids.numel() != 0:
        raise ValueError("indexed routing metadata has an invalid padded block layout")
    block_m = sorted_ids.numel() // expert_ids.numel()
    golden = torch.empty(
        (routed_rows, shape_n), dtype=output_dtype, device=inputs_ref.device
    )
    assigned = torch.zeros(routed_rows, dtype=torch.bool, device=inputs_ref.device)
    for block_id, expert_id in enumerate(expert_ids.cpu().tolist()):
        output_ids = sorted_ids[block_id * block_m : (block_id + 1) * block_m].long()
        output_ids = output_ids[output_ids < routed_rows]
        if output_ids.numel() == 0:
            continue
        input_ids = torch.div(output_ids, top_k, rounding_mode="floor")
        golden[output_ids] = inputs_ref[input_ids].matmul(weight_ref[expert_id].T).to(
            output_dtype
        )
        assigned[output_ids] = True
    if not assigned.all():
        raise ValueError("indexed routing metadata did not assign every output row")
    return golden


def make_run(layer, batch, compute_config_json, tuning_config, gemm_type, shape_m, top_k):
    def run():
        valid_shape_m = 0
        if gemm_type.value == "grouped_masked":
            valid_shape_m = shape_m
        return layer(
            inputs=batch["inputs"],  # noqa: B026
            input_scale=batch["input_scale"],  # noqa: B026
            sorted_ids=batch["sorted_ids"],  # noqa: B026
            expert_ids=batch["expert_ids"],  # noqa: B026
            num_tokens_padded=batch["num_tokens_padded"],  # noqa: B026
            expert_layout=batch["expert_layout"],  # noqa: B026
            valid_shape_m=valid_shape_m,
            compute_config=compute_config_json,
            tuning_config=tuning_config,  # noqa: B026
            top_k=top_k,
        )

    return run


def masked_valid_rows(expert_layout, expert_max_tokens):
    import torch

    rows = []
    for expert_id, count in enumerate(expert_layout.cpu().tolist()):
        start = expert_id * expert_max_tokens
        rows.extend(range(start, start + count))
    return torch.tensor(rows, dtype=torch.long, device="cuda:0")


def comparable_output(output, compare_rows):
    output = output.view(-1, output.size(-1))
    if compare_rows is None:
        return output
    return output.index_select(0, compare_rows)


def worker_main(args_dict, cmd_q, res_q):
    import torch
    import triton

    from humming.config.enum import GemmType

    args = SimpleNamespace(**args_dict)
    gemm_type = GemmType(args.gemm_type)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    layer, torch_dtype, weight_ref = create_layer(args, gemm_type)
    compute_config_json = json.dumps({"use_f16_accum": False, "gemm_type": gemm_type.value})
    batches = {}
    refs = {}

    layer_top_k = args.top_k if not getattr(args, "is_moe_down", False) else 1

    def get_batch(shape_m, block_size_config=None):
        key = (shape_m, block_size_config)
        if key not in batches:
            if not any(batch_key[0] == shape_m for batch_key in batches):
                batches.clear()
            batches[key] = make_inputs(args, gemm_type, torch_dtype, shape_m, block_size_config)
        return batches[key]

    res_q.put({"op": "ready"})
    while True:
        msg = cmd_q.get()
        op = msg["op"]
        if op == "stop":
            os._exit(0)
        shape_m = msg["m"]
        try:
            if op == "set_ref":
                if shape_m not in refs:
                    # indexed metadata needs a block size, but the golden route is
                    # block-size independent; 64 is only an efficient packing choice.
                    ref_block_m = 64 if gemm_type.value == "indexed" else None
                    batch = get_batch(shape_m, ref_block_m)
                    golden = compute_golden(
                        batch, weight_ref, gemm_type, torch_dtype, layer_top_k
                    )
                    rows = None
                    if gemm_type.value == "grouped_masked":
                        rows = masked_valid_rows(
                            batch["expert_layout"], batch["expert_max_tokens"]
                        )
                    refs.clear()
                    refs[shape_m] = (comparable_output(golden, rows), rows)
                res_q.put({"op": op, "error": ""})
                continue

            block_size_config = None
            if gemm_type.value == "indexed":
                block_size_config = msg["config"]["block_shape"][0]
            batch = get_batch(shape_m, block_size_config)
            run = make_run(
                layer,
                batch,
                compute_config_json,
                msg["config"],
                gemm_type,
                shape_m,
                layer_top_k,
            )
            if op == "bench":
                ms = triton.testing.do_bench(run, warmup=msg["warmup"], rep=msg["rep"])
                res_q.put({"op": op, "ms": ms, "error": ""})
            elif op == "compare":
                ref_output, rows = refs[shape_m]
                output = run()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    comparable_output(output, rows), ref_output, rtol=0.05, atol=0.5
                )
                res_q.put({"op": op, "passed": True, "error": ""})
        except Exception as exc:
            try:
                torch.cuda.synchronize()
            except Exception:
                os._exit(43)
            if op == "compare":
                res_q.put({"op": op, "passed": False, "error": _format_short_error(exc)})
            else:
                res_q.put({"op": op, "ms": float("inf"), "error": _format_short_error(exc)})
