from types import SimpleNamespace

import pytest

from humming.tune._worker import (
    default_expert_max_tokens,
    indexed_input_rows,
    masked_input_tokens,
)


def test_indexed_input_rows_w13_is_per_token():
    # shape_m is routed rows; w13 consumes tokens = shape_m // top_k.
    activation_rows, routing_tokens = indexed_input_rows(
        shape_m=2048, top_k=8, is_moe_down=False
    )
    assert routing_tokens == 256
    assert activation_rows == 256
    # routed rows produced by the generator == shape_m.
    assert routing_tokens * 8 == 2048


def test_indexed_input_rows_w2_is_routed_rows():
    activation_rows, routing_tokens = indexed_input_rows(
        shape_m=2048, top_k=8, is_moe_down=True
    )
    assert routing_tokens == 256
    assert activation_rows == 2048


def test_indexed_input_rows_requires_divisible_shape_m():
    with pytest.raises(ValueError, match="divisible by top_k"):
        indexed_input_rows(shape_m=2049, top_k=8, is_moe_down=False)


def test_masked_serving_key_units():
    assert masked_input_tokens(432, 6) == 72

    args = SimpleNamespace(
        expert_max_tokens=None,
        num_experts=48,
        balanced=True,
    )
    assert default_expert_max_tokens(args, routed_rows=432) == 64


def test_masked_input_tokens_rounds_up_non_divisible_keys():
    # Masked serving keys (expected_m * E_local) need not divide top_k:
    # DSV4-Flash EP8 has E_local=32, top_k=6 -> key=64 at capture bs=8.
    assert masked_input_tokens(64, 6) == 11
    assert masked_input_tokens(432, 6) == 72
    assert masked_input_tokens(1, 6) == 1


def test_indexed_default_grid_aligns_to_top_k():
    from humming.config import GemmType
    from humming.tune.__main__ import validate_args

    args = SimpleNamespace(shape_m_list=None, num_experts=32, top_k=6)
    validate_args(args, GemmType.INDEXED)
    assert args.shape_m_list
    assert all(m % 6 == 0 for m in args.shape_m_list)


def test_indexed_explicit_misaligned_grid_rejected():
    from humming.config import GemmType
    from humming.tune.__main__ import validate_args

    args = SimpleNamespace(shape_m_list=[100], num_experts=32, top_k=6)
    with pytest.raises(ValueError, match="not divisible"):
        validate_args(args, GemmType.INDEXED)
