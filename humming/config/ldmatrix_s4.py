"""Eligibility for the first dense signed-s4 physical loader contract."""

from collections.abc import Mapping
from typing import Any

NEW_LOADER = "ldmatrix_s8_s4"
LEGACY_LOADER = "packed_k_legacy"
NEW_LAYOUT = "signed_s4_nk_k64_v1"
LEGACY_LAYOUT = "packed_k_legacy_v1"


def layer_rejection_reasons(config: Mapping[str, Any]) -> tuple[str, ...]:
    checks = (
        (config["sm_version"] == 90, "target"),
        (config["mma_type"] == "wgmma", "mma"),
        (config["num_experts"] == 0, "moe"),
        (config["a_dtype"] == "int8" and config["b_dtype"] == "uint4", "physical_format"),
        (config["use_packed_k_layout"], "packed_k"),
        (not config["has_zero_point"] and not config["is_fp_zero_point"], "zero_point"),
        (not config["use_int_weight_scale"] and not config["use_fused_e8m0_scale"], "preprocess"),
        (
            config["weight_scale_type"] == "group"
            and config["weight_scale_group_size"] == 128
            and config["weight_scale_group_size_n"] in (0, 1)
            and config["weight_scale_2_type"] == "none",
            "weight_scale",
        ),
        (
            config["input_quant_mode"] == "dynamic_token"
            and config["input_scale_group_size"] == 0
            and config["as_dtype"] == "float32",
            "input_scale",
        ),
        (config["bs_dtype"] == config["c_dtype"] == "bfloat16", "scale_output_dtype"),
        (not config["has_bias"], "bias"),
        (config["shape_n"] >= 128 and config["shape_n"] % 128 == 0, "shape_n"),
        (config["shape_k"] >= 128 and config["shape_k"] % 128 == 0, "shape_k"),
        (config["pad_shape_n"] == config["pad_shape_k"] == 0, "padding"),
    )
    return tuple(reason for passed, reason in checks if not passed)


def toolchain_rejection_reasons(compiler: str, version: tuple[int, ...]) -> tuple[str, ...]:
    return () if compiler == "NVCCCompiler" and version >= (13, 4) else ("toolchain_ptx",)


def specialization_rejection_reasons(
    config: Mapping[str, Any],
    shape_n: int,
) -> tuple[str, ...]:
    block = config["block_shape"]
    warp = config["warp_shape"]
    checks = (
        (config.get("gemm_type") in (None, "dense"), "execution_mode"),
        (block[2] == warp[2] == 64, "geometry_k"),
        (block[1] in (128, 256) and warp[1] == block[1] // 4 and shape_n % block[1] == 0, "geometry_n"),
        (block[0] == warp[0] and block[0] in (16, 32, 48, 64, 80, 96, 112, 128), "geometry_m"),
        (
            config.get("use_tma")
            and config.get("use_tma_b", True)
            and config.get("use_warp_spec")
            and config.get("use_mbarrier"),
            "staging",
        ),
        (config.get("num_stages") == 3, "stages"),
        (config.get("multi_cast_size_a", 1) == config.get("multi_cast_size_b", 1) == 1, "multicast"),
        (
            not config.get("use_f16_accum", False)
            and not config.get("use_stream_k", True)
            and not config.get("use_pdl", False)
            and not config.get("reduce_overlap_last_stage_only", False),
            "schedule",
        ),
    )
    return tuple(reason for passed, reason in checks if not passed)


def forced_tuning(block_m_seed: int, shape_n: int) -> dict[str, Any]:
    """The single dense tuning family a prepared signed-s4 layer must run on.

    Pure/CUDA-free so the production forcing (humming.tune._apply_ldmatrix_s4_contract)
    and the CPU contract test share one definition. specialization_rejection_reasons
    must accept exactly what this emits for any layer-eligible shape.
    """
    block_m = min(128, (block_m_seed + 15) // 16 * 16)
    block_n = 256 if shape_n % 256 == 0 else 128
    return {
        "block_shape": (block_m, block_n, 64),
        "warp_shape": (block_m, block_n // 4, 64),
        "use_tma": True,
        "use_tma_b": True,
        "use_warp_spec": True,
        "use_mbarrier": True,
        "use_stream_k": False,
        "use_f16_accum": False,
        "use_pdl": False,
        "num_stages": 3,
        "multi_cast_size_a": 1,
        "multi_cast_size_b": 1,
        "reduce_overlap_last_stage_only": False,
    }


def assert_layout_compatible(prepared: str, required: str) -> None:
    if prepared != required:
        raise ValueError(f"prepared B layout {prepared} does not match loader layout {required}")


def resolve_specialization_loader(
    layer: Mapping[str, Any], tuning: Mapping[str, Any], layer_reasons: tuple[str, ...]
) -> tuple[str, tuple[str, ...]]:
    """Resolve against chosen geometry without reinterpreting already prepared B."""
    if layer["use_ldmatrix_s4"]:
        reasons = layer_reasons + specialization_rejection_reasons(tuning, layer["shape_n"])
        if layer.get("test_force_packed_k_legacy", False):
            reasons += ("test_force_legacy_layout_mismatch",)
        if reasons:
            raise ValueError(f"ldmatrix.s8.s4 prepared layout cannot use specialization: {reasons}")
        return NEW_LOADER, ()
    reasons = (
        ("test_force_legacy",)
        if layer.get("test_force_packed_k_legacy", False)
        else layer_reasons or ("explicit_legacy",)
    )
    return (LEGACY_LOADER if layer["use_packed_k_layout"] else "generic"), reasons
