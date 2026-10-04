import dataclasses
import hashlib
import itertools
import json
import math
import os
import random

from humming import dtypes
from humming.config import ComputeConfig, GemmType, LayerConfig, MmaType, TuningConfig
from humming.config.mma import get_register_budget_error
from humming.device import current_device
from humming.tune import get_heuristics_config
from humming.utils.smem import estimate_smem_size_config, fits_device_smem

NUM_SAMPLED_TUNING_CONFIGS = 100
TEST_TUNING_SEED_ENV = "HUMMING_TEST_TUNING_SEED"
SAMPLED_TUNING_VALUES = {
    "mma_type": tuple(mma_type.value for mma_type in MmaType),
    "num_stages": (2, 3, 4, 5, 6, 8),
    "use_tma": (True, False, 123, 456, 789),
    "use_warp_spec": (True, False),
    "use_mbarrier": (True, False),
    "use_cp_async": (True, False),
    "multi_cast_size_a": (1, 2),
    "multi_cast_size_b": (1, 2),
    "use_stream_k": (True, False),
    "num_ctas_per_sm": (1, 2, 3, 4),
    "raster_group_m": (1, 2, 5, 9),
    "smem_reuse_mode": ("none", "last_stage", "all_stages"),
    "output_chunk_rows": (0, 32, 64, 96, 128, 256),
    "warp_iters": (2, 4, 8),
    "k_warps": (1, 2, 4),
    "warp_shape_n": (16, 32, 64),
    "block_shape_n": (64, 128, 256, 512),
    "m_warps": (1, 2, 4),
    "warp_shape_m": (8, 16, 24, 32, 48, 64, 80, 96, 102, 128, 176, 192, 200, 256),
    "umma_num_dequant_warpgroups": (1, 2),
    "umma_cta_group_size": (1, 2),
}
TMA_FIELDS = (
    "use_tma_a",
    "use_tma_as",
    "use_tma_as2",
    "use_tma_b",
    "use_tma_c",
    "use_tma_bs",
    "use_tma_bs2",
    "use_tma_bzp",
    "use_tma_bias",
)
TUNING_FIELDS = frozenset(field.name for field in dataclasses.fields(TuningConfig))


def _is_legal_mma_type(layer_config, compute_config, mma_type):
    sm_version = layer_config.sm_version
    if layer_config.use_packed_k_layout and mma_type != MmaType.WGMMA:
        return False
    if layer_config.use_block_scaled_mma:
        expected_mma = MmaType.MXMMA if sm_version // 10 == 12 else MmaType.UMMA
        if mma_type != expected_mma or compute_config.use_f16_accum:
            return False
    has_mixed_raw_weights = (
        layer_config.use_raw_weight and layer_config.a_dtype.num_bits != layer_config.b_dtype.num_bits
    )
    if has_mixed_raw_weights and mma_type != MmaType.UMMA:
        return False
    if mma_type == MmaType.UMMA:
        return layer_config.is_umma_supported and not compute_config.use_f16_accum
    if mma_type == MmaType.MXMMA:
        return sm_version // 10 == 12 and layer_config.use_block_scaled_mma
    if mma_type == MmaType.WGMMA:
        if sm_version != 90 or layer_config.a_dtype == dtypes.int4:
            return False
    min_sm = {
        dtypes.float16: 75,
        dtypes.bfloat16: 80,
        dtypes.int8: 75,
        dtypes.int4: 80,
        dtypes.float8e4m3: 89,
        dtypes.float8e5m2: 89,
        dtypes.float8e3m4: 120,
        dtypes.float4e2m1: 120,
        dtypes.float4e0m3: 120,
    }
    if sm_version < min_sm.get(layer_config.a_dtype, 999):
        return False
    if layer_config.a_dtype.is_floating_point_type and layer_config.b_dtype.is_floating_point_type:
        if layer_config.b_dtype.exponent_bits > layer_config.a_dtype.exponent_bits:
            return False
        if layer_config.b_dtype.mantissa_bits > layer_config.a_dtype.mantissa_bits:
            return False
        if layer_config.a_dtype.exponent_bits and not layer_config.b_dtype.exponent_bits:
            return False
    return True


def create_tuning_config(values: dict) -> TuningConfig:
    return TuningConfig(**{key: value for key, value in values.items() if key in TUNING_FIELDS})


def _generate_cartesian(*names: str):
    for values in itertools.product(*(SAMPLED_TUNING_VALUES[name] for name in names)):
        yield dict(zip(names, values, strict=True))


def _is_legal_geometry(
    layer_config: LayerConfig,
    block_shape: tuple[int, int, int],
    warp_shape: tuple[int, int, int],
    mma_type: MmaType,
) -> bool:
    if block_shape[0] > 256:
        return False
    if layer_config.shape_n % block_shape[1] or layer_config.shape_k % block_shape[2]:
        return False
    if any(block % warp for block, warp in zip(block_shape, warp_shape, strict=True)):
        return False
    if warp_shape[1] > 64:
        return False
    if warp_shape[0] % 8:
        return False
    if mma_type == MmaType.MMA and warp_shape[0] % 16:
        return False
    if mma_type == MmaType.MXMMA and warp_shape[0] % 16:
        return False
    use_wgmma = mma_type == MmaType.WGMMA
    if use_wgmma and layer_config.a_dtype.is_integer_type and warp_shape[0] % 16:
        return False
    min_warp_n = 32 if layer_config.a_dtype.num_bits == 16 else 16
    min_warp_k = {16: 32, 8: 64, 4: 128}[layer_config.a_dtype.num_bits]
    if warp_shape[1] < min_warp_n or warp_shape[2] < min_warp_k:
        return False
    if mma_type == MmaType.WGMMA:
        if block_shape[1] // warp_shape[1] % 4:
            return False
        swizzle_bytes = 128 if layer_config.a_dtype.num_bits * block_shape[2] >= 1024 else 64
        if warp_shape[2] > swizzle_bytes * 8 // layer_config.a_dtype.num_bits:
            return False
    if mma_type == MmaType.UMMA:
        if block_shape[0] != warp_shape[0] or block_shape[2] != warp_shape[2]:
            return False
        if block_shape[1] not in (128, 256, 512) or warp_shape[1] != 32:
            return False
        if block_shape[2] * layer_config.a_dtype.num_bits < 512:
            return False
        if layer_config.a_dtype == dtypes.int8 and block_shape[0] > 32 and block_shape[0] % 16:
            return False
    weight_group_size = 0 if layer_config.use_fused_e8m0_scale else layer_config.weight_scale_group_size
    is_warp_k_gt_groupsize = any(
        group_size and group_size < warp_shape[2]
        for group_size in (layer_config.input_scale_group_size, weight_group_size)
    )
    if layer_config.use_packed_k_layout and (warp_shape[2] != 128 or is_warp_k_gt_groupsize):
        return False
    ratios = tuple(block // warp for block, warp in zip(block_shape, warp_shape, strict=True))
    return all(ratio > 0 and ratio & (ratio - 1) == 0 for ratio in ratios)


def _generate_geometry_candidates(
    layer_config: LayerConfig,
    compute_config: ComputeConfig,
    mma_type: MmaType,
) -> list[tuple[dict, dict]]:
    base = {
        "num_sms": current_device.sm_count,
        "use_f16_accum": compute_config.use_f16_accum,
        "mma_type": mma_type.value,
    }
    names = (
        "warp_iters",
        "k_warps",
        "warp_shape_n",
        "block_shape_n",
        "m_warps",
        "warp_shape_m",
    )
    candidates = []
    part_mma_k = 256 // layer_config.a_dtype.num_bits
    for signature in _generate_cartesian(*names):
        signature = {"mma_type": mma_type.value} | signature
        warp_shape = (
            signature["warp_shape_m"],
            signature["warp_shape_n"],
            signature["warp_iters"] * part_mma_k,
        )
        block_shape = (
            signature["m_warps"] * warp_shape[0],
            signature["block_shape_n"],
            signature["k_warps"] * warp_shape[2],
        )
        if _is_legal_geometry(layer_config, block_shape, warp_shape, mma_type):
            config = base | {"block_shape": block_shape, "warp_shape": warp_shape}
            candidates.append((config, signature))
    return candidates


def _resolve_tma_values(
    mode: bool | int,
    seed: int,
) -> tuple[bool, dict[str, bool]]:
    values = dict.fromkeys(TMA_FIELDS, False)
    if mode is False:
        return False, values
    if mode is True:
        values.update(dict.fromkeys(TMA_FIELDS, True))
        return True, values

    digest = hashlib.sha256(f"{seed}\0{mode}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "little"))
    count = rng.randrange(1, len(TMA_FIELDS))
    values.update(dict.fromkeys(rng.sample(TMA_FIELDS, count), True))
    return True, values


def _is_legal_multicast_transfer(
    compute_config: ComputeConfig,
    sm_version: int,
    signature: dict,
    tma_values: dict[str, bool],
) -> bool:
    size_a = signature["multi_cast_size_a"]
    size_b = signature["multi_cast_size_b"]
    if size_a == 1 and size_b == 1:
        return True

    if sm_version not in (90, 100, 103):
        return False
    if compute_config.gemm_type != GemmType.DENSE:
        return False
    if not signature["use_warp_spec"]:
        return False
    if size_a > 1 and size_b > 1:
        return False
    if size_a > 1 and not tma_values["use_tma_a"]:
        return False
    if size_b > 1 and not tma_values["use_tma_b"]:
        return False
    return True


def _generate_transfer_candidates(
    layer_config: LayerConfig,
    compute_config: ComputeConfig,
    mma_type: MmaType,
) -> list[tuple[dict, dict]]:
    base = {
        "num_sms": current_device.sm_count,
        "use_f16_accum": compute_config.use_f16_accum,
        "mma_type": mma_type.value,
    }
    candidates = []
    names = (
        "use_tma",
        "use_warp_spec",
        "use_mbarrier",
        "use_cp_async",
        "multi_cast_size_a",
        "multi_cast_size_b",
    )
    seed = _get_seed(layer_config, compute_config)
    sm_version = current_device.sm_version
    for signature in _generate_cartesian(*names):
        signature = {"mma_type": mma_type.value} | signature
        use_tma, tma_values = _resolve_tma_values(signature["use_tma"], seed)
        if sm_version < 90 and (use_tma or signature["use_warp_spec"]):
            continue
        if sm_version < 80 and (signature["use_mbarrier"] or signature["use_cp_async"]):
            continue
        if (use_tma or signature["use_warp_spec"]) and not signature["use_mbarrier"]:
            continue
        if compute_config.gemm_type == GemmType.INDEXED:
            tma_values.update(use_tma_a=False, use_tma_as=False, use_tma_as2=False, use_tma_c=False)
        if not (
            layer_config.has_input_scale
            and layer_config.input_scale_group_size > 0
            and compute_config.use_m_major_input_scale
        ):
            tma_values.update(use_tma_as=False)
        if not layer_config.has_input_scale_2 or layer_config.is_tensor_input_scale_2:
            tma_values.update(use_tma_as2=False)
        if not _is_legal_multicast_transfer(compute_config, sm_version, signature, tma_values):
            continue
        if mma_type == MmaType.UMMA:
            if not signature["use_warp_spec"] or not signature["use_mbarrier"]:
                continue
            if signature["multi_cast_size_a"] != 1 or signature["multi_cast_size_b"] != 1:
                continue
            if layer_config.use_raw_weight and not tma_values["use_tma_b"]:
                continue
            tma_values["use_tma_as2"] = False
        config = base | signature | tma_values | {"use_tma": use_tma}
        candidates.append((config, signature | tma_values))
    return candidates


def _generate_scheduling_candidates(
    layer_config: LayerConfig,
    compute_config: ComputeConfig,
    mma_type: MmaType,
) -> list[tuple[dict, dict]]:
    base = {
        "num_sms": current_device.sm_count,
        "use_f16_accum": compute_config.use_f16_accum,
        "mma_type": mma_type.value,
    }
    candidates = []
    names = (
        "num_stages",
        "use_stream_k",
        "num_ctas_per_sm",
        "raster_group_m",
        "smem_reuse_mode",
        "output_chunk_rows",
    )
    if mma_type == MmaType.UMMA:
        names += ("umma_num_dequant_warpgroups", "umma_cta_group_size")
    for signature in _generate_cartesian(*names):
        signature = {"mma_type": mma_type.value} | signature
        if mma_type == MmaType.UMMA:
            if layer_config.use_raw_weight and signature["umma_num_dequant_warpgroups"] != 1:
                continue
            if signature["umma_cta_group_size"] == 2:
                if signature["num_ctas_per_sm"] != 1:
                    continue
        if mma_type == MmaType.WGMMA and signature["num_stages"] < 3:
            continue
        if compute_config.use_batch_invariant and signature["use_stream_k"]:
            continue
        if mma_type == MmaType.MXMMA and layer_config.has_zero_point and signature["use_stream_k"]:
            continue
        candidates.append((base | signature, signature))
    return candidates


def _get_covered_pairs(candidate: tuple[dict, dict]):
    signature = candidate[1]
    return {
        ((left, signature[left]), (right, signature[right]))
        for left, right in itertools.combinations(signature, 2)
    }


def _get_seed(layer_config: LayerConfig, compute_config: ComputeConfig) -> int:
    content = layer_config.to_str() + "\0" + compute_config.to_str()
    seed = os.environ.get(TEST_TUNING_SEED_ENV)
    if seed is not None:
        content += "\0" + seed
    content = content.encode()
    return int.from_bytes(hashlib.sha256(content).digest()[:8], "little")


def _select_pairwise(
    candidates: list[tuple[dict, dict]], rng: random.Random, max_count: int | None = None
) -> list[tuple[dict, dict]]:
    pair_sets = [frozenset(_get_covered_pairs(candidate)) for candidate in candidates]
    uncovered = set().union(*pair_sets)
    remaining = list(range(len(candidates)))
    rng.shuffle(remaining)
    selected = []
    while uncovered and remaining:
        if max_count is not None and len(selected) >= max_count:
            break
        candidate_index = max(remaining, key=lambda index: len(pair_sets[index] & uncovered))
        covered = pair_sets[candidate_index] & uncovered
        if not covered:
            break
        selected.append(candidates[candidate_index])
        uncovered.difference_update(covered)
        remaining.remove(candidate_index)
    return selected


def _fits_device_resources(
    layer_config: LayerConfig,
    compute_config: ComputeConfig,
    candidate: tuple[dict, dict],
) -> bool:
    config = candidate[0]
    mma_type = MmaType(config["mma_type"])
    block_shape = config["block_shape"]
    warp_shape = config["warp_shape"]
    m_warps = block_shape[0] // warp_shape[0]
    n_warps = block_shape[1] // warp_shape[1]
    k_warps = block_shape[2] // warp_shape[2]
    num_math_threads = m_warps * n_warps * k_warps * 32
    num_threads = num_math_threads + (128 if config["use_warp_spec"] else 0)
    if mma_type == MmaType.UMMA:
        dequant_threads = 0 if layer_config.use_raw_weight else 128 * config["umma_num_dequant_warpgroups"]
        num_threads = 256 + dequant_threads
    num_ctas_per_sm = config["num_ctas_per_sm"]
    max_threads = current_device.max_threads_per_sm
    registers_per_sm = current_device.max_registers_per_sm
    if num_threads * num_ctas_per_sm > max_threads:
        return False
    launch_bound_registers = registers_per_sm // (num_threads * num_ctas_per_sm) // 8 * 8
    if mma_type == MmaType.UMMA and launch_bound_registers < 40:
        # TMEM transfers and their address operands cannot compile at a 32-register limit.
        return False

    if config["use_warp_spec"] and mma_type != MmaType.UMMA:
        needs_more_load_registers = num_math_threads > 256 or (
            num_ctas_per_sm == 1 and layer_config.a_dtype.num_bits != 16
        )
        load_thread_registers = 40 if needs_more_load_registers else 24
        if launch_bound_registers < load_thread_registers:
            return False

    tuning_config = create_tuning_config(config)
    if (
        get_register_budget_error(layer_config, tuning_config, compute_config.use_f16_accum, registers_per_sm)
        is not None
    ):
        return False
    if mma_type == MmaType.UMMA:
        from humming.tune.sm100 import Sm100UmmaHeuristics

        tmem_columns = Sm100UmmaHeuristics._get_tmem_columns(
            layer_config, block_shape, config["num_stages"], num_ctas_per_sm, config["umma_cta_group_size"]
        )
        if tmem_columns * num_ctas_per_sm > 512:
            return False
        smem_size = estimate_smem_size_config(layer_config, compute_config, tuning_config)
        return smem_size * num_ctas_per_sm <= current_device.max_smem_size
    return fits_device_smem(layer_config, compute_config, tuning_config)


def _try_combine_candidate(
    layer_config: LayerConfig,
    compute_config: ComputeConfig,
    items: tuple[tuple[dict, dict], tuple[dict, dict], tuple[dict, dict]],
) -> tuple[dict, dict] | None:
    geometry_item, transfer_item, scheduling_item = items
    geometry_config, geometry_signature = geometry_item
    transfer_config, transfer_signature = transfer_item
    scheduling_config, scheduling_signature = scheduling_item
    config = geometry_config | transfer_config | scheduling_config
    mma_type = MmaType(config["mma_type"])
    block_shape = config["block_shape"]
    warp_shape = config["warp_shape"]
    m_warps = block_shape[0] // warp_shape[0]
    n_warps = block_shape[1] // warp_shape[1]
    k_warps = block_shape[2] // warp_shape[2]
    num_math_threads = m_warps * n_warps * k_warps * 32
    num_threads = num_math_threads + (128 if config["use_warp_spec"] else 0)
    if mma_type == MmaType.UMMA:
        if config["umma_cta_group_size"] == 2:
            if block_shape[0] % 16 or layer_config.shape_n % (2 * block_shape[1]):
                return None
        num_math_threads = 128
        dequant_threads = 0 if layer_config.use_raw_weight else 128 * config["umma_num_dequant_warpgroups"]
        num_threads = 256 + dequant_threads
    if num_threads > 1024:
        return None
    use_warp_group = config["use_warp_spec"] or mma_type == MmaType.WGMMA
    if use_warp_group and num_math_threads % 128:
        return None
    if mma_type == MmaType.WGMMA and config["num_stages"] < 3:
        return None
    if layer_config.shape_n % (block_shape[1] * config["multi_cast_size_a"]):
        return None
    if compute_config.use_batch_invariant and (config["use_stream_k"] or block_shape[2] != warp_shape[2]):
        return None
    if (
        layer_config.has_zero_point
        and layer_config.is_fp_zero_point
        and config["use_tma_bzp"]
        and block_shape[1] > 256
    ):
        return None

    signature = geometry_signature | transfer_signature | scheduling_signature
    candidate = (config, signature)
    return candidate if _fits_device_resources(layer_config, compute_config, candidate) else None


def _enumerate_backend_candidates(
    layer_config: LayerConfig,
    compute_config: ComputeConfig,
    mma_type: MmaType,
    sample_size: int,
) -> list[tuple[dict, dict]]:
    rng = random.Random(_get_seed(layer_config, compute_config))
    groups = (
        _generate_geometry_candidates(layer_config, compute_config, mma_type),
        _generate_transfer_candidates(layer_config, compute_config, mma_type),
        _generate_scheduling_candidates(layer_config, compute_config, mma_type),
    )
    candidates = []
    signatures = set()

    def add(items):
        candidate = _try_combine_candidate(layer_config, compute_config, items)
        if candidate is None:
            return
        key = json.dumps(candidate[1], sort_keys=True)
        if key not in signatures:
            signatures.add(key)
            candidates.append(candidate)

    reduced_groups = tuple(_select_pairwise(group, rng) for group in groups)
    for items in itertools.product(*reduced_groups):
        add(items)

    target_pool_size = sample_size * 5
    product_size = math.prod(len(group) for group in groups)
    trial_count = min(product_size, target_pool_size * 500)
    for flat_index in rng.sample(range(product_size), trial_count):
        indices = []
        for group in reversed(groups):
            flat_index, index = divmod(flat_index, len(group))
            indices.append(index)
        items = tuple(group[index] for group, index in zip(groups, reversed(indices), strict=True))
        add(items)
        if len(candidates) >= target_pool_size:
            break
    return candidates


def enumerate_test_tuning_configs(
    layer_config: LayerConfig,
    compute_config: ComputeConfig,
    sample_size: int = NUM_SAMPLED_TUNING_CONFIGS,
) -> list[tuple[dict, dict]]:
    candidates = []
    for value in SAMPLED_TUNING_VALUES["mma_type"]:
        mma_type = MmaType(value)
        if _is_legal_mma_type(layer_config, compute_config, mma_type):
            candidates.extend(
                _enumerate_backend_candidates(layer_config, compute_config, mma_type, sample_size)
            )
    return candidates


def sample_test_tuning_configs(
    layer_config: LayerConfig,
    compute_config: ComputeConfig,
    sample_size: int = NUM_SAMPLED_TUNING_CONFIGS,
) -> list[dict]:
    candidates = enumerate_test_tuning_configs(layer_config, compute_config, sample_size)
    rng = random.Random(_get_seed(layer_config, compute_config))
    selected = _select_pairwise(candidates, rng, max_count=sample_size)
    selected_ids = {id(candidate) for candidate in selected}
    remaining = [candidate for candidate in candidates if id(candidate) not in selected_ids]
    rng.shuffle(remaining)
    target_size = min(sample_size, len(candidates))
    selected.extend(remaining[: target_size - len(selected)])
    rng.shuffle(selected)
    return [config for config, _ in selected]


def generate_heuristics_configs(
    layer_config: LayerConfig,
    compute_config: ComputeConfig,
    shape_ms: list[int] | tuple[int, ...],
) -> list[dict]:
    configs = [
        get_heuristics_config(
            layer_config,
            shape_m=shape_m,
            use_f16_accum=compute_config.use_f16_accum,
            use_batch_invariant=compute_config.use_batch_invariant,
            use_m_major_input_scale=compute_config.use_m_major_input_scale,
            gemm_type=compute_config.gemm_type,
        )
        for shape_m in shape_ms
    ]
    if compute_config.use_batch_invariant:
        for config in configs:
            assert not config.get("use_stream_k", False)
            assert config["warp_shape"][2] == config["block_shape"][2]
    return configs
