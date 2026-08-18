import functools
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from humming.utils.jit import hash_to_hex

if TYPE_CHECKING:
    from humming.config import GemmType
    from humming.config import LayerConfig


_CACHE_FLAG_NAMES = (
    "use_batch_invariant",
    "use_f16_accum",
    "use_m_major_input_scale",
)
_TUPLE_CONFIG_KEYS = ("block_shape", "warp_shape")
_LOGGER = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_default_tune_cache_dir() -> str:
    cache_dir = os.getenv("HUMMING_CACHE_DIR")
    if cache_dir is not None:
        return cache_dir
    return os.path.join(os.path.expanduser("~"), ".humming/tune_cache/")


def _get_tune_cache_dir(cache_dir: str | None = None) -> str:
    if cache_dir is not None:
        return cache_dir
    return _get_default_tune_cache_dir()


def _normalize_flags(flags: dict) -> dict:
    return {name: flags.get(name, False) for name in sorted(_CACHE_FLAG_NAMES)}


def _serialize_flags(flags: dict) -> str:
    return json.dumps(_normalize_flags(flags), sort_keys=True, separators=(",", ":"))


def _cache_meta_str(meta: "LayerConfig") -> str:
    # A tuning table depends on the GEMM (shape, dtype, padding, scaling), not on
    # the layer's name. Strip sublayer_name so a table tuned offline (or under a
    # different sublayer label) resolves to the same key as the serving layer,
    # which names its sublayers "w13"/"w2".
    raw = meta.to_str()
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(obj, dict):
        obj.pop("sublayer_name", None)
        return json.dumps(obj, sort_keys=True)
    return raw


def make_cache_key(meta: "LayerConfig", gemm_type: "GemmType", flags: dict) -> str:
    return hash_to_hex(_cache_meta_str(meta) + gemm_type.value + _serialize_flags(flags))


def _make_cache_filename(
    meta: "LayerConfig",
    gemm_type: "GemmType",
    flags: dict,
    cache_dir: str | None = None,
) -> str:
    dirname = _get_tune_cache_dir(cache_dir)
    return os.path.join(dirname, make_cache_key(meta, gemm_type, flags) + ".json")


def _restore_config_tuples(config: dict) -> dict:
    restored = dict(config)
    for key in _TUPLE_CONFIG_KEYS:
        if key in restored and isinstance(restored[key], list):
            restored[key] = tuple(restored[key])
    return restored


def _restore_table(table: list) -> list:
    restored = []
    for row in table:
        # JSON turns tuple-valued kernel shapes into lists; restore the nested
        # config fields. Validate the row shape here so a malformed cache file
        # degrades to a cache miss instead of crashing the serving lookup.
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise ValueError(f"malformed tune table row: {row!r}")
        min_shape_m, max_shape_m, config = row
        if not isinstance(config, dict):
            raise ValueError(f"malformed tune table config: {config!r}")
        restored.append(
            [int(min_shape_m), int(max_shape_m), _restore_config_tuples(config)]
        )
    return restored


def _load_valid_payload(
    meta: "LayerConfig",
    gemm_type: "GemmType",
    flags: dict,
    fingerprint: dict,
    cache_dir: str | None = None,
) -> dict | None:
    filename = _make_cache_filename(meta, gemm_type, flags, cache_dir)
    try:
        with open(filename, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    meta_str = _cache_meta_str(meta)
    normalized_flags = _normalize_flags(flags)
    gemm_type_value = gemm_type.value
    if payload.get("_fingerprint") != fingerprint:
        _LOGGER.debug("tune cache fingerprint mismatch: %s", filename)
        return None
    if payload.get("_meta_str") != meta_str:
        _LOGGER.debug("tune cache meta mismatch: %s", filename)
        return None
    if payload.get("_gemm_type") != gemm_type_value:
        _LOGGER.debug("tune cache gemm type mismatch: %s", filename)
        return None
    if payload.get("_flags") != normalized_flags:
        _LOGGER.debug("tune cache flags mismatch: %s", filename)
        return None

    return payload


def load_table(
    meta: "LayerConfig",
    gemm_type: "GemmType",
    flags: dict,
    fingerprint: dict,
    cache_dir: str | None = None,
) -> list | None:
    """Return a cache table only when metadata and fingerprint match exactly."""
    payload = _load_valid_payload(meta, gemm_type, flags, fingerprint, cache_dir)
    if payload is None:
        return None

    try:
        return _restore_table(payload["table"])
    except (KeyError, TypeError, IndexError, ValueError, OverflowError):
        filename = _make_cache_filename(meta, gemm_type, flags, cache_dir)
        _LOGGER.debug("tune cache table restore failed: %s", filename)
        return None


def load_saved_shape_m_list(
    meta: "LayerConfig",
    gemm_type: "GemmType",
    flags: dict,
    fingerprint: dict,
    cache_dir: str | None = None,
) -> list | None:
    """Return the grid saved with a valid tuning cache, if one is recorded."""
    payload = _load_valid_payload(meta, gemm_type, flags, fingerprint, cache_dir)
    if payload is None:
        return None

    shape_m_list = payload.get("_shape_m_list")
    if not isinstance(shape_m_list, list) or not all(
        isinstance(shape_m, int) and not isinstance(shape_m, bool)
        for shape_m in shape_m_list
    ):
        return None
    return list(shape_m_list)


def save_table(
    meta: "LayerConfig",
    gemm_type: "GemmType",
    flags: dict,
    fingerprint: dict,
    table: list,
    cache_dir: str | None = None,
    shape_m_list: list[int] | None = None,
    per_m: list[dict] | None = None,
) -> str:
    dirname = _get_tune_cache_dir(cache_dir)
    os.makedirs(dirname, exist_ok=True)
    filename = _make_cache_filename(meta, gemm_type, flags, cache_dir)
    payload = {
        "_fingerprint": fingerprint,
        "_meta_str": _cache_meta_str(meta),
        "_gemm_type": gemm_type.value,
        "_flags": _normalize_flags(flags),
        "_shape_m_list": None if shape_m_list is None else list(shape_m_list),
        "_per_m": None if per_m is None else list(per_m),
        "table": table,
        "_created_at": datetime.now(timezone.utc).isoformat(),
    }

    tmp_filename = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=dirname,
            encoding="utf-8",
            prefix="." + os.path.basename(filename) + ".",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_filename = f.name
            json.dump(payload, f, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_filename, filename)
    except Exception:
        if tmp_filename is not None:
            try:
                os.unlink(tmp_filename)
            except OSError:
                pass
        raise

    return filename


def current_fingerprint(device=None) -> dict:
    """Collect the only CUDA-dependent cache fingerprint fields."""
    import torch

    from humming import __version__

    capability = torch.cuda.get_device_capability(device)
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "sm_version": capability[0] * 10 + capability[1],
        "num_sms": torch.cuda.get_device_properties(device).multi_processor_count,
        "humming_version": __version__,
        "torch_cuda_version": torch.version.cuda,
    }
