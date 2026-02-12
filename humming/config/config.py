import dataclasses
import re
from typing import Optional

from humming.config.base import BaseHummingConfigClass
from humming.config.enum import ActivationType, MmaType


@dataclasses.dataclass
class SchedulerConfig(BaseHummingConfigClass):
    use_stream_k: bool = True


@dataclasses.dataclass
class PipelineConfig(BaseHummingConfigClass):
    use_warp_spec: Optional[bool] = None
    num_stages: int = 2
    num_threads: Optional[int] = None
    num_math_threads: Optional[int] = None
    num_load_threads: Optional[int] = None
    use_mbarrier: Optional[bool] = None
    use_cp_async: Optional[bool] = None
    use_tma: Optional[bool] = None
    use_tma_a: Optional[bool] = None
    use_tma_b: Optional[bool] = None
    use_tma_c: Optional[bool] = None
    use_tma_bs: Optional[bool] = None
    use_tma_bzp: Optional[bool] = None
    use_tma_bias: Optional[bool] = None

    _name_map = {
        "use_mbarrier": "kUseMBarrier",
        "use_tma_bs": "kUseTmaBS",
        "use_tma_bzp": "kUseTmaBZP",
    }

    @classmethod
    def from_dict(cls, raw_config):
        clean_config = cls._preprocess_dict(raw_config)
        config = cls(**clean_config)

        assert config.num_math_threads is not None

        if config.use_warp_spec is None:
            config.use_warp_spec = raw_config["sm_version"] >= 90

        if config.use_warp_spec:
            config.num_load_threads = 128
            config.num_threads = config.num_math_threads + 128
        else:
            config.num_load_threads = config.num_math_threads
            config.num_threads = config.num_math_threads

        if config.use_tma is None:
            config.use_tma = raw_config["sm_version"] >= 90

        for name in dir(config):
            if not name.startswith("use_tma_"):
                continue
            if not config.use_tma:
                assert getattr(config, name) is not True
            if getattr(config, name) is None:
                setattr(config, name, config.use_tma)

        if config.use_mbarrier is None:
            config.use_mbarrier = config.use_warp_spec or config.use_tma

        if config.use_cp_async is None:
            config.use_cp_async = raw_config["sm_version"] >= 80

        return config


@dataclasses.dataclass
class QuantParamConfig(BaseHummingConfigClass):
    has_input_scale: Optional[bool] = None
    has_weight_scale: bool = True
    input_scale_group_size: int = 0
    weight_scale_group_size: int = 0
    has_global_scale: bool = False
    has_dynamic_zero_point: bool = False


@dataclasses.dataclass
class EpilogueConfig(BaseHummingConfigClass):
    has_bias: bool = False
    activation_type: ActivationType = ActivationType.NONE

    def prepare_custom_activation_func(self, impl_text):
        if self.activation_type == ActivationType.CUSTOM:
            template = (
                "template <>\n"
                "CUDA_INLINE\n"
                "float activation_func<ActivationType::CUSTOM>(const float a) {{\n"
                "{impl_text}\n"
                "}};"
            )
        elif self.activation_type == ActivationType.CUSTOM_GLU:
            template = (
                "template <>\n"
                "CUDA_INLINE\n"
                "float activation_func<ActivationType::CUSTOM_GLU>(const float2 a) {{\n"
                "{impl_text}\n"
                "}};"
            )
        else:
            return ""

        assert isinstance(impl_text, str) and impl_text.strip()
        impl_text = impl_text.strip()
        impl_text = re.sub("\\s+\n\\s+", "\n  ", impl_text)
        return template.format(impl_text=impl_text)


@dataclasses.dataclass
class MoEConfig(BaseHummingConfigClass):
    top_k: int = 0
    is_moe: bool = False
    is_moe_down: bool = False

    _name_map = {
        "is_moe": "kIsMoE",
        "is_moe_down": "kIsMoEDown",
    }


@dataclasses.dataclass
class MmaConfig(BaseHummingConfigClass):
    mma_type: Optional[MmaType] = None
    use_f16_accum: bool = False

    @classmethod
    def from_dict(cls, raw_config):
        clean_config = cls._preprocess_dict(raw_config)
        config = cls(**clean_config)

        if config.mma_type is None:
            if raw_config["sm_version"] == 90:
                config.mma_type = MmaType.WGMMA
            elif raw_config["sm_version"] >= 75:
                config.mma_type = MmaType.MMA

        return config
