import dataclasses
import json
import re
from enum import Enum
from typing import Any, ClassVar

from humming import dtypes


def name_to_google_cpp_const_style(name: str) -> str:
    if not name:
        return ""
    name = name.strip().lower()
    words = re.split(r"[_ \W]+", name)
    pascal_words = [word.capitalize() for word in words if word]
    return "k" + "".join(pascal_words)


def name_value_to_google_cpp_const_style(name: str, value: Any, keep_name: bool = False) -> str:
    if not keep_name:
        name = name_to_google_cpp_const_style(name)
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif isinstance(value, float):
        value = str(value) + "f"
    elif isinstance(value, int):
        value = str(value) + "u"
    elif isinstance(value, Enum):
        value = f"{value.__class__.__name__}::{value.name.removesuffix('_')}"
    else:
        value = str(value).replace(".", "::")

    return f"static constexpr auto {name} = {value};"


def name_value_to_extern_const_style(name: str, value: Any) -> str:
    name = name.upper()
    if isinstance(value, (bool, int)):
        value = int(value)
        return f'extern "C" __constant__ uint32_t {name} = {value};'
    return ""


def name_value_to_macro_style(name: str, value: Any) -> str:
    name = name.upper()
    if isinstance(value, (bool, int)):
        value = int(value)
        return f"#define HUMMING_{name.upper()} {int(value)}"
    return ""


@dataclasses.dataclass
class BaseHummingConfig:
    _name_map: ClassVar[dict[str, str]] = {}
    _cpp_extra_names: ClassVar[tuple[str, ...]] = ()

    def __post_init__(self):
        pass

    def to_template_args(self) -> dict[str, Any]:
        template_args = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool):
                value = int(value)
            elif isinstance(value, Enum):
                value = value.name
            elif isinstance(value, dtypes.DataType):
                value = value.to_cpp_str()
            template_args[field.name] = value
        return template_args

    def to_cpp_str(
        self,
        cls: type["BaseHummingConfig"] | None = None,
        include_class_name: bool = False,
    ) -> str:
        cls = cls or self.__class__
        str_list = []
        names = [x.name for x in dataclasses.fields(cls)]
        names += list(cls._cpp_extra_names)
        for name in names:
            value = getattr(self, name)
            if not isinstance(value, (bool, int, Enum)):
                continue
            keep_name = name in cls._name_map
            if keep_name:
                name = cls._name_map[name]
            line = name_value_to_google_cpp_const_style(name, value, keep_name)
            str_list.append(line)

        code = "\n".join("  " + x for x in str_list)
        class_name = cls.__name__

        if include_class_name:
            code = f"class {class_name} {{\n{code}\n}};"

        return code

    def to_macro_cpp_str(self, cls: type["BaseHummingConfig"] | None = None) -> str:
        cls = cls or self.__class__
        str_list = []
        names = [x.name for x in dataclasses.fields(cls)]
        names += list(cls._cpp_extra_names)
        for name in names:
            value = getattr(self, name)
            if not isinstance(value, (bool, int, Enum)):
                continue
            line = name_value_to_macro_style(name, value)
            str_list.append(line)

        str_list = [x for x in str_list if x]
        code = "\n".join(x for x in str_list if x)

        return code

    def to_extern_cpp_str(self, cls: type["BaseHummingConfig"] | None = None) -> str:
        cls = cls or self.__class__
        str_list = []
        names = [x.name for x in dataclasses.fields(cls)]
        names += list(cls._cpp_extra_names)
        for name in names:
            value = getattr(self, name)
            if not isinstance(value, (bool, int, Enum)):
                continue
            line = name_value_to_extern_const_style(name, value)
            str_list.append(line)

        str_list = [x for x in str_list if x]
        code = "\n".join(x for x in str_list if x)

        return code

    def to_str(self) -> str:
        res = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Enum):
                value = value.value
            elif isinstance(value, dtypes.DataType):
                value = str(value)
            res[field.name] = value

        return json.dumps(res)
