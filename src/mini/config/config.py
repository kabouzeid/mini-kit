import ast
import importlib.util
import os
import re
import sys
from contextvars import ContextVar
from functools import reduce
from pathlib import Path
from typing import TypeVar

# Just in case: we use a ContextVar instead of a plain global, so injected params stay local per thread / async task.
_params_ctx: ContextVar[dict] = ContextVar("mini_config_params", default={})

T = TypeVar("T")

DELETE = object()  # Sentinel value to delete config entries


def param(name: str, default: T) -> T:
    """Return injected variable value or provided default.

    Usage inside config python file:
        from mini.config import param
        total_steps = param("total_steps", 10000)
    """
    return _params_ctx.get().get(name, default)


def params_dict() -> dict:
    """Return a dict of injected variable values."""
    return _params_ctx.get().copy()


def load_merged_config(paths: list[os.PathLike], params: dict | None = None):
    return reduce(deep_merge_dicts, (load_config(p, params) for p in paths))


def load_config(path: os.PathLike, params: dict | None = None):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("config_module", path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    params = params or {}
    token = _params_ctx.set(params)
    try:
        spec.loader.exec_module(config_module)
    finally:
        _params_ctx.reset(token)
    config = getattr(config_module, "config", {})
    parents = getattr(config_module, "parents", None)
    params = deep_merge_dicts(getattr(config_module, "params", {}), params)
    delete = getattr(config_module, "delete", [])

    if isinstance(parents, str):
        parents = [parents]
    if parents:
        parent_config = load_merged_config(
            [path.parent / Path(p) for p in parents], params
        )
        for key in delete:
            delete_nested(parent_config, parse_key_path(key))
        config = deep_merge_dicts(parent_config, config)

    return config


def save_config(config: dict, path: os.PathLike):
    import black

    # we could also use pprint.pformat, but black looks nicer
    config_str = black.format_str("config = " + repr(config), mode=black.Mode())

    with open(path, "w") as f:
        f.write("# fmt: off\n")  # prevent auto-formatting
        f.write("# Auto-generated config snapshot\n")
        f.write(config_str)


def format_config(config: dict) -> str:
    import black

    return black.format_str(repr(config), mode=black.Mode())


def deep_merge_dicts(base: dict, override: dict):
    base = base.copy()
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = deep_merge_dicts(base[k], v)
        elif v is DELETE:
            base.pop(k, None)
        else:
            base[k] = v
    return base


def apply_overrides(cfg: dict, overrides: list[str]):
    for override in overrides:
        if "+=" in override:
            key, value = override.split("+=", 1)
            keys = parse_key_path(key)
            append_to_nested(cfg, keys, infer_type(value))
        elif "!=" in override:
            key, _ = override.split("!=", 1)
            keys = parse_key_path(key)
            delete_nested(cfg, keys)
        elif "-=" in override:
            key, value = override.split("-=", 1)
            keys = parse_key_path(key)
            remove_value_from_list(cfg, keys, infer_type(value))
        else:
            key, value = override.split("=", 1)
            keys = parse_key_path(key)
            set_nested(cfg, keys, infer_type(value))
    return cfg


def parse_key_path(path: str):
    """Parse 'a.b[0].c' → ['a', 'b', 0, 'c']"""
    tokens = []
    parts = re.split(r"(\[-?\d+\]|\.)", path)
    for part in parts:
        if not part or part == ".":
            continue
        if part.startswith("[") and part.endswith("]"):
            tokens.append(int(part[1:-1]))
        else:
            tokens.append(part)
    return tokens


def set_nested(d: dict, keys, value):
    for i, key in enumerate(keys):
        is_last = i == len(keys) - 1
        if isinstance(key, int):
            while len(d) <= key:
                d.append(None)
            if is_last:
                d[key] = value
            else:
                if d[key] is None:
                    d[key] = {} if isinstance(keys[i + 1], str) else []
                d = d[key]
        else:
            if is_last:
                d[key] = value
            else:
                if key not in d or d[key] is None:
                    d[key] = {} if isinstance(keys[i + 1], str) else []
                d = d[key]


def append_to_nested(d: dict, keys, value):
    for i, key in enumerate(keys):
        is_last = i == len(keys) - 1
        next_key_type = type(keys[i + 1]) if not is_last else None

        if isinstance(key, int):
            while len(d) <= key:
                d.append(None)
            if is_last:
                if d[key] is None:
                    d[key] = []
                if not isinstance(d[key], list):
                    raise ValueError(f"Target at index {key} is not a list")
                d[key].append(value)
            else:
                if d[key] is None:
                    d[key] = {} if next_key_type is str else []
                d = d[key]
        else:
            if is_last:
                if key not in d or not isinstance(d[key], list):
                    d[key] = []
                d[key].append(value)
            else:
                if key not in d or d[key] is None:
                    d[key] = {} if next_key_type is str else []
                d = d[key]


def delete_nested(d: dict, keys):
    for i, key in enumerate(keys[:-1]):
        d = d[key]
    last_key = keys[-1]
    if isinstance(last_key, int):
        if isinstance(d, list) and 0 <= last_key < len(d):
            del d[last_key]
    else:
        d.pop(last_key, None)


def remove_value_from_list(d: dict, keys, value):
    for key in keys:
        d = d[key]
    if isinstance(d, list) and value in d:
        d.remove(value)
    # TODO: this should probably raise if d is not a list


def infer_type(val: str):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val
