import ast
import os
import re
import runpy
from collections.abc import Mapping
from functools import reduce
from pathlib import Path
from typing import Callable, Sequence


class Delete:
    pass


class Replace:
    def __init__(self, value, /):
        self.value = value


def load(path: os.PathLike | Sequence[os.PathLike], variables: dict | None = None):
    """Load config modules from `path`, apply variables, and merge the results."""

    paths = [path] if isinstance(path, (str, os.PathLike)) else path
    specs = [spec for p in paths for spec in _collect_config_specs(Path(p))]

    # last assignment wins. we could also deep merge, but it feels less natural here
    variables = {k: v for _, d in specs for k, v in d.items()} | (variables or {})

    return reduce(
        merge,
        (_build_config(config, variables) for config in [cfg for cfg, _ in specs]),
    )


def _build_config(config: dict | Callable, variables: dict):
    return config(Variables(variables)) if callable(config) else config


def _collect_config_specs(path: os.PathLike) -> list[tuple[dict, dict]]:
    """
    Return the flattened inheritance chain for the config at `path`. Ordered from the farthest parent first.
    """
    path = Path(path).resolve()
    config_module_globs = runpy.run_path(str(path), run_name="__config__")

    config = config_module_globs.get("config", {})
    variables = config_module_globs.get("variables", {})

    parents = config_module_globs.get("parents", None)
    if isinstance(parents, str):
        parents = [parents]

    return [
        parent_cfg_specs
        for parent in parents or []
        for parent_cfg_specs in _collect_config_specs(path.parent / Path(parent))
    ] + [(config, variables)]


def dump(config: dict, path: os.PathLike):
    """Persist a config dictionary to a Python file with Black formatting."""

    import black

    # we could also use pprint.pformat, but black looks nicer
    config_str = black.format_str("config = " + repr(config), mode=black.Mode())

    with open(path, "w") as f:
        f.write("# fmt: off\n")  # prevent auto-formatting
        f.write("# Auto-generated config snapshot\n")
        f.write(config_str)


def format(config: dict) -> str:
    """Return a Black-formatted string for the provided config dictionary."""

    import black

    return black.format_str(repr(config), mode=black.Mode())


def merge(base: dict, override: dict):
    base = base.copy()
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = merge(base[k], v)
        elif isinstance(v, Delete):
            base.pop(k, None)
        elif isinstance(v, Replace):
            base[k] = v.value
        else:
            base[k] = v
    return base


def apply_overrides(cfg: dict, overrides: Sequence[str]):
    cfg = cfg.copy()
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


class Variables(Mapping):
    """
    Simple EasyDict-like read-only wrapper around a dictionary that allows attribute-style access to keys.
    """

    def __init__(self, d: dict):
        self._dict = type(d)((k, self._wrap(v)) for k, v in d.items())

    def _wrap(self, x):
        if isinstance(x, dict):
            return Variables(x)
        elif isinstance(x, list):
            return type(x)(self._wrap(it) for it in x)
        else:
            return x

    def __getattr__(self, key):
        return self._dict[key]

    def __setattr__(self, name, value):
        if name == "_dict":
            super().__setattr__(name, value)
        else:
            raise AttributeError(f"{self.__class__.__name__} is read-only")

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return repr(self._dict)
