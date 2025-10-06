import functools
import importlib
from typing import Any, Callable, Dict, Type, Union


class Registry:
    def __init__(self):
        self._store = {}

    def register(self, name: str = None) -> Callable:
        def wrapper(cls):
            key = name or cls.__name__
            self._store[key] = cls
            return cls

        return wrapper

    def get(self, name: str) -> Type:
        return self._store.get(name)


REGISTRY = Registry()


def build_from_cfg(
    cfg: Union[Dict, list, tuple, Any],
    registry: Registry | None = REGISTRY,
    recursive: bool = False,
) -> Any:
    """Build an object from a configuration dictionary.

    ```python
    @REGISTRY.register()
    class Layer:
        def __init__(self, units: int):
            self.units = units

    @REGISTRY.register()
    class Backbone:
        def __init__(self, layers: list):
            self.layers = layers

    @REGISTRY.register()
    class Model:
        def __init__(self, backbone: Backbone, name: str):
            self.backbone = backbone
            self.name = name

    @REGISTRY.register()
    def l1_loss(x, y, weight):
        return (x - y).abs() * weight

    cfg = {
        'type': 'Model',
        'name': 'ResNet',
        'backbone': {
            'type': 'Backbone',
            'layers': [
                {'type': 'Layer', 'units': 64},
                {'type': 'Layer', 'units': 128},
            ]
        },
        'loss': {
            'type': 'l1_loss',
            'weight': 0.5,
        }
    }

    model = build_from_cfg(cfg, registry=REGISTRY, recursive=True)
    ```
    """
    if not recursive:
        return _instantiate(cfg, registry)

    return _recursive_build(cfg, registry)


def _recursive_build(obj: Any, registry: Registry | None) -> Any:
    if isinstance(obj, dict):
        if "type" in obj:
            # Instantiable object config
            obj = obj.copy()
            for k, v in obj.items():
                if k != "type":
                    obj[k] = _recursive_build(v, registry)
            return _instantiate(obj, registry)
        else:
            # Non-instantiable dict.
            # Recurse, but preseve the original type (e.g., OrderedDict, EasyDict, etc.).
            return type(obj)((k, _recursive_build(v, registry)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        return type(obj)(_recursive_build(v, registry) for v in obj)
    else:
        return obj  # primitive types


def _instantiate(cfg: Dict[str, Any], registry: Registry | None = None) -> Any:
    assert isinstance(cfg, dict) and "type" in cfg, (
        "Config must be a dict with a 'type' key."
    )

    cfg = cfg.copy()
    obj_type = cfg.pop("type")

    partial_prefix = "partial:"
    partial = obj_type.startswith(partial_prefix)
    obj_type = obj_type.removeprefix(partial_prefix)

    if registry and registry.get(obj_type):
        obj = registry.get(obj_type)
    else:
        assert "." in obj_type, (
            f"Object type '{obj_type}' is neither registered nor specifies a module path"
        )
        module_path, obj_name = obj_type.rsplit(".", 1)
        module = importlib.import_module(module_path)
        obj = getattr(module, obj_name)

    pos_args = cfg.pop("*", [])
    if partial:
        return functools.partial(obj, *pos_args, **cfg)
    else:
        return obj(*pos_args, **cfg)
