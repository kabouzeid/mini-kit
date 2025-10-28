# mini.builder

`mini.builder` turns configuration dictionaries into Python objects. It provides a simple registry system and supports fully qualified import paths, making it easy to wire up dependencies from config files without hard-coding factory logic.

> The builder is designed to work seamlessly with `mini.config`, but it's completely standalone—just pass it any dictionary with a `"type"` key and it will instantiate the object for you.

## Highlights

- Register classes and functions with `@REGISTRY.register()` for shorthand references.
- Instantiate registered types or import directly from module paths (e.g., `"torch.nn.Linear"`).
- Recursively build nested objects when `recursive=True`.
- Specify positional arguments with the special `"*"` key.
- Create partial functions or classes using the `"partial:"` prefix.
- Use custom registries to isolate different sets of components.

---

## Quick Tour

1. **Register your classes and functions:**

    ```python
    from mini.builder import REGISTRY

    @REGISTRY.register()
    class Encoder:
        def __init__(self, channels: int):
            self.channels = channels

    @REGISTRY.register()
    class Classifier:
        def __init__(self, encoder, head):
            self.encoder = encoder
            self.head = head
    ```

2. **Build objects from config dictionaries:**

    ```python
    from mini.builder import build_from_cfg

    cfg = {
        "type": "Classifier",
        "encoder": {"type": "Encoder", "channels": 64},
        "head": {"type": "torch.nn.Linear", "*": [64, 10]},
    }

    model = build_from_cfg(cfg, recursive=True)
    ```

    The `build_from_cfg` function:
    - Looks up `"Classifier"` in the registry
    - Recursively builds nested configs when `recursive=True`
    - Falls back to import paths like `"torch.nn.Linear"` when not in the registry

3. **No registration needed for standard library or third-party classes:**

    ```python
    cfg = {
        "type": "torch.optim.AdamW",
        "lr": 3e-4,
        "weight_decay": 0.01,
    }

    optimizer = build_from_cfg(cfg)
    ```

    If a type isn't registered, `build_from_cfg` treats it as a fully qualified module path.

---

## Core Features

### The Registry

A registry is just a name-to-class mapping. Use the decorator to add entries:

```python
from mini.builder import REGISTRY, build_from_cfg

@REGISTRY.register()
class MyLayer:
    def __init__(self, units: int):
        self.units = units

@REGISTRY.register("CustomName")
class AnotherLayer:
    def __init__(self, units: int):
        self.units = units
```

- By default, the class name is used as the key (e.g., `"MyLayer"`).
- You can provide a custom name: `@REGISTRY.register("CustomName")`.

Build objects using the registered name:

```python
layer = build_from_cfg({"type": "MyLayer", "units": 128})
custom = build_from_cfg({"type": "CustomName", "units": 256})
```

### Module Paths

When a type isn't in the registry, `build_from_cfg` treats it as a fully qualified import path:

```python
cfg = {
    "type": "collections.OrderedDict",
    "*": [[("a", 1), ("b", 2)]],
}

od = build_from_cfg(cfg)
```

This is useful for standard library classes, third-party modules, or when you want explicit imports without polluting the registry.

### Recursive Building

Set `recursive=True` to traverse nested dictionaries, lists, and tuples. Any dict with a `"type"` key gets instantiated:

```python
cfg = {
    "type": "Model",
    "encoder": {"type": "Encoder", "channels": 64},
    "layers": [
        {"type": "Layer", "units": 128},
        {"type": "Layer", "units": 256},
    ],
}

model = build_from_cfg(cfg, recursive=True)
```

Without `recursive=True`, only the top-level dict is instantiated—nested configs remain as plain dictionaries.

### Positional Arguments

Use the special `"*"` key to pass positional arguments:

```python
# torch.nn.Linear expects (in_features, out_features)
cfg = {
    "type": "torch.nn.Linear",
    "*": [128, 10],
}

layer = build_from_cfg(cfg)
```

You can mix positional and keyword arguments:

```python
cfg = {
    "type": "torch.nn.Linear",
    "*": [128, 10],
    "bias": False,
}

layer = build_from_cfg(cfg)
```

### Partial Instantiation

Prefix the type with `"partial:"` to create a `functools.partial` object instead of calling the constructor:

```python
@REGISTRY.register()
def loss_fn(pred, target, weight):
    return ((pred - target) ** 2).mean() * weight

cfg = {"type": "partial:loss_fn", "weight": 0.5}
partial_loss = build_from_cfg(cfg)

# Later, call it with the remaining arguments
loss = partial_loss(pred, target)
```

This works with classes too:

```python
cfg = {"type": "partial:torch.optim.Adam", "lr": 1e-3}
optimizer_factory = build_from_cfg(cfg)

# Create the optimizer when you have the model parameters
optimizer = optimizer_factory(model.parameters())
```

### Custom Registries

You can create separate registries to avoid naming conflicts or isolate different sets of components:

```python
from mini.builder import Registry, build_from_cfg

my_registry = Registry()

@my_registry.register()
class CustomComponent:
    def __init__(self, x):
        self.x = x

cfg = {"type": "CustomComponent", "x": 42}
obj = build_from_cfg(cfg, registry=my_registry)
```

Pass `registry=None` to disable registry lookups entirely—only module paths will work.

---

## API Reference

### `REGISTRY`

The default global registry instance. Use `@REGISTRY.register()` to add entries.

### `Registry`

A simple registry class for mapping names to objects (classes or functions).

**Methods:**

- `register(name: str | None = None) -> Callable`
  - Decorator that registers a class or function. If `name` is omitted, the object's `__name__` is used.
- `get(name: str) -> Type`
  - Retrieves a registered object by name. Returns `None` if not found.

### `build_from_cfg`

```python
build_from_cfg(
    cfg: dict | list | tuple | Any,
    registry: Registry | None = REGISTRY,
    recursive: bool = False,
) -> Any
```

Instantiates an object from a configuration dictionary.

**Parameters:**

- `cfg`: Configuration dict (must have a `"type"` key) or any value when `recursive=True`.
- `registry`: Registry to use for name lookups. Pass `None` to disable registry lookups.
- `recursive`: When `True`, recursively builds nested configs. When `False`, only builds the top-level object.

**Returns:**

The instantiated object.

**Raises:**

- `AssertionError` if `cfg` is not a dict or lacks a `"type"` key.
- `ModuleNotFoundError` if the type string is neither registered nor a valid module path.

---

## Tips

- **Organize by domain**: Keep separate registries for models, optimizers, schedulers, etc., if you have naming conflicts.
- **Prefer explicit paths** for standard library or third-party classes—it makes configs more portable.
- **Use recursive building** for complex nested structures. Without it, you'd need to manually build each layer.
- **Combine with `mini.config`** to load configs from Python files and build objects in one go.

---

## Integration Example

Using `mini.config` and `mini.builder` together:

```python
# configs/model.py
config = {
    "model": {
        "type": "Classifier",
        "encoder": {"type": "Encoder", "channels": 64},
        "head": {"type": "torch.nn.Linear", "*": [64, 10]},
    },
    "optimizer": {"type": "torch.optim.AdamW", "lr": 3e-4},
}
```

```python
# main.py
from mini.builder import REGISTRY, build_from_cfg
from mini.config import load_config

@REGISTRY.register()
class Encoder:
    def __init__(self, channels: int):
        self.channels = channels

@REGISTRY.register()
class Classifier:
    def __init__(self, encoder, head):
        self.encoder = encoder
        self.head = head

cfg = load_config("configs/model.py")
model = build_from_cfg(cfg["model"], recursive=True)
optimizer = build_from_cfg(cfg["optimizer"], params=model.parameters())
```

The config stays declarative, and the builder handles instantiation automatically.
