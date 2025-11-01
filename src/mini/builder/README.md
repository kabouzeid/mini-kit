# mini.builder

`mini.builder` turns plain dictionaries into Python objects. It plays nicely with `mini.config` but also works as a standalone helper.

## Essentials

- Register classes and functions with `@register()` for short names.
- Call `build` to instantiate registered types or dotted import paths.
- Opt into recursive instantiation for nested configs.
- Use `"*"` for positional args.
- Prefix with `"partial:"` to create factories instead of calling constructors immediately.
- Create extra `Registry` instances when you want isolated component sets.

## Basic Usage

1. **Register components**

    ```python
    from mini.builder import register

    @register()
    class Encoder:
        def __init__(self, channels: int):
            self.channels = channels

    @register()
    class Classifier:
        def __init__(self, encoder, head):
            self.encoder = encoder
            self.head = head
    ```

2. **Instantiate from config**

    ```python
    from mini.builder import build

    cfg = {
        "type": "Classifier",
        "encoder": {"type": "Encoder", "channels": 64},
        "head": {"type": "torch.nn.Linear", "*": [64, 10]},
    }

    model = build(cfg)
    ```

3. **Fallback to import paths**

    ```python
    optimizer = build(
        {"type": "torch.optim.AdamW", "lr": 3e-4, "weight_decay": 0.01}
    )
    ```

## Feature Highlights

### Registry shortcuts

```python
from mini.builder import register, build

@register("Custom")
class Block: ...

build({"type": "Custom", "width": 256})
```

### Recursive structures

```python
cfg = {
    "type": "Model",
    "encoder": {"type": "Encoder", "channels": 64},
    "layers": [
        {"type": "Layer", "units": 128},
        {"type": "Layer", "units": 256},
    ],
}

model = build(cfg)
```

### Positional arguments

Use the special `"*"` key to pass positional arguments:

```python
cfg = {"type": "torch.nn.Linear", "*": [128, 10], "bias": False}
layer = build(cfg)
```

### Partial factories

```python
@register()
def loss_fn(pred, target, weight):
    return ((pred - target) ** 2).mean() * weight

loss_fn = build({"type": "partial:loss_fn", "weight": 0.5})
loss = loss_fn(pred, target)
```

### Custom registries

```python
from mini.builder import Registry, build

optim_registry = Registry()

@optim_registry.register()
class ToyOptim: ...

optim = build({"type": "ToyOptim"}, registry=optim_registry)
```

Pass `registry=None` to skip registry lookups entirely.

## API Cheatsheet

- `default_registry`: default global registry.
- `register(name: str | None = None, registry=default_registry) -> Callable`: convenience decorator for adding classes/functions to the registry.
- `Registry.register(name: str | None = None) -> Callable`: decorator for adding classes/functions (defaults to the object name).
- `Registry.get(name: str) -> Any`: fetch a registered object; returns `None` when missing.
- `build(cfg, registry=default_registry, recursive=True) -> Any`: instantiate configs, optionally recursing into nested structures.
  - Raises `AssertionError` when `cfg` lacks `"type"` and `ModuleNotFoundError` when lookups fail.

## Example Integration with `mini.config`

```python
from mini.builder import build
from mini.config import load_config

cfg = load_config("configs/model.py")
model = build(cfg["model"])
optimizer = build(cfg["optimizer"], params=model.parameters())
```
