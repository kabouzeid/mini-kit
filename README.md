<p align="center">
  <img src="https://github.com/user-attachments/assets/e8f5ce11-1b57-49d0-9d20-a3ee587a4498" height=200/>
</p>

# mini-kit

Minimal, hackable building blocks for deep learning projects. The goal is to keep every layer small enough that you can read it, tweak it, or drop it into your own codebase.

[![PyPI version](https://img.shields.io/pypi/v/mini-kit.svg)](https://pypi.python.org/pypi/mini-kit)

## Quick Start

```bash
pip install mini-kit
```

You now have three tiny helpers that play nicely together:

1. `mini.config` loads plain-Python configs. Start with a single dictionary, then grow into composable templates and parent chains without learning a new DSL.
2. `mini.builder` turns dictionaries into Python objects. Supports both registry shortcuts and fully qualified import paths.
3. `mini.trainer` provides a lightweight training framework with hooks for logging, checkpointing, and customization.

The example below shows `mini.config` and `mini.builder` in action.

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
from mini.builder import register, build
from mini.config import apply_overrides, load_config

@register()
class Encoder:
    def __init__(self, channels: int):
        self.channels = channels

@register()
class Classifier:
    def __init__(self, encoder, head):
        self.encoder = encoder
        self.head = head

cfg = load_config("configs/model.py")
cfg = apply_overrides(cfg, ["optimizer.lr=1e-3", "model.encoder.channels=128"])

model = build(cfg["model"])
optimizer = build(cfg["optimizer"])
```

- `load_config` executes `configs/model.py`; keep a simple `config = {...}` for small projects, or swap to `def config(...):` and `parents = [...]` when you need templates and composition.
- `apply_overrides` allows for painless command-line overrides: tweak nested keys with a short-hand syntax: `optimizer.lr=...`, append with `+=`, or drop entries with `!=`.
- `build` looks at the `"type"` key, grabs the right constructor (from the registry or import path), and wires up dependencies for you.

More details are in each subpackage's own README:

- `mini.config`: [README](src/mini/config/README.md)
- `mini.builder`: [README](src/mini/builder/README.md)
- `mini.trainer`: [README](src/mini/trainer/README.md)
