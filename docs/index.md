---
icon: lucide/rocket
---

# mini-kit

Minimal, hackable building blocks for deep learning projects. Each layer is small enough to read, tweak, or drop into your own codebase.

## Quick start

```bash
pip install mini-kit
```

You now have three helpers that work together:

- `mini.config` loads plain-Python configs with parents, parameters, and CLI overrides.
- `mini.builder` turns dictionaries into Python objects via registries or dotted import paths.
- `mini.trainer` runs a lightweight PyTorch training loop with hooks for logging and checkpointing.

### Minimal example

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
from mini.builder import register, build
from mini.config import apply_overrides, load

@register()
class Encoder:
    def __init__(self, channels: int):
        self.channels = channels

@register()
class Classifier:
    def __init__(self, encoder, head):
        self.encoder = encoder
        self.head = head

cfg = load("configs/model.py")
cfg = apply_overrides(cfg, ["optimizer.lr=1e-3", "model.encoder.channels=128"])

model = build(cfg["model"])
optimizer = build(cfg["optimizer"])
```

- `load` executes Python config files; `apply_overrides` tweaks nested keys with CLI-friendly syntax.
- `build` resolves `"type"` from a registry or import path and wires dependencies for you.

## Packages

- [mini.config](config.md): Python-first configs with parents, parameters, overrides, and snapshots.
- [mini.builder](builder.md): Registry-based factory that instantiates nested dictionaries.
- [mini.trainer](trainer.md): Minimal PyTorch trainer with hooks for progress, checkpointing, and logging.
