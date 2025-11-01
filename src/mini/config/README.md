# mini.config

`mini.config` keeps configuration logic inside regular Python modules. Start with a single dictionary for quick experiments, then scale into parameterised templates, inheritance chains, and CLI-driven overrides—without learning a new configuration language.

> Everything you write is still Python: functions, conditionals, list comprehensions, imports. The library focuses on loading, layering, and mutating dictionaries so you can slot the result into whatever workflow you already have.

## Highlights

- Load any Python config module or callable with `load`.
- Compose configs via `parents = [...]` chains or by supplying multiple paths at once.
- Parameterise configs with function arguments; pass runtime values with the `params` argument to `load`.
- Adjust values on the fly using `apply_overrides` and a compact CLI-friendly syntax.
- Control merge behaviour with the `Delete()` and `Replace(value)` verbs.
- Snapshot final dictionaries back to Python files with `dump`, or pretty-print them using `format`.

---

## Quick Tour

1. **Define a base config** as a plain dictionary:

    ```python
    # configs/base.py
    config = {
        "model": {
            "encoder": {"channels": 64},
            "head": {"in_channels": 64, "out_channels": 10},
        },
        "optimizer": {"type": "adam", "lr": 3e-4},
        "trainer": {"max_steps": 50_000},
    }
    ```

2. **Extend it with parents and parameters** when the project grows:

    ```python
    # configs/finetune.py
    parents = ["base.py"]

    def config(num_classes=10, max_steps=10_000, warmup_steps=1_000):
        return {
            "model": {
                "head": {"out_channels": num_classes},
            },
            "scheduler": {
                "type": "linear_warmup_cosine_decay",
                "warmup_steps": warmup_steps,
                "decay_steps": max_steps - warmup_steps,
            },
            "trainer": {"max_steps": max_steps},
        }
    ```

3. **Load everything** and apply runtime tweaks:

    ```python
    from mini.config import apply_overrides, load

    cfg = load("configs/finetune.py", params={"num_classes": 5})
    cfg = apply_overrides(cfg, ["optimizer.lr=1e-3"])
    ```

4. **Feed the dictionary wherever you need it**—serialise to JSON, log to disk, or hand it to any factory you already use. Nothing in `mini.config` dictates how values get consumed later.

    ```python
    import json

    with open("runs/finetune_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    ```

    `cfg` stays an ordinary dictionary throughout, so you can reuse it in scripts, notebooks, or third-party libraries.

---

## Anatomy of a Config Module

Each config file is just Python. The loader only pays attention to two attributes:

- **`config` attribute**: dictionary or callable returning a dictionary.
- **`parents` attribute**: string or list of strings pointing to other config files (paths are resolved relative to the current file).

Anything else you import or compute at import time is yours to use.

### Dict vs Callable

- Use a **dictionary** (`config = {...}`) when the configuration is static.
- Use a **callable** when you want parameters. Default argument values become part of the config defaults automatically.

```python
def config(batch_size: int = 64, *, device: str = "cuda"):
    return {
        "data": {"batch_size": batch_size},
        "trainer": {"device": device},
    }
```

### Multiple Parents

```python
# configs/experiment.py
parents = ["base.py", "schedules/cosine.py"]

config = {
    "trainer": {"max_epochs": 40},
}
```

Parents are loaded depth-first (left to right), so later parents override earlier ones before the child applies its updates.

### Loading Several Files at Once

`load` also accepts a `Sequence[path]`. The behaviour mirrors parent chaining but is convenient when you want to compose parts dynamically:

```python
cfg = load(
    [
        "configs/base.py",
        "configs/backbones/resnet.py",
        "configs/modes/eval.py",
    ]
)
```

Again, they are loaded left to right, so later paths override earlier ones.

---

## Runtime Overrides

`apply_overrides(config_dict, sequence_of_strings)` mutates the dictionary in place. Each string uses a compact syntax designed for CLI usage.

### Supported Operations

- `path=value` → assign (dict keys or list indices)
- `path+=value` → append to a list
- `path-=value` → remove a matching element from a list
- `path!=` → delete a key or remove a list index

Examples:

```python
apply_overrides(
    cfg,
    [
        "optimizer.lr=5e-4",
        "trainer.max_steps=10_000",
        "trainer.hooks+='wandb'",
        "trainer.hooks-='checkpoint'",
        "data.pipeline[0]!=",  # drop item at index 0
    ],
)
```

Values are parsed with `ast.literal_eval`, so strings, numbers, booleans, lists, dictionaries, and `None` all work (as long as they don't contain any whitespace). If parsing fails, the raw string is used, so quoting strings is usually not necessary.

### CLI Integration

```python
# cli.py
import argparse
from mini.config import apply_overrides, load

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/finetune.py")
parser.add_argument("--overrides", nargs="*", default=[])
args = parser.parse_args()

cfg = load(args.config)
cfg = apply_overrides(cfg, args.overrides)
```

Run:

```bash
python cli.py --overrides optimizer.lr=2e-4 trainer.hooks+=logging
```

---

## Merge Semantics and Sentinels

When configs are layered, `mini.config` walks the override dictionary and combines it with the base using the following rules:

- If both base and override values are dictionaries, merge them recursively.
- If the override value is `Delete()`, remove the key entirely (note that this is different from assigning e.g., a `None` value).
- If the override value is `Replace(value)`, use `value` as-is without recursive merging.
- Otherwise, assign the override value directly. Lists, tuples, numbers, strings, and other types always replace the prior value.

```python
from mini.config import Delete, Replace, merge

base = {
    "optimizer": {
        "lr": 3e-4,
        "weight_decay": 0.01,
        "schedule": {"type": "linear", "warmup": 1_000},
    },
    "trainer": {"hooks": ["progress", "checkpoint"]},
}

override = {
    "optimizer": {
        "weight_decay": Delete(),
        "schedule": Replace({"type": "cosine", "t_max": 20_000}),  # whole dict swapped
    },
    "trainer": {"steps": 10_000, "hooks": ["progress"]},
}

merged = merge(base, override)
# =>
# {
#   "optimizer": {"lr": 3e-4, "schedule": {"type": "cosine", "t_max": 20_000}},
#   "trainer": {"steps": 10_000, "hooks": ["progress"]},
# }
```

`merge` is exported in case you want to reuse the merge algorithm elsewhere, but `load` and `apply_overrides` already rely on it internally.

---

## Formatting and Snapshots

Freeze the exact configuration you ran:

```python
from pathlib import Path
from mini.config import dump, format

print(format(cfg))  # Black-formatted string
dump(cfg, Path("runs/2024-01-10/config_snapshot.py"))
```

- `format` returns a nicely formatted string—useful for logging.
- `dump` writes the same representation to disk with a short header and `# fmt: off`. Because the file is valid Python, you can load it again with `load`.

---

## Tips for Structuring Configs

- **Organise by concern**: `configs/data/imagenet.py`, `configs/optim/adamw.py`, `configs/modes/eval.py`.
- **Expose helper functions** alongside `config` for reusable snippets (e.g., `def build_backbone(depth): ...`).
- **Pair with object builders only if you want to**: the output is a plain dict, so you can plug it into your own factories, serialise it, or convert it to another format.
- **Prefer parameters** over repeated overrides when you always tweak the same value.

---

## API Reference

- `load(path | Sequence[path], params: dict | None = None) -> dict`
  - Imports each file, resolves parents recursively, merges dictionaries (last wins), and returns the final dictionary. When `config` is callable, it is invoked with merged defaults plus provided `params`.
- `apply_overrides(cfg: dict, overrides: Sequence[str]) -> dict`
  - Applies CLI-style mutations in place and returns the dictionary for convenience.
- `merge(base: dict, override: dict) -> dict`
  - Recursive merge helper exposed for advanced use-cases.
- `Delete()`
  - Verb that deletes keys during merges.
- `Replace(value)`
  - Verb instructing the merge logic to replace a node instead of merging deeper.
- `dump(cfg: dict, path: os.PathLike) -> None`
  - Writes `cfg` to disk using Black formatting.
- `format(cfg: dict) -> str`
  - Returns a Black-formatted string representation.
