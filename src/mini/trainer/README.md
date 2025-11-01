# mini.trainer

`mini.trainer` is a minimalist PyTorch training loop that centralizes the routine sequencing (state builds, accumulation, retries) while leaving distributed setup and model construction under user control.

## What BaseTrainer Handles

`BaseTrainer` stays intentionally small. Out of the box it:

- Builds and restores state for the data loader, model, optimizer, scheduler, and optional grad scaler in a DDP/FSDP-friendly order.
- Runs a resilient step loop that supports gradient accumulation, mixed precision, scale management, NaN/Inf guarding, and data/step timing across single or distributed devices.
- Pushes everything else into hooks. Progress output, checkpoint IO, wandb logging, CUDA memory tracking, and validation all ship as hooks, which makes it easy to compose your own.

Because nearly every behavior lives behind hooks, most real-world setups fit without forking the code. When you do need something exotic, the trainer is compact enough to copy into your project and customize.

## Key Capabilities

- **Subclass contract**: Implement the `build_*()` factories and `forward()`, then rely on the provided `train()` loop.
- **Distributed-ready**: Gradient accumulation, autocast, gradient scaling, `no_sync`, and non-finite checks work seamlessly with PyTorch DDP and FSDP.
- **State management**: Save and restore the full training state (model, optimizer, scheduler, grad scaler, current step and hooks) with correct ordering and distributed-aware handling.
- **Hook system**: Progress, checkpointing, logging (e.g., wandb), EMA, CUDA memory stats, and validation are all hooks you can combine or extend.
- **Instrumentation coverage**: Step/data timings, gradient norms, learning rates, and user metrics are captured for logging hooks to consume.

---

## Quick Start

1. **Subclass `BaseTrainer` and implement the required methods:**

    ```python
    from mini.trainer import BaseTrainer
    import torch
    import torch.nn as nn

    class MyTrainer(BaseTrainer):
        def build_data_loader(self):
            # Return any iterable (DataLoader, generator, etc.)
            dataset = ...
            return torch.utils.data.DataLoader(dataset, batch_size=32)

        def build_model(self):
            model = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
            )
            return model.to(self.device)

        def build_optimizer(self):
            return torch.optim.Adam(self.model.parameters(), lr=1e-3)

        def forward(self, input):
            # Implement your forward pass and loss computation
            x, y = input  # comes from data loader
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = nn.functional.cross_entropy(logits, y)
            
            # Return loss and a dict of metrics to log
            records = {"accuracy": (logits.argmax(1) == y).float().mean().item()}
            return loss, records
    ```

2. **Create and train:**

    ```python
    trainer = MyTrainer(
        max_steps=10_000,
        grad_clip=1.0,
        max_non_finite_grad_retries=3,
        mixed_precision="bf16",  # or "fp16", or None
        gradient_accumulation_steps=4,
        workspace="./runs/experiment_1",
        device="cuda",
    )

    trainer.train()
    ```

3. **Add hooks for logging, checkpointing, and more:**

    ```python
    from mini.trainer import BaseTrainer, ProgressHook, CheckpointingHook, LoggingHook

    class MyTrainer(BaseTrainer):
        # ... (same as above)

        def build_hooks(self):
            return [
                ProgressHook(interval=10, with_records=True),
                CheckpointingHook(interval=1000, path=self.workspace / "checkpoints"),
                LoggingHook(interval=100),
                WandbHook(project="my-project"),  # requires LoggingHook
            ]
    ```

---

## Why mini.trainer?

- **Lightning** takes a framework-style approach: the user defines modules and callbacks while Lightning constructs the training loop, manages distributed wrappers, and coordinates evaluation. This removes boilerplate but makes it harder to diverge from the framework’s lifecycle or debug lower-level issues when the defaults do not match your setup.
- **Hugging Face Accelerate** sits in the middle: you still write the loop, yet the library configures devices, mixed precision, and distributed collectives through helper objects. That consolidation keeps the API uniform across hardware, but the indirection can obscure which PyTorch primitives run at each stage. Debugging becomes hard, as documentation is sparse and the code is complex.
- **mini.trainer** assumes the complementary responsibilities. You hold on to device setup, process groups, and any model wrapping, while the trainer focuses on sequencing: it builds components in the correct order, runs the step loop with accumulation and retry policies, and surfaces metrics to hooks. The implementation stays small enough to audit or copy when a project needs to diverge.

---

## Core Concepts

### The Training Loop

`BaseTrainer.train()` orchestrates the training process:

1. Calls `_build()` to construct the model, optimizer, data loader, and hooks.
2. Iterates over `max_steps`, calling `_run_step()` for each step.
3. Handles gradient accumulation automatically.
4. Invokes hooks at key points (before/after train, before/after step).

You typically don't override `train()` itself; implement the `build_*()` and `forward()` methods instead.

### Required Methods

Subclasses must implement:

- **`build_data_loader() -> Iterable`**: Return an iterable that yields training data.
- **`build_model() -> nn.Module`**: Construct and return the model.
- **`build_optimizer() -> torch.optim.Optimizer`**: Create the optimizer.
- **`forward(input) -> tuple[torch.Tensor | None, dict]`**: Perform a forward pass and return `(loss, records)`.
  - `loss`: The scalar loss tensor. If `None`, the backward pass is skipped and the step is retried.
  - `records`: A nested dict of numeric metrics that you want to log (e.g., `{"accuracy": 0.95, "metrics": {"f1": 0.9}}`).

### Optional Methods

You can override these to customize behavior:

- **`build_lr_scheduler() -> torch.optim.lr_scheduler.LRScheduler | None`**: Return a learning rate scheduler (default: `None`).
- **`build_grad_scaler() -> torch.amp.GradScaler`**: Customize the gradient scaler for mixed precision (default: enabled for FP16).
- **`build_hooks() -> list[BaseHook]`**: Return a list of hooks (default: `[]`).

### Gradient Accumulation

Set `gradient_accumulation_steps` to accumulate gradients over multiple forward/backward passes before updating parameters. The trainer automatically:

- Scales the loss by `1 / gradient_accumulation_steps`.
- Calls `model.no_sync()` during accumulation steps (if using DDP/FSDP) to skip gradient synchronization until the final step.

### Mixed Precision

Pass `mixed_precision="fp16"` or `"bf16"` to enable automatic mixed precision training with `torch.autocast`. The trainer:

- Uses `torch.amp.GradScaler` for FP16 to handle gradient scaling.
- Disables the scaler for BF16 (which doesn't need gradient scaling).

### Non-Finite Gradient Handling

If gradients become NaN or Inf, the trainer can retry the step:

- Set `max_non_finite_grad_retries` to a positive integer to enable retries.
- The trainer will reset gradients and re-run the forward/backward pass.
- If retries are exhausted, a `RuntimeError` is raised.

### State Management

Save and load the full (possibly distributed) training state (model, optimizer, scheduler, hooks):

```python
# Save checkpoint
state = trainer.state_dict()

# Resume training
trainer.load_state_dict(state)
```

The `CheckpointingHook` automates this including saving and loading from disk for you.

---

## Hooks

Hooks let you inject custom logic at key points in the training loop. All hooks inherit from `BaseHook` and can override these methods:

- `on_before_train(trainer)`: Called once before training starts.
- `on_after_train(trainer)`: Called once after training finishes.
- `on_before_step(trainer)`: Called before each training step.
- `on_after_step(trainer)`: Called after each training step.
- `on_log(trainer, records, dry_run)`: Called when the trainer logs metrics.
- `on_log_images(trainer, records, dry_run)`: Called when the trainer logs images.
- `on_state_dict(trainer, state_dict)`: Called when saving a checkpoint.
- `on_load_state_dict(trainer, state_dict)`: Called when loading a checkpoint.

### Built-in Hooks

#### `ProgressHook`

Prints training progress to the console.

```python
ProgressHook(
    interval=10,          # log every N steps
    with_records=True,    # include extra metrics in the output
    sync=False,           # synchronize metrics across ranks
    eta_warmup=10,        # steps to warm up ETA calculation
)
```

**Example output:**

```
Step   100/10000: step 0.2500s data 0.0100s eta 6:00:00 loss 0.5234 grad_norm 2.3456 lr_0 1.00e-03
```

#### `LoggingHook`

Aggregates metrics and calls `trainer.log(records)` at regular intervals. Use this to send metrics to experiment trackers.

```python
LoggingHook(
    interval=100,  # aggregate and log every N steps
    sync=True,     # synchronize metrics across ranks
)
```

Implement `trainer.log()` or add a hook that handles `on_log` (e.g. WandbHook) to handle the aggregated records.
```

#### `CheckpointingHook`

Saves checkpoints at regular intervals and handles automatic resuming.

```python
CheckpointingHook(
    interval=1000,              # save every N steps
    keep_previous=2,            # keep the last N checkpoints
    keep_interval=5000,         # keep checkpoints every N steps (in addition to the last N)
    path="checkpoints",         # directory to save checkpoints (relative to workspace)
    load="latest",              # load the latest checkpoint in the workspace on startup ("latest", a specific path, or None)
    exit_signals=[signal.SIGTERM, signal.SIGINT],  # save on these signals before exiting
    exit_code="128+signal",     # exit code to use after signal handling
    exit_wait=60.0,             # wait time before exiting (useful to get TIMEOUT instead of FAILED slurm job status)
)
```

Checkpoints are saved as `checkpoint_step_{step}.pt`, and the latest is symlinked as `checkpoint_latest.pt`.

#### `CudaMaxMemoryHook`

Tracks and logs the maximum GPU memory allocated during training.

```python
CudaMaxMemoryHook()
```

Adds `max_memory` (in GiB) to `trainer.step_info` for other hooks to use.

#### `EmaHook`

Maintains an exponential moving average (EMA) of model weights.

```python
EmaHook(
    decay=0.9999,               # EMA decay rate
)
```

Access the EMA model via `hook.ema_model`.

#### `WandbHook`

Logs metrics and images to Weights & Biases.

```python
WandbHook(
    project="my-project",
    name="experiment-1",
    config={"lr": 1e-3, "batch_size": 32},
    image_format = "png",
    # ... (other wandb.init arguments)
)
```

Call `trainer.log()` and `trainer.log_images()` to send data to wandb.

#### `ImageFileLoggerHook`

Saves images to disk (useful for debugging or visualization).

```python
ImageFileLoggerHook(
    image_format = "png",
)
```

Call `trainer.log_images({"image_name": pil_image})` to save images.

---

## Advanced Features

### Distributed Training

The trainer integrates with PyTorch's DDP and FSDP. Just wrap your model with `DistributedDataParallel` or `FullyShardedDataParallel` in `build_model()`.

### Unwrapping Models

The trainer provides a helper to unwrap compiled, distributed, or EMA-wrapped models:

```python
unwrapped = trainer.unwrap(trainer.model)
# or use the property:
unwrapped = trainer.unwrapped_model
```

This is useful for accessing the base model's methods or parameters.

### Custom Hooks

Create your own hooks by subclassing `BaseHook`:

```python
from mini.trainer import BaseHook

class CustomHook(BaseHook):
    def on_after_step(self, trainer):
        if trainer.step % 100 == 0:
            print(f"Custom hook triggered at step {trainer.step}")
```

Add it to your trainer:

```python
def build_hooks(self):
    return [CustomHook(), ProgressHook(interval=10)]
```

---

## Utilities

### `map_nested_tensor`

Apply a function to all tensors in a nested structure:

```python
from mini.trainer import map_nested_tensor

input = {"x": torch.randn(2, 3), "y": [torch.randn(4, 5)]}
output = map_nested_tensor(lambda t: t.to("cuda"), input)
```

Useful for moving data to devices or converting dtypes.

### `key_average`

Average numeric values across a list of nested dicts:

```python
from mini.trainer.utils import key_average

records = [
    {"loss": 0.5, "metrics": {"acc": 0.9}},
    {"loss": 0.6, "metrics": {"acc": 0.85}},
]
avg = key_average(records)
# => {"loss": 0.55, "metrics": {"acc": 0.875}}
```

Used internally by hooks to aggregate metrics.

---

## API Reference

### `BaseTrainer`

**Constructor Parameters:**

- `max_steps` (int): Total number of training steps.
- `grad_clip` (float | None): Maximum gradient norm for clipping (default: `None`).
- `max_non_finite_grad_retries` (int | None): Number of retries for non-finite gradients (default: `None`).
- `mixed_precision` (str | None): `"fp16"`, `"bf16"`, or `None` (default: `None`).
- `gradient_accumulation_steps` (int | None): Number of steps to accumulate gradients (default: 1).
- `workspace` (Path | str | None): Directory for saving checkpoints and logs (default: `None`).
- `device` (torch.device | str | int): Device for training (default: `"cuda"`).
- `no_sync_accumulate` (bool): Skip gradient synchronization during accumulation (default: `True`).
- `state_dict_options` (StateDictOptions | None): Options for distributed state dict handling (default: `None`).
- `logger` (Logger | None): Python logger instance (default: creates a new logger).

**Key Attributes:**

- `step` (int): Current training step (0 before training starts).
- `model` (nn.Module): The model being trained.
- `optimizer` (torch.optim.Optimizer): The optimizer.
- `lr_scheduler` (torch.optim.lr_scheduler.LRScheduler | None): The learning rate scheduler.
- `data_loader` (Iterable): The data loader.
- `hooks` (list[BaseHook]): List of active hooks.
- `step_info` (dict): Metrics from the current step (populated during `_run_step()`).

**Methods:**

- `train()`: Start training.
- `state_dict() -> dict`: Get the full training state (e.g. called by a checkpoint hook).
- `load_state_dict(state_dict)`: Restore training state (e.g. called by a checkpoint hook).
- `log(records, dry_run=False)`: Log metrics (delegated to hooks: `on_log`).
- `log_images(records, dry_run=False)`: Log images (delegated to hooks: `on_log_images`).

---

## Tips

- **Use hooks for side effects**: Logging, checkpointing, and validation are best handled via hooks.
- **Combine with `mini.config` and `mini.builder`**: Define your training setup in config files and use the builder to instantiate the trainer.

---

## Integration Example

Using `mini.trainer` with `mini.config` and `mini.builder`:

```python
# configs/train.py
config = {
    "type": "MyTrainer",
    "max_steps": 10_000,
    "grad_clip": 1.0,
    "mixed_precision": "bf16",
    "gradient_accumulation_steps": 4,
    "workspace": "./runs/experiment_1",
    "data": {
        "type": "torch.utils.data.DataLoader",
        "dataset": {"type": "ToyDataset"},
        "batch_size": 32,
    },
    "model": {
        "type": "MyModel",
        "in_dim": 784,
        "hidden_dim": 128,
        "out_dim": 10,
    },
    "optimizer": {
        "type": "torch.optim.AdamW",
        "lr": 3e-4,
    },
}
```

```python
# train.py
import logging

import torch
from torch import nn
from torch.utils.data import IterableDataset

from mini.builder import REGISTRY, build_from_cfg
from mini.config import load_config
from mini.trainer import BaseTrainer, CheckpointingHook, ProgressHook

logging.basicConfig(level=logging.INFO)


@REGISTRY.register()
class MyTrainer(BaseTrainer):
    def __init__(self, data, model, optimizer, **kwargs):
        super().__init__(**kwargs)
        self.data_cfg = data
        self.model_cfg = model
        self.optimizer_cfg = optimizer

    def build_data_loader(self):
        return build_from_cfg(self.data_cfg, recursive=True)

    def build_model(self):
        model = build_from_cfg(self.model_cfg, recursive=True)
        return model.to(self.device)

    def build_optimizer(self):
        return build_from_cfg(self.optimizer_cfg | {"params": self.model.parameters()})

    def forward(self, input):
        x, y = input
        x, y = x.to(self.device), y.to(self.device)
        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        acc = (logits.argmax(1) == y).float().mean().item()
        return loss, {"accuracy": acc}

    def build_hooks(self):
        return [
            ProgressHook(interval=1, with_records=True),
            CheckpointingHook(interval=1000, path=self.workspace / "checkpoints"),
        ]


@REGISTRY.register()
class ToyDataset(IterableDataset):
    def __iter__(self):
        while True:
            x = torch.randn(1, 28, 28)
            y = torch.randint(0, 10, (1,)).item()
            yield x.view(-1), y


@REGISTRY.register()
class MyModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


try:
    device = torch.device(0)
    torch.distributed.init_process_group(
        world_size=1, rank=0, store=torch.distributed.HashStore(), device_id=device
    )

    cfg = load_config("configs/train.py")
    trainer = build_from_cfg(cfg | {"device": device}, recursive=False)
    trainer.train()
finally:
    torch.distributed.destroy_process_group()
```

This approach keeps your training code declarative and easy to modify via config files.
