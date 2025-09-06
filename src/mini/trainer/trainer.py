import time
import warnings
from contextlib import closing, nullcontext
from logging import Logger
from numbers import Number
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    TypeAlias,
    Union,
)

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)

if TYPE_CHECKING:
    from .hooks import BaseHook

Records: TypeAlias = dict[str, Union[Number, "Records"]]


class BaseTrainer:
    def __init__(
        self,
        max_steps: int,
        grad_clip: float | None,
        max_non_finite_grad_retries: int | None,
        mixed_precision: str | None,
        gradient_accumulation_steps: int | None,
        workspace: Path | str | None,
        logger: Logger,
        device: torch.device | str | int,
        no_sync_accumulate: bool = True,  # can make sense to disable this for FSDP
        state_dict_options: StateDictOptions | None = None,
    ):
        self.step = 0  # refers to the last begun step. incremented *before* each step
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.max_non_finite_grad_retries = max_non_finite_grad_retries
        match mixed_precision:
            case "fp16":
                self.mixed_precision = torch.float16
            case "bf16":
                self.mixed_precision = torch.bfloat16
            case None:
                self.mixed_precision = None
            case _:
                raise ValueError(f"Unsupported mixed precision: {mixed_precision}")
        self.device = torch.device(device)
        self.gradient_accumulation_steps = gradient_accumulation_steps or 1
        self.workspace = Path(workspace) if workspace is not None else None
        self.logger = logger
        self.no_sync_accumulate = no_sync_accumulate
        self.state_dict_options = state_dict_options

    def _build(self):
        self.logger.debug("_build()")
        self.data_loader = self.build_data_loader()
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.lr_scheduler = self.build_lr_scheduler()
        self.grad_scaler = self.build_grad_scaler()
        self.hooks = self.build_hooks()

    def build_data_loader(self) -> Iterable:
        raise NotImplementedError

    def build_model(self) -> nn.Module:
        raise NotImplementedError

    def build_optimizer(self) -> torch.optim.Optimizer:
        raise NotImplementedError

    def build_lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        return None

    def build_hooks(self) -> list["BaseHook"]:
        return []

    def build_grad_scaler(self) -> torch.amp.GradScaler:
        return torch.amp.GradScaler(
            self.device.type, enabled=self.mixed_precision == torch.float16
        )

    def state_dict(self) -> dict[str, Any]:
        self.logger.debug("state_dict()")
        model_state_dict, optimizer_state_dict = get_state_dict(
            self.model, self.optimizer, options=self.state_dict_options
        )
        state_dict = {
            "model": model_state_dict,
            "training_state": {
                "step": self.step,
                "optimizer": optimizer_state_dict,
                "lr_scheduler": self.lr_scheduler.state_dict()
                if self.lr_scheduler
                else None,
                "grad_scaler": self.grad_scaler.state_dict(),
            },
        }
        for h in self.hooks:
            h.on_state_dict(self, state_dict)
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.logger.debug("load_state_dict()")
        training_state = state_dict["training_state"]

        self.step = training_state["step"]
        self.logger.info(f"=> Resuming from step {self.step} ...")

        if self.lr_scheduler is not None:
            # NOTE: order is important. load the optimizer AFTER lr_scheduler. https://github.com/pytorch/pytorch/issues/119168
            self.logger.info("=> Loading LR scheduler state ...")
            self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
        self.logger.info("=> Loading grad scaler state ...")
        self.grad_scaler.load_state_dict(training_state["grad_scaler"])

        self.logger.info("=> Loading model and optimizer state ...")
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=training_state["optimizer"],
            options=self.state_dict_options,
        )

        self.logger.info("=> Loading hook states ...")
        for h in self.hooks:
            h.on_load_state_dict(self, state_dict)

    def train(self):
        self._build()
        self._before_train()

        self.model.train()
        self.optimizer.zero_grad()  # just in case

        # attempt to explicitly close the iterator since it likely owns resources such as worker processes
        with maybe_closing(iter(self.data_loader)) as data_iter:
            while self.step < self.max_steps:
                self.step += 1
                self.step_info = {}
                self._before_step()

                step_time = time.perf_counter()
                self._run_step(data_iter)
                self.step_info["step_time"] = time.perf_counter() - step_time

                self._after_step()

        self._after_train()

    # the only difference is that we add the accumulate context and do the warning
    def _run_step(self, data_iter: Iterator):
        """
        Run a single optimizer step of training.
        Args:
            data_iter (Iterator): Data iterator.
        """

        def reset_step_info():
            self.step_info["loss"] = []
            self.step_info["records"] = []

        reset_step_info()
        self.step_info["data_time"] = []
        non_finite_grad_retry_count = 0
        i_acc = 0
        while i_acc < self.gradient_accumulation_steps:
            is_accumulating = i_acc < self.gradient_accumulation_steps - 1
            no_sync_accumulate = (
                self.model.no_sync()
                if self.no_sync_accumulate
                and is_accumulating
                and hasattr(self.model, "no_sync")
                else nullcontext()
            )  # for DDP and FSDP

            data_time = time.perf_counter()
            input = next(data_iter)
            self.step_info["data_time"].append(time.perf_counter() - data_time)

            with no_sync_accumulate:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.mixed_precision,
                    enabled=bool(self.mixed_precision),
                ):
                    self.logger.debug(f"{self.step}-{i_acc} forward()")
                    loss, records = self.forward(input)
                if loss is None:
                    if isinstance(
                        self.model,
                        (
                            torch.nn.parallel.DistributedDataParallel,
                            torch.distributed.fsdp.FullyShardedDataParallel,
                        ),
                    ):
                        # TODO: find a better way to handle this
                        # It seems that each DDP forward call is expected to be followed by a backward pass, as DDP maintains internal state after the forward pass that anticipates a backward step.
                        # While it might work with `broadcast_buffers=False` or if the backward pass is collectively skipped across all ranks,
                        # this behavior is not officially documented as safe and could result in undefined behavior.
                        # Since `Trainer.forward` may also return None before calling `DDP.forward`, this is just a warning rather than an error.
                        # I think the same thing applies to FSDP, but I haven't confirmed it.
                        warnings.warn(
                            "Loss is None; skipping backward step. Ensure self.model.forward was not called in self.forward to avoid undefined behavior in DDP and FSDP.",
                            LossNoneWarning,
                        )
                    continue  # skip the backward & optimizer step
                if not torch.isfinite(
                    loss
                ):  # TODO: check if device sync slows down training
                    self.logger.warning(
                        f"Loss is non-finite ({loss.item()}). records={records}"
                    )
                    # we will handle non-finite later at the optimizer step, the warning is just for debugging
                    # keep in mind that at least for DDP, we must still call backward() to avoid undefined behavior!

                self.step_info["loss"].append(loss.detach())
                self.step_info["records"].append(records)
                loss = loss / self.gradient_accumulation_steps
                self.logger.debug(f"{self.step}-{i_acc} backward()")
                self.grad_scaler.scale(loss).backward()
                i_acc += 1  # only increment after an actual backward pass

            if not is_accumulating:
                if not self.grad_scaler.is_enabled():
                    # only skip non-finite grads if the scaler is disabled (the scaler needs to process non-finite grads to adjust the scale)
                    if any(
                        (not torch.isfinite(p.grad).all())
                        for p in self.model.parameters()
                        if p.grad is not None
                    ):
                        if self.max_non_finite_grad_retries is None or (
                            non_finite_grad_retry_count
                            < self.max_non_finite_grad_retries
                        ):
                            non_finite_grad_retry_count += 1
                            self.logger.warning(
                                f"Gradient is non-finite. Retrying step {self.step} (retry {non_finite_grad_retry_count}"
                                + (
                                    f"/{self.max_non_finite_grad_retries})."
                                    if self.max_non_finite_grad_retries is not None
                                    else ")."
                                )
                            )
                            self.optimizer.zero_grad()
                            # TODO: check if we also need to "reset" (is that a thing?) the scaler here
                            reset_step_info()
                            i_acc = 0  # start accumulation again
                            continue
                        else:
                            raise RuntimeError(
                                "Gradient is non-finite. Exceeded maximum retries for non-finite gradients."
                            )
                self.grad_scaler.unscale_(self.optimizer)
                if self.grad_clip is not None:
                    self.step_info["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                self.logger.debug(f"{self.step}-{i_acc - 1} step()")
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

    def forward(self, input: Any) -> tuple[torch.Tensor | None, Records]:
        """
        Perform the forward pass of the model.

        Args:
            input (Any): Input data from the dataloader.

        Returns:
            1. The computed loss tensor. If None, the backward and optimizer steps will be skipped.
            Note: When using DDP or FSDP, ensure that None is not returned after a forward pass
            to avoid potential undefined behavior.
            2. A nested dictionary with str keys and Number values to be averaged and logged.
        """
        raise NotImplementedError

    @classmethod
    def unwrap(self, module: nn.Module) -> nn.Module:
        match module:
            case torch._dynamo.eval_frame.OptimizedModule():
                return self.unwrap(module._orig_mod)
            case (
                torch.nn.parallel.DistributedDataParallel()
                | torch.nn.parallel.DataParallel()
                | torch.optim.swa_utils.AveragedModel()
            ):
                return self.unwrap(module.module)
        return module

    @property
    def unwrapped_model(self):
        return self.unwrap(self.model)

    def log(self, records: dict[str, Any], dry_run: bool = False):
        """
        Log the records to e.g., an experiment tracker or the console.
        Args:
            records (dict): Nested dict of numeric records with string keys to log.
        """
        self.logger.debug("log()")
        for h in self.hooks:
            h.on_log(self, records, dry_run=dry_run)

    def log_images(self, records: dict[str, Any], dry_run: bool = False):
        """
        Log images to e.g., an experiment tracker.
        Args:
            records (dict): Nested dict of PIL images with string keys to log.
        """
        self.logger.debug("log_images()")
        for h in self.hooks:
            h.on_log_images(self, records, dry_run=dry_run)

    def _before_train(self):
        self.logger.debug("_before_train()")
        for h in self.hooks:
            h.on_before_train(self)

    def _after_train(self):
        self.logger.debug("_after_train()")
        for h in self.hooks:
            h.on_after_train(self)

    def _before_step(self):
        self.logger.debug("_before_step()")
        for h in self.hooks:
            h.on_before_step(self)

    def _after_step(self):
        self.logger.debug("_after_step()")
        for h in self.hooks:
            h.on_after_step(self)


def maybe_closing(obj):
    """Return a context manager that closes `obj` if it has a .close() method, otherwise does nothing."""
    return closing(obj) if callable(getattr(obj, "close", None)) else nullcontext(obj)


def map_nested_tensor(f: Callable[[torch.Tensor], Any], obj: Any):
    if isinstance(obj, torch.Tensor):
        return f(obj)
    elif isinstance(obj, (list, tuple, set)):
        return type(obj)(map_nested_tensor(f, o) for o in obj)
    elif isinstance(obj, dict):
        return type(obj)((k, map_nested_tensor(f, v)) for k, v in obj.items())
    else:
        return obj


class LossNoneWarning(UserWarning):
    pass
