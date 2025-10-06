import os
import shutil
import signal
import sys
import tempfile
import time
import warnings
from datetime import timedelta
from numbers import Number
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

import torch
import torch.distributed as dist
from PIL import Image
from PIL.Image import Image as PILImage
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn

try:
    import wandb
except ImportError:
    # only needed for WandbHook
    pass

from .trainer import BaseTrainer, Records
from .utils import flatten_nested_dict, key_average


class BaseHook:
    def on_before_train(self, trainer: BaseTrainer):
        pass

    def on_before_step(self, trainer: BaseTrainer):
        pass

    def on_after_step(self, trainer: BaseTrainer):
        pass

    def on_after_train(self, trainer: BaseTrainer):
        pass

    def on_log(self, trainer: BaseTrainer, records: dict, dry_run: bool = False):
        pass

    def on_log_images(self, trainer: BaseTrainer, records: dict, dry_run: bool = False):
        pass

    def on_state_dict(self, trainer: BaseTrainer, state_dict: dict):
        pass

    def on_load_state_dict(self, trainer: BaseTrainer, state_dict: dict):
        pass


class _StatsHook(BaseHook):
    def __init__(
        self,
        interval: int,
        sync: bool,
    ):
        self.interval = interval
        self.sync = sync
        self.reset()

    def reset(self):
        self.losses = []
        self.records_ls = []
        self.grad_norms = []
        self.data_times = []
        self.step_times = []
        self.max_memories = []

    def on_after_step(self, trainer: BaseTrainer):
        # collect and aggregate over accumulation steps
        self.losses.append(torch.stack(trainer.step_info["loss"]).mean())
        if trainer.grad_clip is not None:
            self.grad_norms.append(trainer.step_info["grad_norm"])
        self.records_ls.append(key_average(trainer.step_info["records"]))
        self.data_times.append(sum(trainer.step_info["data_time"]))  # total
        self.step_times.append(trainer.step_info["step_time"])
        if "max_memory" in trainer.step_info:
            self.max_memories.append(trainer.step_info["max_memory"])

        if trainer.step % self.interval == 0 or trainer.step == trainer.max_steps:
            # aggregate over steps
            loss = torch.stack(self.losses).mean()
            grad_norm = torch.stack(self.grad_norms).mean() if self.grad_norms else None
            records = key_average(self.records_ls)
            data_time = sum(self.data_times) / len(self.data_times)
            step_time = sum(self.step_times) / len(self.step_times)
            max_memory = max(self.max_memories) if self.max_memories else None

            if self.sync:
                # aggregate accross all ranks
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                if grad_norm is not None:
                    dist.all_reduce(grad_norm, op=dist.ReduceOp.AVG)

                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(
                    gathered,
                    {
                        "records": records,
                        "data_time": data_time,
                        "step_time": step_time,
                        "max_memory": max_memory,
                    },
                )
                records = key_average([stat["records"] for stat in gathered])
                data_time = sum(stat["data_time"] for stat in gathered) / len(gathered)
                step_time = sum(stat["step_time"] for stat in gathered) / len(gathered)
                if "max_memory" in trainer.step_info:
                    max_memory = max(stat["max_memory"] for stat in gathered)

            self.process_stats(
                trainer,
                loss.item(),
                grad_norm.item() if grad_norm is not None else None,
                step_time,
                data_time,
                max_memory,
                records,
            )
            self.reset()

    def process_stats(
        self,
        trainer: BaseTrainer,
        loss: float,
        grad_norm: float | None,
        step_time: float,
        data_time: float,
        max_memory: float | None,
        records: Records,
    ):
        raise NotImplementedError("Subclasses must implement this method.")


class ETATracker:
    def __init__(self, total_steps: int, warmup_steps: int):
        """
        Track the estimated time of arrival (ETA) for the training process. Call step() after each training step.
        Args:
            total_steps (int): Total number of times step() will be called.
            warmup_steps (int): Number of warmup steps before starting to track ETA. At least the first step must be warmup.
        """
        assert total_steps > 0, "Total steps must be greater than 0"
        self.total_steps = total_steps
        assert warmup_steps > 0, "Warmup steps must be greater than 0"
        self.warmup_steps = warmup_steps
        self.steps_done = 0
        self.timing_start = None
        self.timed_steps = 0

    def step(self):
        self.steps_done += 1
        if self.steps_done == self.warmup_steps:
            self.timing_start = time.perf_counter()
        if self.steps_done > self.warmup_steps:
            self.timed_steps += 1

    def get_eta(self):
        if self.timed_steps == 0:
            return None

        elapsed = time.perf_counter() - self.timing_start
        avg_step_time = elapsed / self.timed_steps
        steps_remaining = self.total_steps - self.steps_done
        eta_seconds = avg_step_time * steps_remaining
        return timedelta(seconds=int(eta_seconds))


class ProgressHook(_StatsHook):
    def __init__(
        self,
        interval: int = 1,
        with_records: bool = False,
        sync: bool = False,
        eta_warmup: int = 10,
    ):
        super().__init__(interval=interval, sync=sync)
        self.with_records = with_records
        self.eta_warmup = eta_warmup

    def on_before_train(self, trainer: BaseTrainer):
        super().on_before_train(trainer)
        trainer.logger.info("=> Starting training ...")
        self.eta_tracker = ETATracker(
            total_steps=trainer.max_steps - trainer.step,
            warmup_steps=self.eta_warmup,
        )

    def on_after_train(self, trainer: BaseTrainer):
        super().on_after_train(trainer)
        trainer.logger.info("=> Finished training")

    def on_before_step(self, trainer: BaseTrainer):
        super().on_before_step(trainer)
        self.lrs = [
            (i, param_group["lr"])
            for i, param_group in enumerate(trainer.optimizer.param_groups)
        ]  # record the LR before the scheduler steps

    def on_after_step(self, trainer: BaseTrainer):
        self.eta_tracker.step()  # should be called before process_stats
        super().on_after_step(trainer)

    def process_stats(
        self,
        trainer: BaseTrainer,
        loss: float,
        grad_norm: float | None,
        step_time: float,
        data_time: float,
        max_memory: float | None,
        records: Records,
    ):
        eta = self.eta_tracker.get_eta()
        trainer.logger.info(
            f"Step {trainer.step:>{len(str(trainer.max_steps))}}/{trainer.max_steps}:"
            + f" step {step_time:.4f}s data {data_time:.4f}s"
            + (f" eta {eta}" if eta is not None else "")
            + (f" mem {max_memory:#.3g}GiB" if max_memory is not None else "")
            + f" loss {loss:.4f}"
            + (f" grad_norm {grad_norm:.4f}" if grad_norm is not None else "")
            + (" " + " ".join(f"lr_{i} {lr:.2e}" for i, lr in self.lrs))
            + (
                (
                    " | "
                    + " ".join(
                        f"{'/'.join(k)} {f'{v:#.4g}' if isinstance(v, Number) else v}"
                        for k, v in flatten_nested_dict(records).items()
                    )
                )
                if self.with_records
                else ""
            )
        )


class LoggingHook(_StatsHook):
    """
    Log aggregated training statistics and records to a `log(records: dict[str, Any])` method defined in the trainer.
    """

    def __init__(
        self,
        interval: int = 10,
        sync: bool = True,
    ):
        super().__init__(interval, sync)

    def process_stats(
        self,
        trainer: BaseTrainer,
        loss: float,
        grad_norm: float | None,
        step_time: float,
        data_time: float,
        max_memory: float | None,
        records: Records,
    ):
        lrs = [
            (i, param_group["lr"])
            for i, param_group in enumerate(trainer.optimizer.param_groups)
        ]
        trainer.log(
            {
                "train": records
                | ({"grad_norm": grad_norm} if grad_norm is not None else {})
                | ({"max_memory": max_memory} if max_memory is not None else {})
                | {
                    "loss": loss,
                    "data_time": data_time,
                    "step_time": step_time,
                    "lr": {f"group_{i}": lr for i, lr in lrs},
                }
            }
        )


class CheckpointingHook(BaseHook):
    """
    Save checkpoints at regular intervals. The latest checkpoint is always saved as a symlink.
    """

    def __init__(
        self,
        interval: int,
        keep_previous: int = 0,  # keep N previous checkpoints
        keep_interval: int = 0,  # keep checkpoints of every N-th step
        path: Path | str = "checkpoint",
        load: Path | str | Literal["latest"] | None = "latest",
        exit_signals: list[signal.Signals] | signal.Signals = None,
        exit_code: int | Literal["128+signal"] = "128+signal",
        exit_wait: timedelta | float = 0.0,
    ):
        assert interval > 0
        assert keep_previous >= 0
        self.interval = interval
        self.keep_previous = keep_previous
        self.keep_interval = keep_interval
        self.path = Path(path)
        self.load_path = Path(load) if load is not None else None

        self.local_exit_signal: signal.Signals = -1  # not a valid value
        exit_signals = exit_signals if exit_signals is not None else []
        if not isinstance(exit_signals, Iterable):
            exit_signals = [exit_signals]
        for sig in exit_signals:
            signal.signal(sig, lambda *args: setattr(self, "local_exit_signal", sig))
        self.has_exit_signal_handlers = len(exit_signals) > 0
        self.exit_code = exit_code
        self.exit_wait = (
            exit_wait.total_seconds() if isinstance(exit_wait, timedelta) else exit_wait
        )

    def on_before_train(self, trainer: BaseTrainer):
        if self.has_exit_signal_handlers:
            # micro optimization: allocate signal tensor only once
            self.dist_exit_signal = torch.tensor(
                self.local_exit_signal, dtype=torch.int32, device=trainer.device
            )
        load_path = self.load_path
        if load_path is not None:
            # handles 'latest' and regular checkpoints
            if len(load_path.parts) == 1 and not load_path.is_absolute():
                load_path = self.path / load_path
                if not load_path.is_absolute():
                    assert trainer.workspace is not None
                    load_path = trainer.workspace / load_path
            if not load_path.is_dir():
                # nonexistent path is only ok if we're loading the 'latest' checkpoint
                assert str(self.load_path) == "latest", (
                    f"Checkpoint path {load_path} does not exist"
                )
                return

            trainer.logger.info(f"=> Loading checkpoint from {load_path} ...")
            state_dict = {
                file.with_suffix("").name: torch.load(
                    file, map_location=trainer.device, weights_only=True
                )
                for file in load_path.iterdir()
                if file.is_file() and file.suffix == ".pt"
            }
            trainer.logger.debug(f"Checkpoint contains: {', '.join(state_dict.keys())}")
            trainer.load_state_dict(state_dict)

    def on_before_step(self, trainer: BaseTrainer):
        if self.has_exit_signal_handlers:
            self.dist_exit_signal.fill_(self.local_exit_signal)
            # micro optimization: reduce async during step and read after step
            self.dist_exit_signal_work = dist.all_reduce(
                self.dist_exit_signal, op=dist.ReduceOp.MAX, async_op=True
            )

    def on_after_step(self, trainer: BaseTrainer):
        save_and_exit = False
        if self.has_exit_signal_handlers:
            self.dist_exit_signal_work.wait()
            exit_signal = self.dist_exit_signal.item()
            save_and_exit = exit_signal != -1

        # NOTE: Check if last step here (not in on_after_train) to avoid saving twice
        if (
            trainer.step % self.interval == 0
            or trainer.step == trainer.max_steps
            or save_and_exit
        ):
            if save_and_exit:
                trainer.logger.info(
                    f"=> Caught signal {exit_signal}. Saving checkpoint before exit ..."
                )
            self._save_checkpoint(
                trainer,
                keep=self.keep_interval > 0 and trainer.step % self.keep_interval == 0,
            )
            if save_and_exit:
                dist.barrier()
                if self.exit_wait > 0:
                    trainer.logger.info(
                        f"=> Waiting {self.exit_wait:.0f} seconds before exit ..."
                    )
                    time.sleep(self.exit_wait)  # try wait for the Slurm job timeout
                exit_code = (
                    128 + exit_signal
                    if self.exit_code == "128+signal"
                    else sys.exit(self.exit_code)
                )
                trainer.logger.info(f"=> Exiting (code: {exit_code})")
                sys.exit(exit_code)

    def _save_checkpoint(self, trainer: BaseTrainer, keep: bool):
        """Save a model checkpoint.

        Raises only if writing the current checkpoint fails. Issues encountered
        while retaining or pruning older checkpoints are logged but not raised.
        """

        dist.barrier()

        state_dict = trainer.state_dict()

        # TODO: all rank gathered states
        # gathered_random_states = [None] * dist.get_world_size()
        # dist.gather_object(
        #     get_random_state(),
        #     gathered_random_states if dist.get_rank() == 0 else None,
        #     dst=0,
        # )

        if dist.get_rank() == 0:
            # make dir
            save_path = self.path / str(trainer.step)
            if not save_path.is_absolute():
                assert trainer.workspace is not None
                save_path = trainer.workspace / save_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.logger.info(f"=> Saving checkpoint to {save_path} ...")

            # save
            tmp_save_path = self._get_tmp_save_dir(save_path)
            for name, sub_state_dict in state_dict.items():
                torch.save(sub_state_dict, tmp_save_path / f"{name}.pt")
            tmp_save_path.rename(save_path)

            # symlink latest
            latest_symlink = save_path.parent / "latest"
            if latest_symlink.is_symlink():
                latest_symlink.unlink()
            if latest_symlink.exists():
                trainer.logger.error(
                    f"{latest_symlink} already exists and is not a symlink. Will not create 'latest' symlink."
                )
            else:
                latest_symlink.symlink_to(save_path.name, target_is_directory=True)

            if keep:
                keep_path = save_path.with_name(save_path.name + "_keep")
                trainer.logger.info(
                    f"=> Marking checkpoint for keeping {keep_path} ..."
                )
                # retain checkpoint via symlink
                try:
                    save_path.rename(keep_path)
                    save_path.symlink_to(keep_path.name, target_is_directory=True)
                except Exception:
                    trainer.logger.exception(
                        f"Could not rename/symlink checkpoint for keeping {keep_path} ..."
                    )
                # # retain checkpoint via hard-linked copy (saves space, survives pruning of original)
                # try:
                #     shutil.copytree(save_path, keep_path, copy_function=os.link)
                # except Exception:
                #     trainer.logger.exception(
                #         f"Could not copy checkpoint for keeping {keep_path} ..."
                #     )

            # prune
            prev_ckpts = sorted(
                [
                    p
                    for p in save_path.parent.iterdir()
                    if p.is_dir()
                    and self._is_int(p.name)
                    and int(p.name) < trainer.step
                ],
                key=lambda p: int(p.name),
            )
            for p in (
                prev_ckpts[: -self.keep_previous]
                if self.keep_previous > 0
                else prev_ckpts
            ):
                trainer.logger.info(f"=> Pruning checkpoint {p} ...")
                try:
                    if p.is_symlink():
                        p.unlink()
                    else:
                        shutil.rmtree(p)
                except Exception:
                    trainer.logger.exception(f"Could not remove {p}")

    @staticmethod
    def _get_tmp_save_dir(path: Path):
        mask = os.umask(0)  # only way to get the umask is to set it
        os.umask(mask)
        tmp_save_path = Path(
            tempfile.mkdtemp(prefix=path.name + ".tmp.", dir=path.parent)
        )
        os.chmod(tmp_save_path, 0o777 & ~mask)  # set default mkdir permissions
        return tmp_save_path

    @staticmethod
    def _is_int(s: str):
        try:
            int(s)  # let's make absolutely sure that constructing and int will work
            return str.isdecimal(s)  # this filters out stuff like '+3' and '-3'
        except ValueError:
            return False


class CudaMaxMemoryHook(BaseHook):
    def on_before_step(self, trainer: BaseTrainer):
        torch.cuda.reset_peak_memory_stats(trainer.device)

    def on_after_step(self, trainer: BaseTrainer):
        trainer.step_info["max_memory"] = torch.cuda.max_memory_allocated(
            trainer.device
        ) / (1024**3)  # GiB


class EmaHook(BaseHook):
    def __init__(self, decay: float):
        self.decay = decay

    def on_before_train(self, trainer: BaseTrainer):
        trainer.logger.info("=> Creating EMA model ...")
        # Note that AveragedModel does not seem to support FSDP. It will crash here.
        self.ema_model = AveragedModel(trainer.model, avg_fn=get_ema_avg_fn(self.decay))

    def on_after_step(self, trainer: BaseTrainer):
        self.ema_model.update_parameters(trainer.model)

    def on_load_state_dict(self, trainer: BaseTrainer, state_dict: dict):
        trainer.logger.info("=> Loading EMA model state ...")
        set_model_state_dict(self.ema_model, state_dict["ema_model"])

    def on_state_dict(self, trainer: BaseTrainer, state_dict: dict):
        # Note: sadly, we need to keep the AveragedModel wrapper, to save its n_averaged buffer
        state_dict["ema_model"] = get_model_state_dict(self.ema_model)


class WandbHook(BaseHook):
    def __init__(
        self,
        project: str,
        config: dict[str, Any] | str | None = None,
        tags: Sequence[str] | None = None,
        image_format: str | None | Callable[[str], str | None] = "png",
        **wandb_kwargs,
    ):
        self.project = project
        self.config = config
        self.tags = tags
        if callable(image_format):
            self.image_format = image_format
        else:
            self.image_format = lambda _: image_format
        self.wandb_kwargs = wandb_kwargs

    def on_before_train(self, trainer: BaseTrainer):
        if dist.get_rank() == 0:
            wandb_run_id = self._load_wandb_run_id(trainer)

            tags = os.getenv("WANDB_TAGS", "")
            tags = list(self.tags) + (tags.split(",") if tags else [])  # concat
            tags = list(dict.fromkeys(tags))  # deduplicate while preserving order

            # it seems that we should use resume_from={run_id}?_{step} in wandb.init instead, but it's not well documented
            self.wandb = wandb.init(
                project=os.getenv("WANDB_PROJECT", self.project),
                dir=os.getenv("WANDB_DIR", trainer.workspace),
                id=os.getenv("WANDB_RUN_ID", wandb_run_id),
                resume=os.getenv("WANDB_RESUME", "must" if wandb_run_id else None),
                config=self.config,
                tags=tags,
                **self.wandb_kwargs,
            )
            if not self.wandb.disabled:
                self._save_wandb_run_id(trainer, self.wandb.id)

    def on_after_train(self, trainer: BaseTrainer):
        if dist.get_rank() == 0:
            self.wandb.finish()

    def on_log(self, trainer: BaseTrainer, records: dict, dry_run: bool = False):
        if dist.get_rank() == 0:
            data = {"/".join(k): v for k, v in flatten_nested_dict(records).items()}
            if not dry_run:
                self.wandb.log(data, step=trainer.step)
            else:
                trainer.logger.debug(f"Dry run log. Would log: {data}")

    def on_log_images(self, trainer: BaseTrainer, records: dict, dry_run: bool = False):
        if dist.get_rank() == 0:
            wandb_data = {}
            for k, img in flatten_nested_dict({"vis": records}).items():
                file_type = self.image_format(k[-1])
                wandb_data.setdefault("/".join(k[:-1]), []).append(
                    wandb.Image(
                        self._ensure_jpeg_compatible(img)
                        if file_type in ["jpg", "jpeg"]
                        else img,
                        caption=k[-1],
                        file_type=file_type,
                    )
                )

            if not dry_run:
                self.wandb.log(wandb_data, step=trainer.step)
            else:
                trainer.logger.debug(f"Dry run log. Would log: {wandb_data}")

    @staticmethod
    def _ensure_jpeg_compatible(img: PILImage, bg_color: tuple = (255, 255, 255)):
        if img.mode in ("RGB", "L"):
            return img
        elif img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, bg_color)
            background.paste(img, mask=img.getchannel("A"))
            return background
        else:
            warnings.warn(
                f"Trying to convert {img.mode} to RGB in a best-effort manner."
            )
            return img.convert("RGB")

    @staticmethod
    def _wandb_run_id_file_name(trainer: BaseTrainer):
        return trainer.workspace / "wandb_run_id"

    @classmethod
    def _save_wandb_run_id(cls, trainer: BaseTrainer, run_id: str):
        cls._wandb_run_id_file_name(trainer).write_text(run_id)

    @classmethod
    def _load_wandb_run_id(cls, trainer: BaseTrainer):
        f = cls._wandb_run_id_file_name(trainer)
        if f.exists():
            return f.read_text()
        return None


class ImageFileLoggerHook(BaseHook):
    def __init__(
        self,
        image_format: str | Callable[[str], str] = "png",
    ):
        if callable(image_format):
            self.image_format = image_format
        else:
            self.image_format = lambda _: image_format

    def on_log_images(self, trainer: BaseTrainer, records: dict, dry_run: bool = False):
        if dist.get_rank() == 0:
            for k, img in flatten_nested_dict(records).items():
                p = trainer.workspace / "visualizations" / str(trainer.step) / Path(*k)
                p = Path(str(p) + "." + self.image_format(k[-1]))
                if not dry_run:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    img.save(p)
                else:
                    trainer.logger.debug(f"Dry run log. Would save {img} to: {p}")
