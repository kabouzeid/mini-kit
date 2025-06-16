import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist

from .trainer import BaseTrainer
from .utils import key_average


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
        with_records: bool,
        sync: bool,
    ):
        self.interval = interval
        self.with_records = with_records
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
        records: dict,
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
        super().__init__(interval=interval, with_records=with_records, sync=sync)
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

    def on_after_step(self, trainer: BaseTrainer):
        super().on_after_step(trainer)
        self.eta_tracker.step()

    def process_stats(
        self,
        trainer: BaseTrainer,
        loss: float,
        grad_norm: float | None,
        step_time: float,
        data_time: float,
        max_memory: float | None,
        records: dict,
    ):
        lrs = [
            (i, param_group["lr"])
            for i, param_group in enumerate(trainer.optimizer.param_groups)
        ]
        trainer.logger.info(
            f"Step {trainer.step:>{len(str(trainer.max_steps))}}/{trainer.max_steps}: loss={loss:.4f}, {', '.join(f'lr_{i}={lr:.2e}' for i, lr in lrs)}"
            + (f", grad_norm={grad_norm:.4f}" if grad_norm is not None else "")
            + f", step={step_time:.4f}s, data={data_time:.4f}s, eta={self.eta_tracker.get_eta()}"
            + (f", mem={max_memory:.1f}GiB" if max_memory is not None else "")
            + (f", records={records}" if self.with_records else "")  # TODO: format
        )


class LoggerHook(_StatsHook):
    """
    Log aggregated training statistics and records to a `log(records: dict[str, Any])` method defined in the trainer.
    """

    def __init__(
        self,
        interval: int = 10,
        with_records: bool = True,
        sync: bool = True,
    ):
        super().__init__(interval, with_records, sync)

    def process_stats(
        self,
        trainer: BaseTrainer,
        loss: float,
        grad_norm: float | None,
        step_time: float,
        data_time: float,
        max_memory: float | None,
        records: dict,
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


class CheckpointHook(BaseHook):
    """
    Save checkpoints at regular intervals. The latest checkpoint is always saved as a symlink.
    """

    def __init__(
        self,
        interval: int,
        keep: int = 1,
        path: Path | str = "checkpoint",
        load: Path | str | Literal["latest"] | None = "latest",
    ):
        assert interval > 0
        assert keep > 0
        self.interval = interval
        self.keep = keep
        self.path = Path(path)
        self.load_path = Path(load) if load is not None else None
        self.saved_checkpoints = []

    def on_before_train(self, trainer: BaseTrainer):
        load_path = self.load_path
        if load_path is not None:
            # handles 'latest' and 'step_xxx' checkpoints
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

            trainer.logger.info(f"=> Loading checkpoint from {load_path}")
            state_dict = {
                file.with_suffix("").name: torch.load(
                    file, map_location=trainer.device, weights_only=True
                )
                for file in load_path.iterdir()
                if file.is_file() and file.suffix == ".pt"
            }
            trainer.logger.debug(f"Checkpoint contains: {', '.join(state_dict.keys())}")
            trainer.load_state_dict(state_dict)

    def on_after_step(self, trainer: BaseTrainer):
        if trainer.step % self.interval == 0 or trainer.step == trainer.max_steps:
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
                save_path = self.path / f"step_{trainer.step}"
                if not save_path.is_absolute():
                    assert trainer.workspace is not None
                    save_path = trainer.workspace / save_path
                trainer.logger.info(f"=> Saving checkpoint to {save_path}")
                save_path.mkdir(parents=True, exist_ok=True)

                # save
                for name, sub_state_dict in state_dict.items():
                    torch.save(sub_state_dict, save_path / f"{name}.pt")

                # symlink latest
                latest_symlink = save_path.parent / "latest"
                if latest_symlink.is_symlink():
                    latest_symlink.unlink()
                elif latest_symlink.exists():
                    trainer.logger.warning(
                        f"{latest_symlink} already exists and is not a symlink."
                    )
                latest_symlink.symlink_to(save_path.name, target_is_directory=True)

                # clean up old checkpoints
                self.saved_checkpoints.append(save_path)
                while len(self.saved_checkpoints) > self.keep:
                    to_remove = self.saved_checkpoints.pop(0)
                    try:
                        shutil.rmtree(to_remove)
                    except OSError as e:
                        trainer.logger.warning(f"Could not remove {to_remove}: {e}")


class CudaMaxMemoryHook(BaseHook):
    def on_before_step(self, trainer: BaseTrainer):
        torch.cuda.reset_peak_memory_stats(trainer.device)

    def on_after_step(self, trainer: BaseTrainer):
        trainer.step_info["max_memory"] = torch.cuda.max_memory_allocated(
            trainer.device
        ) / (1024**3)  # GiB
