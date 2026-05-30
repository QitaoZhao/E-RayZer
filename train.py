from __future__ import annotations

import argparse
import importlib
import inspect
import math
import os
import random
import time
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import erayzer_core  # noqa: F401  # registers model.* and utils.* aliases
from erayzer_core.losses import PerceptualLoss
from training.datasets import build_dataset as build_config_dataset


def _to_edict(value: Any) -> Any:
    if isinstance(value, dict):
        return edict({k: _to_edict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_edict(v) for v in value]
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def _parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[parts[-1]] = value


def load_config(path: str, overrides: Iterable[Tuple[str, str]]) -> edict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    for key, value in overrides:
        _set_nested(data, key, _parse_value(value))
    return _to_edict(data)


def setup_distributed() -> Tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return device, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_model(config: edict, device: torch.device, rank: int) -> torch.nn.Module:
    module_name, class_name = config.model.class_name.rsplit(".", 1)
    model_cls = importlib.import_module(module_name).__dict__[class_name]
    model = model_cls(config).to(device)

    resume_ckpt = config.training.get("resume_ckpt")
    if resume_ckpt and str(resume_ckpt).lower() != "none":
        checkpoint = torch.load(resume_ckpt, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if is_rank0(rank):
            print(f"Loaded checkpoint: {resume_ckpt}")
        if missing and is_rank0(rank):
            print(f"Missing keys: {len(missing)}")
        if unexpected and is_rank0(rank):
            print(f"Unexpected keys: {len(unexpected)}")
    return model


def make_dataloader(
    dataset: torch.utils.data.Dataset,
    config: edict,
    rank: int,
    world_size: int,
) -> DataLoader:
    training = config.training
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    num_workers = int(training.get("num_workers", 4))
    dataloader_kwargs = {}
    if num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
        dataloader_kwargs["prefetch_factor"] = int(training.get("prefetch_factor", 8))
    dataloader = DataLoader(
        dataset,
        batch_size=int(training.batch_size_per_gpu),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=worker_init_fn,
        **dataloader_kwargs,
    )
    return dataloader


def build_dataloader(config: edict, rank: int, world_size: int) -> Tuple[torch.utils.data.Dataset, DataLoader]:
    dataset = build_config_dataset(config, split="train")
    dataloader = make_dataloader(dataset, config, rank, world_size)
    return dataset, dataloader


def configure_optimizer(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
) -> torch.optim.Optimizer:
    param_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}
    decay_params = [param for param in param_dict.values() if param.dim() >= 2]
    nodecay_params = [param for param in param_dict.values() if param.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    try:
        first_param = next(model.parameters())
    except StopIteration:
        first_param = None
    extra_args = {"fused": True} if fused_available and first_param is not None and first_param.is_cuda else {}
    return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)


def get_job_overview(
    *,
    num_gpus: int,
    num_epochs: int,
    num_train_samples: int,
    batch_size_per_gpu: int,
    grad_accum_steps: int,
    max_fwdbwd_passes: int,
) -> edict:
    batch_size_per_fwdbwd_pass = batch_size_per_gpu * num_gpus
    num_fwdbwd_passes_per_epoch = max(1, int(num_train_samples / batch_size_per_fwdbwd_pass))
    batch_size_per_param_update = batch_size_per_fwdbwd_pass * grad_accum_steps
    num_param_updates_per_epoch = max(1, int(num_fwdbwd_passes_per_epoch / grad_accum_steps))
    num_epochs = min(num_epochs, int(max_fwdbwd_passes / num_fwdbwd_passes_per_epoch) + 1)
    return edict(
        batch_size_per_fwdbwd_pass=batch_size_per_fwdbwd_pass,
        batch_size_per_param_update=batch_size_per_param_update,
        num_fwdbwd_passes_per_epoch=num_fwdbwd_passes_per_epoch,
        num_param_updates_per_epoch=num_param_updates_per_epoch,
        num_fwdbwd_passes=num_fwdbwd_passes_per_epoch * num_epochs,
        num_param_updates=num_param_updates_per_epoch * num_epochs,
        num_epochs=num_epochs,
    )


def configure_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_train_steps: int,
    warmup_steps: int,
    scheduler_type: str = "cosine",
) -> torch.optim.lr_scheduler.LambdaLR:
    total_train_steps = max(1, int(total_train_steps))
    warmup_steps = max(0, int(warmup_steps))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if scheduler_type == "constant":
            return 1.0
        progress = float(current_step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_type == "linear":
            return max(0.0, 1.0 - progress)
        if scheduler_type == "cosine":
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        raise ValueError(f"Unsupported training.scheduler_type: {scheduler_type}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_loss_modules(config: edict, device: torch.device) -> edict:
    modules = edict()
    if float(config.training.get("perceptual_loss_weight", 0.0)) > 0.0:
        modules.perceptual = PerceptualLoss(
            weight_path=config.training.get("perceptual_loss_weight_path")
        ).to(device)
    return modules


def compute_loss(result: edict, batch: Dict[str, torch.Tensor], config: edict, loss_modules: edict) -> edict:
    pred = result.render.clamp(0.0, 1.0)
    target = result.get("target_image", batch["image"])
    if pred.shape[-2:] != target.shape[-2:]:
        b, v = target.shape[:2]
        target = F.interpolate(
            target.reshape(b * v, *target.shape[2:]),
            size=pred.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(b, v, target.shape[2], pred.shape[-2], pred.shape[-1])

    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    perceptual = pred.new_tensor(0.0)
    if hasattr(loss_modules, "perceptual"):
        b, v, c, h, w = pred.shape
        perceptual = loss_modules.perceptual(
            pred.reshape(b * v, c, h, w),
            target.reshape(b * v, c, h, w),
        )
    loss = float(config.training.get("l2_loss_weight", 1.0)) * mse
    loss = loss + float(config.training.get("l1_loss_weight", 0.0)) * l1
    loss = loss + float(config.training.get("perceptual_loss_weight", 0.0)) * perceptual
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-8))
    return edict(
        loss=loss,
        mse=mse.detach(),
        l1=l1.detach(),
        perceptual=perceptual.detach(),
        psnr=psnr.detach(),
    )


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
    fwdbwd_pass_step: int,
    param_update_step: int,
    config: edict,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    module = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "model": module.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "step": fwdbwd_pass_step,
            "fwdbwd_pass_step": fwdbwd_pass_step,
            "param_update_step": param_update_step,
            "config": _to_plain(config),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--set", "-s", action="append", nargs=2, default=[])
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    device, rank, local_rank, world_size = setup_distributed()
    random.seed(777 + rank)
    np.random.seed(777 + rank)
    torch.manual_seed(777 + rank)
    torch.backends.cuda.matmul.allow_tf32 = bool(config.training.get("use_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(config.training.get("use_tf32", True))

    dataset, dataloader = build_dataloader(config, rank, world_size)
    if len(dataloader) == 0:
        raise RuntimeError(
            "Training dataloader is empty. Increase data_repeat or reduce "
            f"training.batch_size_per_gpu={config.training.batch_size_per_gpu} "
            f"for dataset length {len(dataset)} across world_size={world_size}."
        )
    model = build_model(config, device, rank)
    loss_modules = build_loss_modules(config, device)
    if world_size > 1:
        device_ids = [local_rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids, find_unused_parameters=True)

    grad_accum = int(config.training.get("grad_accum_steps", 1))
    max_steps = int(config.training.max_fwdbwd_passes)
    job_overview = get_job_overview(
        num_gpus=world_size,
        num_epochs=int(config.training.get("num_epochs", 1000000)),
        num_train_samples=len(dataset),
        batch_size_per_gpu=int(config.training.batch_size_per_gpu),
        grad_accum_steps=grad_accum,
        max_fwdbwd_passes=max_steps,
    )
    optimizer = configure_optimizer(
        model,
        float(config.training.get("weight_decay", 0.05)),
        float(config.training.lr),
        (float(config.training.get("beta1", 0.9)), float(config.training.get("beta2", 0.95))),
    )
    lr_scheduler = configure_lr_scheduler(
        optimizer,
        job_overview.num_param_updates,
        int(config.training.get("warmup", 0)),
        str(config.training.get("scheduler_type", "cosine")),
    )
    resume_ckpt = config.training.get("resume_ckpt")
    step = 0
    param_update_step = 0
    if (
        resume_ckpt
        and str(resume_ckpt).lower() != "none"
        and not bool(config.training.get("reset_training_state", False))
    ):
        checkpoint = torch.load(resume_ckpt, map_location="cpu")
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        step = int(checkpoint.get("fwdbwd_pass_step", checkpoint.get("step", 0)))
        param_update_step = int(checkpoint.get("param_update_step", step))

    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config.training.get("use_amp", True))
        and str(config.training.get("amp_dtype", "bf16")).lower() == "fp16"
    )
    amp_dtype = torch.bfloat16 if str(config.training.get("amp_dtype", "bf16")).lower() == "bf16" else torch.float16
    amp_enabled = bool(config.training.get("use_amp", True)) and device.type == "cuda"

    checkpoint_dir = os.path.abspath(config.training.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if is_rank0(rank):
        with open(os.path.join(checkpoint_dir, "config.yaml"), "w", encoding="utf-8") as handle:
            yaml.safe_dump(_to_plain(config), handle, sort_keys=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    while step < max_steps:
        if hasattr(dataset, "update_iteration"):
            dataset.update_iteration(step)
        dataloader = make_dataloader(dataset, config, rank, world_size)
        if len(dataloader) == 0:
            raise RuntimeError(
                "Training dataloader is empty after rebuild. Increase data_repeat or reduce "
                f"training.batch_size_per_gpu={config.training.batch_size_per_gpu}."
            )
        cur_epoch = step // len(dataloader)
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(cur_epoch)
        epoch_step_limit = len(dataloader)
        if bool(config.training.view_selector.get("use_curriculum", False)):
            epoch_step_limit = min(epoch_step_limit, int(config.training.get("max_iter_epoch", 100)))

        epoch_steps = 0
        for batch in dataloader:
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            autocast_ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
                if device.type == "cuda"
                else nullcontext()
            )
            should_update = (step + 1) % grad_accum == 0
            sync_ctx = (
                model.no_sync()
                if isinstance(model, DDP) and not should_update
                else nullcontext()
            )
            with sync_ctx:
                with autocast_ctx:
                    result = model(batch, iter=step)
                    metrics = compute_loss(result, batch, config, loss_modules)
                    loss = metrics.loss / grad_accum
                scaler.scale(loss).backward()

            if should_update:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(config.training.get("grad_clip_norm", 1.0)),
                )
                allowed_gradnorm = float(config.training.get("grad_clip_norm", 1.0)) * float(
                    config.training.get("allowed_gradnorm_factor", 2.0)
                )
                no_pass_steps = int(config.training.get("no_pass_steps", 10))
                skip_optimizer_step = bool(grad_norm > allowed_gradnorm and (step + 1) > no_pass_steps)
                if skip_optimizer_step and is_rank0(rank):
                    print(
                        f"WARNING: step {step + 1} grad norm too large "
                        f"{float(grad_norm):.6f} > {allowed_gradnorm:.6f}, skipping optimizer step"
                    )
                if not skip_optimizer_step:
                    scaler.step(optimizer)
                    scaler.update()
                    param_update_step += 1
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                grad_norm = torch.tensor(float("nan"), device=device)

            step += 1
            epoch_steps += 1
            if is_rank0(rank) and (step == 1 or step % int(config.training.get("print_every", 20)) == 0):
                elapsed = time.time() - start_time
                print(
                    f"step {step}/{max_steps} "
                    f"param_update_step={param_update_step} "
                    f"loss={metrics.loss.item():.6f} "
                    f"mse={metrics.mse.item():.6f} "
                    f"l1={metrics.l1.item():.6f} "
                    f"perceptual={metrics.perceptual.item():.6f} "
                    f"psnr={metrics.psnr.item():.2f} "
                    f"lr={optimizer.param_groups[0]['lr']:.8f} "
                    f"grad_norm={float(grad_norm):.3f} "
                    f"time={elapsed:.1f}s"
                )

            if is_rank0(rank) and step % int(config.training.get("checkpoint_every", 2000)) == 0:
                save_checkpoint(
                    os.path.join(checkpoint_dir, f"ckpt_{step:016d}.pt"),
                    model,
                    optimizer,
                    lr_scheduler,
                    step,
                    param_update_step,
                    config,
                )

            if step >= max_steps or epoch_steps >= epoch_step_limit:
                break

    if is_rank0(rank) and bool(config.training.get("save_last", True)):
        save_checkpoint(
            os.path.join(checkpoint_dir, "last.pt"),
            model,
            optimizer,
            lr_scheduler,
            step,
            param_update_step,
            config,
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
