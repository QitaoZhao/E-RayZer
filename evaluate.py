from __future__ import annotations

import argparse
import csv
import importlib
import os
from contextlib import nullcontext
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import erayzer_core  # noqa: F401
from erayzer_core.utils.camera_eval import evaluate_camera_pose_metrics
from training.datasets import build_dataset as build_config_dataset


def _to_edict(value: Any) -> Any:
    if isinstance(value, dict):
        return edict({k: _to_edict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_edict(v) for v in value]
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


def build_model(config: edict, device: torch.device, rank: int) -> torch.nn.Module:
    module_name, class_name = config.model.class_name.rsplit(".", 1)
    model_cls = importlib.import_module(module_name).__dict__[class_name]
    model = model_cls(config).to(device)
    ckpt_path = config.evaluation.checkpoint_path
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        print(f"Loaded checkpoint: {ckpt_path}")
    if missing and rank == 0:
        print(f"Missing keys: {len(missing)}")
    if unexpected and rank == 0:
        print(f"Unexpected keys: {len(unexpected)}")
    return model


def build_dataloader(config: edict, rank: int, world_size: int) -> DataLoader:
    training = config.training
    dataset = build_config_dataset(config, split="eval")
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    return DataLoader(
        dataset,
        batch_size=int(training.get("batch_size_per_gpu", 1)),
        shuffle=False,
        sampler=sampler,
        num_workers=int(training.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def _save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--set", "-s", action="append", nargs=2, default=[])
    args = parser.parse_args()

    config = load_config(args.config, args.set)
    config.inference = False
    device, rank, local_rank, world_size = setup_distributed()

    dataloader = build_dataloader(config, rank, world_size)
    model = build_model(config, device, rank)
    if world_size > 1:
        device_ids = [local_rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids)
    model.eval()

    out_dir = os.path.abspath(config.evaluation.out_dir)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    amp_dtype = torch.bfloat16 if str(config.training.get("amp_dtype", "bf16")).lower() == "bf16" else torch.float16
    amp_enabled = bool(config.training.get("use_amp", True)) and device.type == "cuda"
    max_batches = int(config.evaluation.get("max_batches", 0))
    save_examples = int(config.evaluation.get("save_examples", 8))

    mse_sum = torch.tensor(0.0, device=device)
    image_count = torch.tensor(0.0, device=device)
    sample_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
                if device.type == "cuda"
                else nullcontext()
            )
            with ctx:
                result = model(batch)

            pred = result.render.clamp(0, 1)
            target = result.get("target_image", batch["image"])
            if pred.shape[-2:] != target.shape[-2:]:
                b, v = target.shape[:2]
                target = F.interpolate(
                    target.reshape(b * v, *target.shape[2:]),
                    size=pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).reshape(b, v, target.shape[2], pred.shape[-2], pred.shape[-1])

            mse_per_image = (pred - target).pow(2).flatten(2).mean(dim=2)
            mse_sum += mse_per_image.sum()
            image_count += mse_per_image.numel()
            psnr_per_image = -10.0 * torch.log10(mse_per_image.clamp_min(1e-8))

            for sample_idx in range(pred.shape[0]):
                scene_id = int(batch["index"][sample_idx, 0, 1].detach().cpu().item())
                sample_mse = float(mse_per_image[sample_idx].mean().detach().cpu().item())
                sample_psnr = float(psnr_per_image[sample_idx].mean().detach().cpu().item())
                rpa = evaluate_camera_pose_metrics(
                    batch["c2w"][sample_idx].detach(),
                    result.c2w[sample_idx].detach(),
                )
                sample_rows.append(
                    {
                        "basename": f"{scene_id:06d}",
                        "mse": sample_mse,
                        "psnr": sample_psnr,
                        **rpa,
                    }
                )

            if rank == 0 and batch_idx < save_examples:
                sample_dir = os.path.join(out_dir, f"sample_{batch_idx:04d}")
                os.makedirs(sample_dir, exist_ok=True)
                for view_idx in range(pred.shape[1]):
                    _save_tensor_image(pred[0, view_idx], os.path.join(sample_dir, f"pred_{view_idx:02d}.png"))
                    _save_tensor_image(target[0, view_idx], os.path.join(sample_dir, f"target_{view_idx:02d}.png"))

            if max_batches > 0 and batch_idx + 1 >= max_batches:
                break

    if world_size > 1:
        dist.all_reduce(mse_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(image_count, op=dist.ReduceOp.SUM)
        gathered_rows: list[list[dict[str, Any]]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered_rows, sample_rows)
        sample_rows = [row for rows in gathered_rows for row in rows]

    if rank == 0:
        unique_rows = {}
        for row in sample_rows:
            unique_rows.setdefault(row["basename"], row)
        sample_rows = sorted(unique_rows.values(), key=lambda row: row["basename"])
        num_views = int(config.training.get("num_views", 0))
        dedup_image_count = len(sample_rows) * num_views
        raw_mse = (mse_sum / image_count.clamp_min(1)).item()
        raw_psnr = -10.0 * np.log10(max(raw_mse, 1e-8))
        avg_sample_mse = float(np.mean([row["mse"] for row in sample_rows])) if sample_rows else raw_mse
        avg_sample_psnr = float(np.mean([row["psnr"] for row in sample_rows])) if sample_rows else raw_psnr
        mse = avg_sample_mse
        psnr = avg_sample_psnr
        num_images = dedup_image_count if sample_rows else int(image_count.item())
        avg_rpa = {
            key: float(np.mean([row[key] for row in sample_rows])) if sample_rows else 0.0
            for key in ["RPA@5", "RPA@15", "RPA@30"]
        }
        summary_path = os.path.join(out_dir, "metrics.txt")
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(f"mse: {mse:.8f}\n")
            handle.write(f"psnr: {psnr:.4f}\n")
            handle.write(f"RPA@5: {avg_rpa['RPA@5']:.6f}\n")
            handle.write(f"RPA@15: {avg_rpa['RPA@15']:.6f}\n")
            handle.write(f"RPA@30: {avg_rpa['RPA@30']:.6f}\n")
            handle.write(f"num_images: {num_images}\n")
        csv_path = os.path.join(out_dir, "summary.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            fieldnames = ["basename", "mse", "psnr", "RPA@5", "RPA@15", "RPA@30"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in sample_rows:
                writer.writerow(
                    {
                        "basename": row["basename"],
                        "mse": f"{row['mse']:.8f}",
                        "psnr": f"{row['psnr']:.4f}",
                        "RPA@5": f"{row['RPA@5']:.6f}",
                        "RPA@15": f"{row['RPA@15']:.6f}",
                        "RPA@30": f"{row['RPA@30']:.6f}",
                    }
                )
            handle.write("\n")
            writer.writerow(
                {
                    "basename": "average",
                    "mse": f"{avg_sample_mse:.8f}",
                    "psnr": f"{avg_sample_psnr:.4f}",
                    "RPA@5": f"{avg_rpa['RPA@5']:.6f}",
                    "RPA@15": f"{avg_rpa['RPA@15']:.6f}",
                    "RPA@30": f"{avg_rpa['RPA@30']:.6f}",
                }
            )
        print(f"Evaluation written to {out_dir}")
        print(f"mse={mse:.8f}, psnr={psnr:.4f}, num_images={num_images}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
