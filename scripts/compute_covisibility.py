"""Compute UFM covisibility or DINOv2 similarity for a converted DL3DV manifest.

See docs/data.md for usage. UFM mode requires a separate UFM installation and checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_dl3dv_manifest import read_json, read_list, reject_remote_path, resolve, scene_name


def read_scenes(manifest_path):
    """Use the same manifest, frame order, and path conventions as public training."""
    reject_remote_path(str(manifest_path), name="dataset manifest")
    manifest_path = Path(manifest_path).absolute()
    entries = read_list(str(manifest_path))
    if not entries:
        raise ValueError(f"No scenes found in {manifest_path}")
    for entry in entries:
        scene_path = resolve(entry, base_dir=str(manifest_path.parent))
        scene = read_json(scene_path)
        frames = scene.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"{scene_path} has no non-empty frames list")
        paths = [Path(resolve(str(frame["file_path"]), base_dir=str(Path(scene_path).parent)))
                 for frame in frames]
        yield scene_name(scene, scene_path), paths


def default_intervals(num_frames):
    if num_frames < 100:
        return [1, 2, 4, 6, 8, 16, 32, 40]
    if num_frames < 500:
        return [2, 5, 10, 15, 20, 40, 80, 100]
    return [5, 10, 20, 35, 50, 100, 200, 300]


def sample_triplets(num_frames, interval, count):
    import torch

    # Preserve the original start range and asymmetric +1 indexing exactly.
    max_start = num_frames - 2 * interval - 2
    if max_start <= 0:
        return []
    starts = torch.linspace(0, max_start - 1, steps=count).long().tolist()
    return [(s, s + interval + 1, s + 2 * interval + 1) for s in starts]


def load_image(path, image_size=256, patch_size=16):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image:
        # Match DL3DVDataset._load_and_transform_image with center_crop=True.
        image = image.convert("RGB")
        w, h = image.size
        scale = image_size / min(w, h)
        new_w, new_h = (int(w * scale), image_size) if h < w else (image_size, int(h * scale))
        new_w = max(patch_size, (new_w // patch_size) * patch_size)
        new_h = max(patch_size, (new_h // patch_size) * patch_size)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        # Center square crop, after resizing and rounding down to patch multiples.
        size = min(new_w, new_h)
        left, top = (new_w - size) // 2, (new_h - size) // 2
        image = image.crop((left, top, left + size, top + size))
        pixels = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1)


def summarize_covisibility(probabilities):
    """Reduce [num_triplets, 2, H, W] source-coordinate masks to scene stats."""
    if probabilities.ndim != 4 or probabilities.shape[1] != 2:
        raise ValueError("Expected masks with shape [num_triplets, 2, H, W]")
    if probabilities.shape[0] < 2:
        raise ValueError("At least two triplets are needed for sample standard deviation")
    masks = probabilities > 0.5
    union = masks.any(dim=1).float().mean(dim=(1, 2))
    per_view = masks.float().mean(dim=(2, 3)).mean(dim=1)
    # Match torch.std() in the original generator: sample std (N - 1).
    return {
        "avg_union": round(union.mean().item(), 4),
        "std_union": round(union.std(unbiased=True).item(), 4),
        "avg_per_view": round(per_view.mean().item(), 4),
        "std_per_view": round(per_view.std(unbiased=True).item(), 4),
    }


def compute_interval(model, paths, triplets, device, batch_size):
    import torch

    masks = []
    # Bound GPU memory by processing a few triplets (two image pairs each).
    with torch.inference_mode():
        for start in range(0, len(triplets), batch_size):
            indices = triplets[start : start + batch_size]
            images = {i: load_image(paths[i]) for i in sorted({i for t in indices for i in t})}
            batch = torch.stack([torch.stack([images[i] for i in t]) for t in indices])
            n, _, c, h, w = batch.shape
            source = batch[:, 1:2].repeat(1, 2, 1, 1, 1).reshape(-1, c, h, w)
            target = batch[:, [0, 2]].reshape(-1, c, h, w)
            # Preserve the original statistics generator's input convention:
            # [0, 1] float RGB with data_norm_type="dinov2", without mean/std
            # normalization. Changing this changes the released statistics.
            result = model.predict_correspondences_batched(
                source_image=source.to(device),
                target_image=target.to(device),
                data_norm_type="dinov2",
            )
            probability = result.covisibility.mask
            if tuple(probability.shape) != (n * 2, h, w):
                raise ValueError(f"Unexpected UFM mask shape: {tuple(probability.shape)}")
            masks.append(probability.reshape(n, 2, h, w).cpu())
    return summarize_covisibility(torch.cat(masks, dim=0))


def compute_dino_interval(model, paths, triplets, device, batch_size):
    import torch
    import torch.nn.functional as F

    scores = []
    with torch.inference_mode():
        for start in range(0, len(triplets), batch_size):
            indices = triplets[start : start + batch_size]
            images = {i: load_image(paths[i]) for i in sorted({i for t in indices for i in t})}
            batch = torch.stack([torch.stack([images[i] for i in t]) for t in indices])
            n, _, c, h, w = batch.shape
            images = batch.reshape(-1, c, h, w).to(device)
            mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            images = (images - mean) / std
            images = F.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
            features = model(images).reshape(n, 3, -1)  # Global DINOv2 features.
            left, middle, right = features.unbind(dim=1)
            scores.append(((F.cosine_similarity(left, middle, dim=1)
                            + F.cosine_similarity(middle, right, dim=1)) / 2).cpu())
    # Average triplets, including a potentially smaller last inference batch.
    return {"cos_sim": round(torch.cat(scores).mean().item(), 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["ufm", "dino"], default="ufm")
    parser.add_argument("--manifest-path", required=True,
                        help="Converted DL3DV manifest, as used by training")
    parser.add_argument("--checkpoint", type=Path, help="UFM ufm_560_base.pt; required for --mode ufm")
    parser.add_argument("--output-root", required=True,
                        help="Write <scene_name>/covisibility.json under this directory")
    parser.add_argument("--intervals", type=int, nargs="+", help="Override length-dependent intervals")
    parser.add_argument("--num-triplets", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1, help="Triplets per inference batch")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    min_triplets = 2 if args.mode == "ufm" else 1
    if args.num_triplets < min_triplets or args.batch_size < 1:
        parser.error(f"--num-triplets must be >= {min_triplets} and --batch-size must be >= 1")
    if args.intervals is not None and any(d < 1 for d in args.intervals):
        parser.error("--intervals must be positive")
    reject_remote_path(args.manifest_path, name="dataset manifest")
    reject_remote_path(args.output_root, name="statistics root")

    # Keep --help usable without the optional inference dependencies installed.
    if args.mode == "ufm":
        if args.checkpoint is None or not args.checkpoint.is_file():
            parser.error("--mode ufm requires --checkpoint pointing to an existing UFM checkpoint")
        from uniflowmatch.models.ufm import UniFlowMatchConfidence

        model = UniFlowMatchConfidence.from_pretrained_ckpt(str(args.checkpoint))
        compute = compute_interval
    else:
        import torch

        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        compute = compute_dino_interval
    model = model.eval().to(args.device)
    for name, paths in read_scenes(args.manifest_path):
        stats = {}
        for interval in args.intervals or default_intervals(len(paths)):
            triplets = sample_triplets(len(paths), interval, args.num_triplets)
            if not triplets:
                print(f"{name}: skipping interval {interval}: sequence too short")
                continue
            stats[str(interval)] = compute(
                model, paths, triplets, args.device, args.batch_size
            )
            print(name, interval, stats[str(interval)])
        if not stats:
            print(f"{name}: no valid intervals; no output written")
            continue
        output = Path(args.output_root) / name / "covisibility.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(stats, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
