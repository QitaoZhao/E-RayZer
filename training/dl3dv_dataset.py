from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ViewSelection:
    indices: List[int]
    target_covisibility: Optional[float] = None


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle]
    return [line for line in lines if line and not line.startswith("#")]


def _reject_remote_path(path: str, *, name: str) -> None:
    if path.startswith("s3://"):
        raise ValueError(
            f"Public E-RayZer training expects local {name}. Got S3 path: {path}"
        )


def _resolve_path(path: str, *, base_dir: str) -> str:
    _reject_remote_path(path, name="files")
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _scene_name(scene_json: Dict[str, Any], scene_path: str) -> str:
    if scene_json.get("scene_name"):
        return str(scene_json["scene_name"])
    return os.path.basename(os.path.dirname(scene_path))


def _sampling_interval_for_target(
    stats: Dict[str, Any],
    target: float,
    *,
    max_allowed_interval: float,
    key_name: str = "avg_per_view",
    tol: float = 0.05,
) -> Optional[float]:
    intervals = sorted(int(k) for k in stats.keys())
    if not intervals:
        return None

    values: List[Tuple[int, float]] = []
    for interval in intervals:
        item = stats[str(interval)]
        if key_name not in item:
            continue
        values.append((interval, float(item[key_name])))
    if not values:
        return None

    if target >= values[0][1] and target <= values[0][1] + tol:
        return values[0][0] if values[0][0] <= max_allowed_interval else None
    if target <= values[-1][1] and target >= values[-1][1] - tol:
        return values[-1][0] if values[-1][0] <= max_allowed_interval else None

    for (i1, cov1), (i2, cov2) in zip(values[:-1], values[1:]):
        if cov1 >= target >= cov2:
            if abs(cov1 - cov2) < 1e-6:
                estimate = float(i1)
            else:
                alpha = (target - cov2) / (cov1 - cov2)
                estimate = i2 + alpha * (i1 - i2)
            return estimate if estimate <= max_allowed_interval else None
    return None


class DL3DVDataset(Dataset):
    """Local DL3DV dataset used by the public training script.

    The list file contains one scene metadata JSON path per line. Each scene JSON
    follows the processed DL3DV format:

    {
      "scene_name": "optional_scene_id",
      "frames": [
        {"file_path": "images/000000.jpg", "w2c": [[...]], "fx": ..., "fy": ..., "cx": ..., "cy": ...}
      ]
    }
    """

    def __init__(
        self,
        dataset_path: str,
        *,
        image_size: int,
        patch_size: int = 16,
        num_views: int,
        num_input_views: int,
        view_selector: Dict[str, Any],
        covisibility_root: Optional[str] = None,
        feature_similarity_root: Optional[str] = None,
        center_crop: bool = True,
        random_shuffle: bool = False,
        max_resample_tries: int = 100,
        normalize_intrinsics: bool = True,
        data_repeat: int = 1,
        is_eval: bool = False,
    ) -> None:
        _reject_remote_path(dataset_path, name="dataset manifests")
        if covisibility_root:
            _reject_remote_path(covisibility_root, name="covisibility labels")
        if feature_similarity_root:
            _reject_remote_path(feature_similarity_root, name="feature similarity labels")

        self.dataset_path = os.path.abspath(dataset_path)
        self.dataset_dir = os.path.dirname(self.dataset_path)
        self.scene_paths = [
            _resolve_path(path, base_dir=self.dataset_dir)
            for path in _read_list(self.dataset_path)
        ]
        if not self.scene_paths:
            raise ValueError(f"No scene metadata paths found in {self.dataset_path}")

        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.num_views = int(num_views)
        self.num_input_views = int(num_input_views)
        self.view_selector = dict(view_selector or {"type": "random"})
        self.covisibility_root = (
            os.path.abspath(covisibility_root) if covisibility_root else None
        )
        self.feature_similarity_root = (
            os.path.abspath(feature_similarity_root) if feature_similarity_root else None
        )
        self.center_crop = bool(center_crop)
        self.random_shuffle = bool(random_shuffle)
        self.max_resample_tries = max(1, int(max_resample_tries))
        self.normalize_intrinsics = bool(normalize_intrinsics)
        self.data_repeat = max(1, int(data_repeat))
        self.is_eval = bool(is_eval)
        self.current_iteration = 0

    def __len__(self) -> int:
        return len(self.scene_paths) * self.data_repeat

    def update_iteration(self, iteration: int) -> None:
        self.current_iteration = int(iteration)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        for attempt in range(self.max_resample_tries):
            scene_index = index % len(self.scene_paths) if attempt == 0 else random.randrange(len(self.scene_paths))
            sample = self._try_get_scene(scene_index)
            if sample is not None:
                return sample
        raise RuntimeError(
            f"Could not sample {self.num_views} views after "
            f"{self.max_resample_tries} tries"
        )

    def _try_get_scene(self, scene_index: int) -> Optional[Dict[str, torch.Tensor]]:
        scene_path = self.scene_paths[scene_index]
        scene = _read_json(scene_path)
        frames = scene.get("frames", [])
        if len(frames) < self.num_views:
            return None

        name = _scene_name(scene, scene_path)
        selection = self._select_views(frames, name)
        if len(selection.indices) < self.num_views:
            return None

        scene_dir = os.path.dirname(scene_path)
        images: List[torch.Tensor] = []
        intrinsics: List[torch.Tensor] = []
        c2ws: List[torch.Tensor] = []

        for frame_idx in selection.indices:
            frame = frames[frame_idx]
            image_path = _resolve_path(str(frame["file_path"]), base_dir=scene_dir)
            image, intr = self._load_and_transform_image(image_path, frame)
            images.append(image)
            intrinsics.append(intr)

            w2c = np.asarray(frame["w2c"], dtype=np.float32)
            c2ws.append(torch.from_numpy(np.linalg.inv(w2c)).float())

        image_indices = torch.tensor(selection.indices, dtype=torch.long).unsqueeze(-1)
        scene_indices = torch.full_like(image_indices, scene_index)
        return {
            "image": torch.stack(images, dim=0),
            "fxfycxcy": torch.stack(intrinsics, dim=0),
            "c2w": torch.stack(c2ws, dim=0),
            "index": torch.cat([image_indices, scene_indices], dim=-1),
        }

    def _load_and_transform_image(
        self, image_path: str, frame: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        fx, fy, cx, cy = (
            float(frame["fx"]),
            float(frame["fy"]),
            float(frame["cx"]),
            float(frame["cy"]),
        )

        if height < width:
            resize_to_y = self.image_size
            resize_to_x = int(width * (self.image_size / height))
        else:
            resize_to_x = self.image_size
            resize_to_y = int(height * (self.image_size / width))
        resize_to_x = max(self.patch_size, (resize_to_x // self.patch_size) * self.patch_size)
        resize_to_y = max(self.patch_size, (resize_to_y // self.patch_size) * self.patch_size)
        scale_x = resize_to_x / width
        scale_y = resize_to_y / height
        image = image.resize((resize_to_x, resize_to_y), Image.Resampling.LANCZOS)
        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y
        width, height = image.size

        if self.center_crop and width != height:
            crop_size = min(width, height)
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            image = image.crop((left, top, left + crop_size, top + crop_size))
            cx -= left
            cy -= top
            width = height = crop_size

        if self.normalize_intrinsics:
            intr = torch.tensor(
                [fx / self.image_size, fy / self.image_size, cx / self.image_size, cy / self.image_size],
                dtype=torch.float32,
            )
        else:
            intr = torch.tensor([fx, fy, cx, cy], dtype=torch.float32)

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return tensor, intr

    def _select_views(self, frames: Sequence[Dict[str, Any]], scene_name: str) -> ViewSelection:
        selector_type = str(self.view_selector.get("type", "random"))
        if selector_type == "random":
            indices = sorted(random.sample(range(len(frames)), self.num_views))
            return ViewSelection(indices)
        if selector_type == "two_frame":
            return self._select_two_frame(frames)
        if selector_type in {"covisibility", "feat_sim"}:
            return self._select_stats_based_views(frames, scene_name, selector_type)
        raise ValueError(f"Unsupported view_selector.type: {selector_type}")

    def _select_two_frame(self, frames: Sequence[Dict[str, Any]]) -> ViewSelection:
        use_curriculum = bool(self.view_selector.get("use_curriculum", False))
        if use_curriculum:
            max_iter = max(1, int(self.view_selector.get("curriculum_iter", 30000)))
            progress = min(self.current_iteration / max_iter, 1.0)
            final_min = float(self.view_selector.get("min_frame_dist", 10))
            final_max = float(self.view_selector.get("max_frame_dist", final_min))
            start_min = float(self.view_selector.get("curriculum_start_min_frame_dist", 48))
            start_max = float(self.view_selector.get("curriculum_start_max_frame_dist", 64))
            min_dist = int(start_min + (final_min - start_min) * progress)
            max_dist = int(start_max + (final_max - start_max) * progress)
        else:
            min_dist = int(self.view_selector.get("min_frame_dist", 10))
            max_dist = int(self.view_selector.get("max_frame_dist", min_dist))
        max_dist = max(min_dist, max_dist)
        frame_dist = random.randint(min_dist, max_dist)
        if len(frames) <= frame_dist:
            return ViewSelection([])
        if self.is_eval:
            indices = np.linspace(0, frame_dist, self.num_views, dtype=int).tolist()
            return ViewSelection(indices, float(frame_dist))
        start = random.randint(0, len(frames) - frame_dist - 1)
        end = start + frame_dist
        middle_count = self.num_views - 2
        if frame_dist - 1 < middle_count:
            return ViewSelection([])
        middle = sorted(random.sample(range(start + 1, end), middle_count))
        return ViewSelection([start, *middle, end], float(frame_dist))

    def _select_stats_based_views(
        self,
        frames: Sequence[Dict[str, Any]],
        scene_name: str,
        selector_type: str,
    ) -> ViewSelection:
        if selector_type == "feat_sim":
            stats_root = self.feature_similarity_root
            default_key = "cos_sim"
            root_name = "feature_similarity_root"
        else:
            stats_root = self.covisibility_root
            default_key = "avg_per_view"
            root_name = "covisibility_root"
        if stats_root is None:
            raise ValueError(f"dataset.{root_name} is required for {selector_type} sampling")

        progress = 1.0
        if self.view_selector.get("use_curriculum", False):
            max_iter = max(1, int(self.view_selector.get("curriculum_iter", 30000)))
            progress = min(self.current_iteration / max_iter, 1.0)

        final_min = float(self.view_selector.get("max_frame_dist", 0.5))
        final_max = float(self.view_selector.get("min_frame_dist", 1.0))
        start_min = float(self.view_selector.get("curriculum_start_max_frame_dist", final_min))
        start_max = float(self.view_selector.get("curriculum_start_min_frame_dist", final_max))
        cur_min = start_min + (final_min - start_min) * progress
        cur_max = start_max + (final_max - start_max) * progress
        target = random.uniform(cur_min, cur_max)

        stats_path = os.path.join(stats_root, scene_name, "covisibility.json")
        if not os.path.isfile(stats_path):
            return ViewSelection([])
        stats = _read_json(stats_path)

        max_interval = len(frames) / max(1, self.num_input_views - 1)
        interval = _sampling_interval_for_target(
            stats,
            target,
            max_allowed_interval=max_interval,
            key_name=str(self.view_selector.get("key_name", default_key)),
            tol=float(self.view_selector.get("tol", 0.05)),
        )
        if interval is None or not math.isfinite(interval):
            return ViewSelection([])

        interval = max(1, int(interval))
        total_dist = max(self.num_views, (self.num_input_views - 1) * interval)
        if len(frames) <= total_dist:
            return ViewSelection([])

        start = random.randint(0, len(frames) - total_dist - 1)
        end = start + total_dist
        middle_count = self.num_views - 2
        sort_frames = bool(self.view_selector.get("sort_frames", True))
        if sort_frames:
            middle = sorted(random.sample(range(start + 1, end), middle_count))
            indices = [start, *middle, end]
        else:
            indices = random.sample(range(start, end + 1), self.num_views)
        if self.random_shuffle:
            random.shuffle(indices)
        return ViewSelection(indices, target)
