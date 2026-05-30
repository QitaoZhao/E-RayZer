from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List


def reject_remote_path(path: str, *, name: str) -> None:
    if path.startswith("s3://"):
        raise ValueError(f"Expected local {name}, got S3 path: {path}")


def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve(path: str, *, base_dir: str) -> str:
    reject_remote_path(path, name="file")
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def scene_name(scene: Dict[str, Any], scene_path: str) -> str:
    return str(scene.get("scene_name") or os.path.basename(os.path.dirname(scene_path)))


def require_keys(item: Dict[str, Any], keys: Iterable[str], *, context: str) -> None:
    missing = [key for key in keys if key not in item]
    if missing:
        raise ValueError(f"{context} is missing keys: {', '.join(missing)}")


def check_scene(scene_path: str, *, covisibility_root: str | None) -> None:
    scene = read_json(scene_path)
    frames = scene.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{scene_path} has no non-empty frames list")

    scene_dir = os.path.dirname(scene_path)
    for frame_idx, frame in enumerate(frames[:2]):
        require_keys(
            frame,
            ["file_path", "w2c", "fx", "fy", "cx", "cy"],
            context=f"{scene_path} frame {frame_idx}",
        )
        image_path = resolve(str(frame["file_path"]), base_dir=scene_dir)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)

    if covisibility_root:
        covis_path = os.path.join(
            covisibility_root,
            scene_name(scene, scene_path),
            "covisibility.json",
        )
        if not os.path.isfile(covis_path):
            raise FileNotFoundError(covis_path)
        covis = read_json(covis_path)
        if not covis:
            raise ValueError(f"{covis_path} is empty")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="dl3dv")
    parser.add_argument("--manifest-path")
    parser.add_argument("--dataset-path", help=argparse.SUPPRESS)
    parser.add_argument("--covisibility-root")
    parser.add_argument("--max-scenes", type=int, default=8)
    args = parser.parse_args()

    if args.dataset_name != "dl3dv":
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    manifest_path = args.manifest_path or args.dataset_path
    if not manifest_path:
        raise ValueError("--manifest-path is required")

    reject_remote_path(manifest_path, name="dataset manifest")
    if args.covisibility_root:
        reject_remote_path(args.covisibility_root, name="covisibility root")

    dataset_path = os.path.abspath(manifest_path)
    dataset_dir = os.path.dirname(dataset_path)
    scene_paths = [
        resolve(path, base_dir=dataset_dir)
        for path in read_list(dataset_path)
    ]
    if not scene_paths:
        raise ValueError(f"No scenes found in {dataset_path}")

    checked = 0
    for scene_path in scene_paths[: args.max_scenes]:
        check_scene(scene_path, covisibility_root=args.covisibility_root)
        checked += 1

    print(f"OK: checked {checked} scene(s) from {dataset_path}")


if __name__ == "__main__":
    main()
