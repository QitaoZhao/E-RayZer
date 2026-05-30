from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


DL3DV_WORLD_TO_OPENCV_WORLD = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
OPENGL_CAMERA_TO_OPENCV_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


def read_list(path: Path) -> list[Path]:
    with path.open("r", encoding="utf-8") as handle:
        lines = [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]
    base = path.parent
    return [
        Path(line) if Path(line).is_absolute() else (base / line).resolve()
        for line in lines
    ]


def read_scene_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def discover_camera_files(raw_root: Path) -> list[Path]:
    patterns = ("opencv_cameras.json", "opencv_cameras*.json", "transforms.json")
    files: set[Path] = set()
    for pattern in patterns:
        files.update(raw_root.rglob(pattern))

    by_scene_id: dict[str, Path] = {}
    for path in sorted(files):
        scene_id = camera_scene_id(path)
        current = by_scene_id.get(scene_id)
        if current is None or camera_file_priority(path) < camera_file_priority(current):
            by_scene_id[scene_id] = path
    return [by_scene_id[scene_id] for scene_id in sorted(by_scene_id)]


def camera_scene_id(path: Path) -> str:
    if path.name == "transforms.json" and path.parent.name in {"nerfstudio", "gaussian_splat"}:
        return path.parent.parent.name
    return path.parent.name


def camera_file_priority(path: Path) -> int:
    if path.name == "opencv_cameras.json":
        return 0
    if path.name.startswith("opencv_cameras"):
        return 1
    if path.name == "transforms.json":
        return 2
    return 3


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir: Path, files: Iterable[Path]) -> None:
    checksum_path = output_dir / "checksums.sha256"
    unique_files = sorted(
        {path for path in files if path.exists() and path.is_file()},
        key=lambda item: item.as_posix(),
    )
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in unique_files:
            rel = os.path.relpath(path, output_dir)
            handle.write(f"{sha256_file(path)}  {Path(rel).as_posix()}\n")


def copy_or_link(src: Path, dst: Path, *, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unsupported materialize mode: {mode}")


def copy_tree_or_link(src: Path, dst: Path, *, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    if mode == "copy":
        shutil.copytree(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=True)
    else:
        raise ValueError(f"Unsupported materialize mode: {mode}")


def common_top_frame_dir(frames: list[dict[str, Any]]) -> str | None:
    top_dirs: set[str] = set()
    for frame in frames:
        rel_path = Path(str(frame["file_path"]))
        if rel_path.is_absolute() or ".." in rel_path.parts or len(rel_path.parts) < 2:
            return None
        top_dirs.add(rel_path.parts[0])
    if len(top_dirs) != 1:
        return None
    return next(iter(top_dirs))


def normalize_frame(frame: dict[str, Any], *, rel_path: str) -> dict[str, Any]:
    required = ["w2c", "fx", "fy", "cx", "cy"]
    missing = [key for key in required if key not in frame]
    if missing:
        raise ValueError(f"Frame {rel_path!r} is missing keys: {', '.join(missing)}")
    out = {
        "file_path": rel_path,
        "w2c": frame["w2c"],
        "fx": frame["fx"],
        "fy": frame["fy"],
        "cx": frame["cx"],
        "cy": frame["cy"],
    }
    if "h" in frame:
        out["h"] = frame["h"]
    if "w" in frame:
        out["w"] = frame["w"]
    return out


def is_transforms_scene(camera_json: dict[str, Any]) -> bool:
    frames = camera_json.get("frames", [])
    return bool(frames and "transform_matrix" in frames[0] and "w2c" not in frames[0])


def resolve_source_image(camera_path: Path, src_rel_path: Path) -> tuple[Path, Path]:
    if src_rel_path.is_absolute():
        return src_rel_path, Path(src_rel_path.name)

    candidates = [src_rel_path]
    if src_rel_path.parts and src_rel_path.parts[0] == "images":
        rest = src_rel_path.parts[1:]
        for image_dir in ("images_4", "images_2", "images_8", "images_undistort"):
            candidates.append(Path(image_dir, *rest))

    for candidate_rel in candidates:
        candidate = camera_path.parent / candidate_rel
        if candidate.is_file():
            return candidate, candidate_rel

    return camera_path.parent / src_rel_path, src_rel_path


def transforms_c2w_to_opencv_w2c(transform_matrix: list[list[float]]) -> list[list[float]]:
    c2w = np.asarray(transform_matrix, dtype=np.float64)
    opencv_c2w = (
        DL3DV_WORLD_TO_OPENCV_WORLD
        @ c2w
        @ OPENGL_CAMERA_TO_OPENCV_CAMERA
    )
    return np.linalg.inv(opencv_c2w).tolist()


def transforms_camera_params(
    *,
    camera_json: dict[str, Any],
    image_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    image = cv2.imread(image_path.as_posix(), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    height, width = image.shape[:2]
    raw_width = float(camera_json.get("w", width))
    raw_height = float(camera_json.get("h", height))
    scale_x = width / raw_width
    scale_y = height / raw_height
    camera_matrix = np.array(
        [
            [float(camera_json["fl_x"]) * scale_x, 0.0, float(camera_json["cx"]) * scale_x],
            [0.0, float(camera_json["fl_y"]) * scale_y, float(camera_json["cy"]) * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.array(
        [
            float(camera_json.get("k1", 0.0)),
            float(camera_json.get("k2", 0.0)),
            float(camera_json.get("p1", 0.0)),
            float(camera_json.get("p2", 0.0)),
            float(camera_json.get("k3", 0.0)),
        ],
        dtype=np.float64,
    )
    undistorted_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        (width, height),
        0,
        (width, height),
        centerPrincipalPoint=False,
    )
    return camera_matrix, distortion, undistorted_matrix, width, height


def normalize_transforms_frame(
    frame: dict[str, Any],
    *,
    rel_path: str,
    camera_matrix: np.ndarray,
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "file_path": rel_path,
        "w2c": transforms_c2w_to_opencv_w2c(frame["transform_matrix"]),
        "h": height,
        "w": width,
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
    }


def select_entries(entries: list[Path], args: argparse.Namespace) -> list[Path]:
    if args.scene_id_list is not None:
        requested_ids = read_scene_ids(args.scene_id_list)
        if args.offset:
            requested_ids = requested_ids[args.offset :]
        if args.limit > 0:
            requested_ids = requested_ids[: args.limit]
        if args.sample_count > 0:
            if args.sample_count > len(requested_ids):
                raise ValueError(
                    f"--sample-count {args.sample_count} exceeds selected list length "
                    f"{len(requested_ids)}"
                )
            requested_ids = random.Random(args.seed).sample(requested_ids, args.sample_count)
        if not requested_ids:
            raise ValueError("No scene ids selected")
        by_scene_id: dict[str, Path] = {}
        duplicates: set[str] = set()
        for entry in entries:
            scene_id = camera_scene_id(entry)
            if scene_id in by_scene_id:
                duplicates.add(scene_id)
            by_scene_id[scene_id] = entry
        if duplicates:
            raise ValueError(
                "Duplicate opencv camera parent directories for scene ids: "
                + ", ".join(sorted(duplicates)[:10])
            )
        missing = [scene_id for scene_id in requested_ids if scene_id not in by_scene_id]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} scene id(s) from {args.scene_id_list} were not found "
                f"under the selected raw data. First missing id: {missing[0]}"
            )
        entries = [by_scene_id[scene_id] for scene_id in requested_ids]
    else:
        if args.offset:
            entries = entries[args.offset :]
        if args.limit > 0:
            entries = entries[: args.limit]
        if args.sample_count > 0:
            if args.sample_count > len(entries):
                raise ValueError(
                    f"--sample-count {args.sample_count} exceeds selected list length "
                    f"{len(entries)}"
                )
            entries = random.Random(args.seed).sample(entries, args.sample_count)
    return entries


def copy_stats(
    *,
    stats_root: Path | None,
    output_dir: Path,
    dataset_name: str,
    scene_name: str,
    output_subdir: str,
    overwrite: bool,
    written_files: list[Path],
) -> None:
    if stats_root is None:
        return
    src = stats_root / scene_name / "covisibility.json"
    if not src.is_file():
        raise FileNotFoundError(src)
    dst = output_dir / output_subdir / dataset_name / scene_name / "covisibility.json"
    copy_or_link(src, dst, mode="copy", overwrite=overwrite)
    written_files.append(dst)


def convert_sequence(
    *,
    camera_path: Path,
    output_dir: Path,
    dataset_name: str,
    scene_name_source: str,
    image_mode: str,
    copy_whole_image_dir: bool,
    overwrite: bool,
    covisibility_root: Path | None,
    feature_similarity_root: Path | None,
    written_files: list[Path],
) -> tuple[str, Path]:
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    with camera_path.open("r", encoding="utf-8") as handle:
        camera_json = json.load(handle)

    frames = camera_json.get("frames", [])
    if not frames:
        raise ValueError(f"{camera_path} has no frames")
    has_transforms = is_transforms_scene(camera_json)

    if scene_name_source == "json":
        scene_name = str(camera_json.get("scene_name") or camera_scene_id(camera_path))
    elif scene_name_source == "parent":
        scene_name = camera_scene_id(camera_path)
    else:
        raise ValueError(f"Unsupported scene_name_source: {scene_name_source}")
    scene_dir = output_dir / "scenes" / scene_name

    copied_image_dir = False
    if image_mode != "none" and copy_whole_image_dir and not has_transforms:
        frame_dir = common_top_frame_dir(frames)
        if frame_dir is not None:
            src_dir = camera_path.parent / frame_dir
            dst_dir = scene_dir / frame_dir
            if src_dir.is_dir():
                copy_tree_or_link(src_dir, dst_dir, mode=image_mode, overwrite=overwrite)
                copied_image_dir = True

    out_frames: list[dict[str, Any]] = []
    transform_camera_matrix: np.ndarray | None = None
    transform_distortion: np.ndarray | None = None
    transform_undistorted_matrix: np.ndarray | None = None
    transform_width = 0
    transform_height = 0
    for frame in frames:
        src_rel_path = Path(str(frame["file_path"]))
        src_image, actual_rel_path = resolve_source_image(camera_path, src_rel_path)

        if has_transforms:
            if transform_camera_matrix is None:
                (
                    transform_camera_matrix,
                    transform_distortion,
                    transform_undistorted_matrix,
                    transform_width,
                    transform_height,
                ) = transforms_camera_params(
                    camera_json=camera_json,
                    image_path=src_image,
                )
            rel_path = f"images_undistort/{src_image.name}"
            dst_image = scene_dir / rel_path
            dst_image.parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not dst_image.is_file():
                image = cv2.imread(src_image.as_posix(), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(src_image)
                undistorted = cv2.undistort(
                    image,
                    transform_camera_matrix,
                    transform_distortion,
                    None,
                    transform_undistorted_matrix,
                )
                cv2.imwrite(dst_image.as_posix(), undistorted)
            written_files.append(dst_image)
            out_frames.append(
                normalize_transforms_frame(
                    frame,
                    rel_path=rel_path,
                    camera_matrix=transform_undistorted_matrix,
                    width=transform_width,
                    height=transform_height,
                )
            )
            continue

        rel_path = actual_rel_path.as_posix()
        if image_mode == "none":
            rel_path = src_image.resolve().as_posix()
        elif src_rel_path.is_absolute() or ".." in src_rel_path.parts:
            rel_path = f"images/{src_rel_path.name}"

        dst_image = scene_dir / rel_path
        if image_mode != "none" and not copied_image_dir:
            if not src_image.is_file():
                raise FileNotFoundError(src_image)
            copy_or_link(src_image, dst_image, mode=image_mode, overwrite=overwrite)
            written_files.append(dst_image)
        elif image_mode != "none" and not dst_image.exists():
            raise FileNotFoundError(dst_image)

        out_frames.append(normalize_frame(frame, rel_path=rel_path))

    out_scene = {
        "scene_name": scene_name,
        "frames": out_frames,
        "source_camera_file": camera_path.name,
    }
    out_scene_path = scene_dir / "scene.json"
    write_json(out_scene_path, out_scene)
    written_files.append(out_scene_path)

    copy_stats(
        stats_root=covisibility_root,
        output_dir=output_dir,
        dataset_name=dataset_name,
        scene_name=scene_name,
        output_subdir="covisibility",
        overwrite=overwrite,
        written_files=written_files,
    )
    copy_stats(
        stats_root=feature_similarity_root,
        output_dir=output_dir,
        dataset_name=dataset_name,
        scene_name=scene_name,
        output_subdir="dino_similarity",
        overwrite=overwrite,
        written_files=written_files,
    )

    return scene_name, out_scene_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert local DL3DV raw opencv_cameras.json or Hugging Face "
            "transforms.json files into E-RayZer scene JSONs."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--raw-root",
        type=Path,
        help=(
            "Local DL3DV root downloaded from Hugging Face; transforms.json and "
            "opencv_cameras*.json files are discovered recursively."
        ),
    )
    source.add_argument(
        "--sequence-list",
        type=Path,
        help="Text file with one local transforms.json or opencv_cameras*.json path per line.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--dataset-name", default="dl3dv")
    parser.add_argument(
        "--scene-name-source",
        choices=["parent", "json"],
        default="parent",
        help=(
            "How to name output scenes. The release default is parent, which uses "
            "the camera JSON parent directory and keeps scene JSON, "
            "covisibility, and DINO feature-similarity paths aligned."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=0)
    parser.add_argument(
        "--scene-id-list",
        type=Path,
        help=(
            "Optional text file with one DL3DV sequence id per line. Sequence ids "
            "must match the parent directory of the camera JSON. This is the "
            "recommended release path for fixed mini-train and test splits."
        ),
    )
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--image-mode", choices=["copy", "symlink", "none"], default="copy")
    parser.add_argument("--copy-whole-image-dir", action="store_true")
    parser.add_argument("--covisibility-root", type=Path)
    parser.add_argument("--feature-similarity-root", type=Path)
    parser.add_argument("--include-source-paths", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.raw_root is not None:
        entries = discover_camera_files(args.raw_root.resolve())
    else:
        entries = read_list(args.sequence_list)
    entries = select_entries(entries, args)
    if not entries:
        raise ValueError("No transforms.json or opencv_cameras files selected")

    written_files: list[Path] = []
    manifest_entries: list[str] = []
    scene_ids: list[str] = []
    for camera_path in entries:
        scene_id, scene_path = convert_sequence(
            camera_path=camera_path,
            output_dir=output_dir,
            dataset_name=args.dataset_name,
            scene_name_source=args.scene_name_source,
            image_mode=args.image_mode,
            copy_whole_image_dir=args.copy_whole_image_dir,
            overwrite=args.overwrite,
            covisibility_root=args.covisibility_root.resolve() if args.covisibility_root else None,
            feature_similarity_root=(
                args.feature_similarity_root.resolve()
                if args.feature_similarity_root
                else None
            ),
            written_files=written_files,
        )
        scene_ids.append(scene_id)
        manifest_entries.append(
            Path(os.path.relpath(scene_path, output_dir / "manifests")).as_posix()
        )
        print(f"[{len(scene_ids)}/{len(entries)}] converted {scene_id}", flush=True)

    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / f"{args.split_name}.txt"
    scene_ids_path = manifests_dir / f"{args.split_name}_scene_ids.txt"
    manifest_path.write_text("\n".join(manifest_entries) + "\n", encoding="utf-8")
    scene_ids_path.write_text("\n".join(scene_ids) + "\n", encoding="utf-8")
    written_files.extend([manifest_path, scene_ids_path])

    source_list_path = None
    if args.include_source_paths:
        source_list_path = manifests_dir / f"{args.split_name}_source_camera_json.txt"
        source_list_path.write_text(
            "\n".join(path.as_posix() for path in entries) + "\n",
            encoding="utf-8",
        )
        written_files.append(source_list_path)

    metadata = {
        "dataset": args.dataset_name,
        "source_format": "local_dl3dv_camera_json",
        "split_name": args.split_name,
        "sequences": len(scene_ids),
        "seed": args.seed if args.sample_count > 0 else None,
        "sample_count": args.sample_count if args.sample_count > 0 else None,
        "image_mode": args.image_mode,
        "copy_whole_image_dir": args.copy_whole_image_dir,
        "has_covisibility": bool(args.covisibility_root),
        "has_feature_similarity": bool(args.feature_similarity_root),
        "manifest": f"manifests/{args.split_name}.txt",
        "scene_ids": f"manifests/{args.split_name}_scene_ids.txt",
        "source_camera_json": (
            f"manifests/{args.split_name}_source_camera_json.txt"
            if source_list_path is not None
            else None
        ),
        "format": {
            "manifest": "One scene JSON path per line, resolved relative to the manifest file.",
            "scene_json": "Each scene has scene_name and frames with file_path, w2c, fx, fy, cx, cy.",
        },
    }
    metadata_path = output_dir / "metadata.json"
    write_json(metadata_path, metadata)
    written_files.append(metadata_path)
    write_checksums(output_dir, written_files)

    print(f"Converted {len(scene_ids)} sequence(s)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
