from __future__ import annotations

import argparse
import csv
import os
import pickle
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download


META_URL = "https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv"
BENCHMARK_REPO = "DL3DV/DL3DV-Benchmark"
RESOLUTION_TO_REPO = {
    "480P": "DL3DV/DL3DV-ALL-480P",
    "960P": "DL3DV/DL3DV-ALL-960P",
    "2K": "DL3DV/DL3DV-ALL-2K",
    "4K": "DL3DV/DL3DV-ALL-4K",
}


def read_scene_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def download_metadata(cache_dir: Path, metadata_csv: Path | None) -> Path:
    if metadata_csv is not None:
        return metadata_csv

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "DL3DV-valid.csv"
    if not path.is_file():
        print(f"Downloading DL3DV metadata: {META_URL}", flush=True)
        urllib.request.urlretrieve(META_URL, path)
    return path


def load_scene_batches(metadata_path: Path) -> dict[str, str]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"hash", "batch"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{metadata_path} must contain columns: {', '.join(sorted(required))}"
            )
        return {str(row["hash"]): str(row["batch"]) for row in reader}


def extract_zip(zip_path: Path, output_dir: Path, batch: str) -> None:
    batch_dir = output_dir / batch
    batch_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(batch_dir)


def download_with_retries(
    *,
    repo_id: str,
    filename: str,
    output_dir: Path,
    cache_dir: Path,
    token: str | None,
    max_retries: int,
) -> Path:
    last_error: BaseException | None = None
    for attempt in range(1, max_retries + 1):
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                local_dir=output_dir,
                cache_dir=cache_dir,
                token=token,
            )
            return Path(path)
        except KeyboardInterrupt:
            raise
        except BaseException as error:
            last_error = error
            if attempt == max_retries:
                break
            sleep_s = min(30, 2 * attempt)
            print(
                f"Download failed for {filename}; retrying in {sleep_s}s "
                f"({attempt}/{max_retries})",
                flush=True,
            )
            time.sleep(sleep_s)

    raise RuntimeError(
        f"Failed to download {repo_id}/{filename}. Make sure your Hugging Face "
        f"account has access to https://huggingface.co/datasets/{repo_id} and "
        "that HF_TOKEN is set or you are logged in with `hf auth login`."
    ) from last_error


def list_benchmark_files(
    *,
    api: HfApi,
    repo_id: str,
    scene_id: str,
) -> list[str]:
    base = f"{scene_id}/nerfstudio"
    files = [f"{base}/transforms.json"]
    image_items = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo=f"{base}/images_4",
        recursive=False,
        expand=False,
    )
    for item in image_items:
        path = getattr(item, "path", "")
        if path.lower().endswith((".png", ".jpg", ".jpeg")):
            files.append(path)
    if len(files) <= 1:
        raise FileNotFoundError(f"No benchmark images found for {scene_id} in {repo_id}")
    return files


def benchmark_erayzer_files(all_files: list[str]) -> list[str]:
    files: list[str] = []
    for path in all_files:
        parts = Path(path).parts
        if len(parts) < 3 or parts[1] != "nerfstudio":
            continue
        if parts[2] == "transforms.json":
            files.append(path)
        elif (
            parts[2] == "images_4"
            and path.lower().endswith((".png", ".jpg", ".jpeg"))
        ):
            files.append(path)
    return files


def load_benchmark_filelist(
    *,
    repo_id: str,
    output_dir: Path,
    cache_dir: Path,
    token: str | None,
    max_retries: int,
) -> dict[str, list[str]]:
    benchmark_root = output_dir / "benchmark"
    download_with_retries(
        repo_id=repo_id,
        filename="benchmark-meta.csv",
        output_dir=benchmark_root,
        cache_dir=cache_dir,
        token=token,
        max_retries=max_retries,
    )
    filelist_path = download_with_retries(
        repo_id=repo_id,
        filename=".cache/filelist.bin",
        output_dir=benchmark_root,
        cache_dir=cache_dir,
        token=token,
        max_retries=max_retries,
    )
    with Path(filelist_path).open("rb") as handle:
        return pickle.load(handle)


def download_benchmark_scene(
    *,
    api: HfApi,
    repo_id: str,
    scene_id: str,
    output_dir: Path,
    cache_dir: Path,
    token: str | None,
    max_retries: int,
    max_workers: int,
    filepath_dict: dict[str, list[str]] | None = None,
) -> None:
    if filepath_dict is None:
        files = list_benchmark_files(api=api, repo_id=repo_id, scene_id=scene_id)
    else:
        if scene_id not in filepath_dict:
            raise ValueError(f"{scene_id} is not in benchmark filelist")
        files = benchmark_erayzer_files(filepath_dict[scene_id])
    download_benchmark_files_parallel(
        repo_id=repo_id,
        filenames=files,
        output_dir=output_dir,
        cache_dir=cache_dir,
        token=token,
        max_retries=max_retries,
        max_workers=max_workers,
        label=scene_id,
    )


def download_benchmark_scenes_bulk(
    *,
    repo_id: str,
    scene_ids: list[str],
    output_dir: Path,
    cache_dir: Path,
    token: str | None,
    max_retries: int,
    max_workers: int,
) -> None:
    filepath_dict = load_benchmark_filelist(
        repo_id=repo_id,
        output_dir=output_dir,
        cache_dir=cache_dir,
        token=token,
        max_retries=max_retries,
    )
    filenames: list[str] = []
    for index, scene_id in enumerate(scene_ids, start=1):
        if scene_id not in filepath_dict:
            raise ValueError(f"{scene_id} is not in benchmark filelist")
        filenames.extend(benchmark_erayzer_files(filepath_dict[scene_id]))
        if index == 1 or index % 10 == 0 or index == len(scene_ids):
            print(
                f"Listed benchmark scenes {index}/{len(scene_ids)} "
                f"({len(filenames)} files)",
                flush=True,
            )
    filenames = list(dict.fromkeys(filenames))
    download_benchmark_files_parallel(
        repo_id=repo_id,
        filenames=filenames,
        output_dir=output_dir,
        cache_dir=cache_dir,
        token=token,
        max_retries=max_retries,
        max_workers=max_workers,
        label="benchmark files",
    )


def download_benchmark_files_parallel(
    *,
    repo_id: str,
    filenames: list[str],
    output_dir: Path,
    cache_dir: Path,
    token: str | None,
    max_retries: int,
    max_workers: int,
    label: str,
) -> None:
    benchmark_root = output_dir / "benchmark"
    total = len(filenames)
    if total == 0:
        raise ValueError("No benchmark files selected")

    def download_one(filename: str) -> None:
        download_with_retries(
            repo_id=repo_id,
            filename=filename,
            output_dir=benchmark_root,
            cache_dir=cache_dir,
            token=token,
            max_retries=max_retries,
        )

    print(
        f"Downloading {total} benchmark file(s) with {max_workers} workers",
        flush=True,
    )
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, filename): filename for filename in filenames}
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed == 1 or completed % 500 == 0 or completed == total:
                print(f"    {label}: {completed}/{total}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download raw DL3DV sequences selected by a fixed sequence-id list "
            "from the official Hugging Face images+poses release."
        )
    )
    parser.add_argument("--scene-id-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--resolution",
        choices=sorted(RESOLUTION_TO_REPO),
        default="960P",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "all", "benchmark"],
        default="auto",
        help=(
            "auto uses DL3DV-ALL zip files when available and falls back to "
            "DL3DV-Benchmark. all requires every id to exist in DL3DV-ALL. "
            "benchmark downloads every id from DL3DV-Benchmark."
        ),
    )
    parser.add_argument("--benchmark-repo", default=BENCHMARK_REPO)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--keep-zip", action="store_true")
    parser.add_argument("--clean-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    cache_dir = output_dir / ".cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_ids = read_scene_ids(args.scene_id_list)
    if args.offset:
        scene_ids = scene_ids[args.offset :]
    if args.limit > 0:
        scene_ids = scene_ids[: args.limit]
    if not scene_ids:
        raise ValueError("No scene ids selected")

    metadata_path = None
    scene_batches: dict[str, str] = {}
    missing_from_all: list[str] = []
    if args.source in {"auto", "all"}:
        metadata_path = download_metadata(cache_dir, args.metadata_csv)
        scene_batches = load_scene_batches(metadata_path)
        missing_from_all = [
            scene_id for scene_id in scene_ids if scene_id not in scene_batches
        ]
    if args.source == "all" and missing_from_all:
        raise ValueError(
            f"{len(missing_from_all)} scene id(s) were not found in {metadata_path}. "
            f"First missing id: {missing_from_all[0]}"
        )
    if args.source == "benchmark":
        missing_from_all = list(scene_ids)
    elif missing_from_all:
        print(
            f"{len(missing_from_all)} scene id(s) are not in {metadata_path.name}; "
            f"falling back to {args.benchmark_repo}.",
            flush=True,
        )

    repo_id = RESOLUTION_TO_REPO[args.resolution]
    if args.source != "benchmark":
        print(f"Repo: {repo_id}", flush=True)
    if args.source == "benchmark" or missing_from_all:
        print(f"Benchmark fallback repo: {args.benchmark_repo}", flush=True)
    print(f"Scenes: {len(scene_ids)}", flush=True)
    downloaded = 0
    api = HfApi()

    if args.source == "benchmark" and not args.dry_run:
        download_benchmark_scenes_bulk(
            repo_id=args.benchmark_repo,
            scene_ids=scene_ids,
            output_dir=output_dir,
            cache_dir=cache_dir,
            token=args.token,
            max_retries=args.max_retries,
            max_workers=args.max_workers,
        )
        if args.clean_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)
        for scene_id in scene_ids:
            scene_dir = output_dir / "benchmark" / scene_id / "nerfstudio"
            if not scene_dir.is_dir():
                raise FileNotFoundError(scene_dir)
        print(f"Downloaded {len(scene_ids)}/{len(scene_ids)} scene(s) to {output_dir}")
        return

    for index, scene_id in enumerate(scene_ids, start=1):
        if args.source != "benchmark" and scene_id in scene_batches:
            batch = scene_batches[scene_id]
            scene_dir = output_dir / batch / scene_id
            rel_zip = f"{batch}/{scene_id}.zip"
        else:
            batch = None
            scene_dir = output_dir / "benchmark" / scene_id / "nerfstudio"
            rel_zip = None

        if args.dry_run:
            if rel_zip is None:
                print(
                    f"[{index}/{len(scene_ids)}] {args.benchmark_repo}/"
                    f"{scene_id}/nerfstudio/[transforms.json + images_4]",
                    flush=True,
                )
            else:
                print(f"[{index}/{len(scene_ids)}] {repo_id}/{rel_zip}", flush=True)
            continue

        if scene_dir.is_dir() and not args.overwrite:
            print(f"[{index}/{len(scene_ids)}] exists {scene_dir}", flush=True)
            downloaded += 1
            continue

        if rel_zip is None:
            download_benchmark_scene(
                api=api,
                repo_id=args.benchmark_repo,
                scene_id=scene_id,
                output_dir=output_dir,
                cache_dir=cache_dir,
                token=args.token,
                max_retries=args.max_retries,
                max_workers=args.max_workers,
                filepath_dict=None,
            )
        else:
            zip_path = download_with_retries(
                repo_id=repo_id,
                filename=rel_zip,
                output_dir=output_dir,
                cache_dir=cache_dir,
                token=args.token,
                max_retries=args.max_retries,
            )
            extract_zip(zip_path, output_dir, batch)
            if not args.keep_zip:
                zip_path.unlink(missing_ok=True)
        if args.clean_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)

        if not scene_dir.is_dir():
            raise FileNotFoundError(
                f"Expected extracted scene directory was not created: {scene_dir}"
            )
        print(f"[{index}/{len(scene_ids)}] downloaded {scene_id}", flush=True)
        downloaded += 1

    if args.dry_run:
        print("Dry run complete.")
    else:
        print(f"Downloaded {downloaded}/{len(scene_ids)} scene(s) to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
