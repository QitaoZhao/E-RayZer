from __future__ import annotations

import argparse
import functools
import os
import shutil
import time
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import gradio as gr
import numpy as np
import torch
import yaml
from PIL import Image
from huggingface_hub import hf_hub_download

from app_core.engine import get_engine

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(THIS_DIR, "config", "erayzer_inference.yaml")
DEFAULT_OUTPUT_ROOT = os.path.join(THIS_DIR, "experiments", "inference", "erayzer")
EXAMPLES_DIR = os.path.join(THIS_DIR, "examples")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
HF_REPO_ID = "qitaoz/E-RayZer"
HF_REVISION = "main"
HF_CKPT_FILENAME = "checkpoints/erayzer_multi.pt"
DEFAULT_LOCAL_CKPT = os.path.join(THIS_DIR, HF_CKPT_FILENAME)
DEFAULT_CKPT = f"https://huggingface.co/{HF_REPO_ID}/blob/{HF_REVISION}/{HF_CKPT_FILENAME}"

EXAMPLES_LIST: List[str] = []
EXAMPLES_FULL: List[List[List[str]]] = []


def _is_hf_url(value: str) -> bool:
    return value.startswith("https://huggingface.co/")


def _parse_hf_url(url: str) -> Tuple[str, str, str]:
    """Extract repo id, revision, and file path from a huggingface.co URL."""
    parsed = urlparse(url)
    if parsed.netloc != "huggingface.co":
        raise ValueError(f"Unsupported checkpoint host: {parsed.netloc}")

    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) < 3:
        raise ValueError(f"Malformed Hugging Face URL: {url}")

    repo_id = "/".join(segments[:2])
    pointer = segments[2]
    if pointer in {"blob", "resolve", "raw"}:
        if len(segments) < 5:
            raise ValueError(f"Missing file path in Hugging Face URL: {url}")
        revision = segments[3]
        file_path = "/".join(segments[4:])
    else:
        revision = HF_REVISION
        file_path = "/".join(segments[2:])
    return repo_id, revision, file_path


def _maybe_existing_checkpoint(path: str) -> Optional[str]:
    expanded = os.path.expanduser(path)
    if os.path.isfile(expanded):
        return os.path.abspath(expanded)
    if not os.path.isabs(expanded):
        repo_relative = os.path.join(THIS_DIR, expanded)
        if os.path.isfile(repo_relative):
            return repo_relative
    return None


def _get_default_checkpoint() -> str:
    existing = _maybe_existing_checkpoint(DEFAULT_LOCAL_CKPT)
    if existing:
        return existing
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_CKPT_FILENAME,
        revision=HF_REVISION,
    )


def _resolve_checkpoint_path(ckpt_arg: Optional[str]) -> str:
    """Resolve user-provided checkpoint path or download from Hugging Face."""
    if ckpt_arg:
        if ckpt_arg == DEFAULT_CKPT:
            return _get_default_checkpoint()
        existing = _maybe_existing_checkpoint(ckpt_arg)
        if existing:
            return existing
        default_candidates = {
            DEFAULT_LOCAL_CKPT,
            os.path.relpath(DEFAULT_LOCAL_CKPT, THIS_DIR),
            HF_CKPT_FILENAME,
        }
        if os.path.normpath(os.path.expanduser(ckpt_arg)) in {
            os.path.normpath(candidate) for candidate in default_candidates
        }:
            return _get_default_checkpoint()
        if _is_hf_url(ckpt_arg):
            repo_id, revision, file_path = _parse_hf_url(ckpt_arg)
            return hf_hub_download(repo_id=repo_id, filename=file_path, revision=revision)
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_arg}")
    return _get_default_checkpoint()


def info_fn() -> None:
    gr.Info("Images prepared for E-RayZer inference!")


def get_select_index(evt: gr.SelectData):
    if not EXAMPLES_FULL:
        raise gr.Error("No bundled examples available in this build.")

    index = evt.index
    if isinstance(index, (list, tuple)):
        index = index[-1]
    if index is None or index < 0 or index >= len(EXAMPLES_FULL):
        raise gr.Error("Invalid example selection.")
    return EXAMPLES_FULL[index][0], index


def check_img_input(batch):
    if not batch or not batch.get("image_paths"):
        raise gr.Error(
            "Please upload or select images, then preprocess them before running inference."
        )


def _discover_examples(root: str) -> Tuple[List[str], List[List[List[str]]]]:
    if not os.path.isdir(root):
        return [], []

    categories: List[str] = []
    bundles: List[List[List[str]]] = []
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        files = [
            os.path.join(folder, file)
            for file in sorted(os.listdir(folder))
            if os.path.splitext(file)[1].lower() in IMAGE_EXTS
        ]
        if files:
            categories.append(name)
            bundles.append([files])
    return categories, bundles


def _materialize_paths(file_block: Sequence[object]) -> List[str]:
    paths: List[str] = []
    for item in file_block or []:
        # Gradio Files components typically return objects with .name, or raw strings.
        if hasattr(item, "name") and item.name:
            paths.append(item.name)
        else:
            paths.append(str(item))
    return paths


def _load_image(path: str) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))


def _load_gallery(paths: Sequence[str]) -> List[np.ndarray]:
    return [_load_image(path) for path in paths]


def _load_yaml(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_config_path(config_path: str, value: Optional[str], fallback: str) -> str:
    path = value or fallback
    if os.path.isabs(path):
        return path
    repo_root = os.path.dirname(os.path.abspath(config_path))
    if os.path.basename(repo_root) == "config":
        repo_root = os.path.dirname(repo_root)
    return os.path.normpath(os.path.join(repo_root, path))


def _default_paths_for_config(config_path: str) -> Tuple[str, str]:
    config = _load_yaml(config_path)
    return (
        _resolve_config_path(
            config_path,
            config.get("inference_checkpoint_path"),
            DEFAULT_CKPT,
        ),
        _resolve_config_path(
            config_path,
            config.get("inference_out_dir"),
            DEFAULT_OUTPUT_ROOT,
        ),
    )


def preprocess(
    output_root: str,
    image_block: Sequence[object],
    selected: Optional[int] = None,
):
    local_paths = _materialize_paths(image_block)
    if not local_paths:
        raise gr.Error("Please upload images or pick an example before preprocessing.")

    cate_name = (
        time.strftime("%m%d_%H%M%S")
        if selected is None or selected >= len(EXAMPLES_LIST)
        else EXAMPLES_LIST[selected]
    )

    demo_dir = os.path.join(output_root, "demo", cate_name)
    shutil.rmtree(demo_dir, ignore_errors=True)
    source_dir = os.path.join(demo_dir, "source")
    processed_dir = os.path.join(demo_dir, "processed")  # reserved for future processing
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    processed_paths: List[str] = []
    processed_gallery: List[np.ndarray] = []
    for src_path in local_paths:
        fname = os.path.basename(src_path)
        dest_path = os.path.join(source_dir, fname)
        shutil.copy(src_path, dest_path)
        processed_paths.append(dest_path)
        processed_gallery.append(_load_image(dest_path))

    batch = {"cate_name": cate_name, "image_paths": processed_paths}
    return processed_gallery, batch


def run_inference(
    defaults: Dict[str, str],
    batch: Dict[str, object],
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    check_img_input(batch)

    engine = get_engine(
        defaults["config"],
        defaults["ckpt"],
        defaults["device"],
        defaults["output_dir"],
    )

    progress(0.1, desc="Running E-RayZer inference")
    gallery_paths, archive, log, glb_path, video_path = engine.run(
        batch["image_paths"]
    )
    progress(1.0, desc="Done")

    model_asset = glb_path if glb_path and os.path.exists(glb_path) else None
    video_asset = video_path if video_path and os.path.exists(video_path) else None

    return _load_gallery(gallery_paths), model_asset, video_asset, archive, log


def build_demo(args) -> gr.Blocks:
    global EXAMPLES_LIST, EXAMPLES_FULL
    EXAMPLES_LIST, EXAMPLES_FULL = _discover_examples(EXAMPLES_DIR)

    config_path = args.config or DEFAULT_CONFIG
    config = _load_yaml(config_path)
    inferred_device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = _resolve_checkpoint_path(args.ckpt)
    defaults = {
        "config": config_path,
        "ckpt": ckpt_path,
        "device": inferred_device,
        "output_dir": args.output_dir or _resolve_config_path(
            config_path,
            config.get("inference_out_dir"),
            DEFAULT_OUTPUT_ROOT,
        ),
    }

    preprocess_fn = functools.partial(preprocess, defaults["output_dir"])
    run_inference_fn = functools.partial(run_inference, defaults)

    _TITLE = "E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training"
    _DESCRIPTION = """
    <div>
    <a style="display:inline-block" href="https://qitaozhao.github.io/E-RayZer"><img src='https://img.shields.io/badge/project_page-8A2BE2'></a>
    <a style="display:inline-block; margin-left: .5em" href='https://github.com/QitaoZhao/E-RayZer'><img src='https://img.shields.io/github/stars/QitaoZhao/E-RayZer?style=social'/></a>
    </div>
    E-RayZer reconstructs camera poses, target-view renders, Gaussian point clouds, and turntable videos from unordered multi-view images.
    """

    # Use helper so theme doesn’t break older Gradio.
    with gr.Blocks(title=_TITLE) as demo:
        gr.Markdown(f"# {_TITLE}")
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            # Left column: inputs & controls
            with gr.Column(scale=2):
                image_block = gr.Files(
                    label="Upload multi-view images",
                    file_count="multiple",
                    file_types=["image"],
                )

                gr.Markdown(
                    "Upload your images above or pick a curated example below."
                )

                max_examples = 5
                gallery_value = (
                    [example[0][0] for example in EXAMPLES_FULL]
                    if EXAMPLES_FULL
                    else []
                )
                visible_examples = gallery_value[:max_examples]

                examples_gallery = gr.Gallery(
                    value=visible_examples,
                    label="Examples",
                    show_label=True,
                    columns=4,
                )

                selected = gr.State()
                batch_state = gr.State()

                preprocessed = gr.Gallery(
                    label="Preprocessed Images",
                    show_label=True,
                    columns=4,
                    height=256,
                )

                run_inference_btn = gr.Button(
                    "Run Inference", variant="primary"
                )

            # Right column: outputs
            with gr.Column(scale=4):
                output_gallery = gr.Gallery(
                    label="Predicted target views",
                    columns=4,
                    height=256,
                )
                with gr.Row():
                    preview_size = 360
                    with gr.Column(scale=3):
                        output_3d = gr.Model3D(
                            label="Gaussian point cloud",
                            height=preview_size,
                            interactive=False,
                            clear_color=[0.0, 0.0, 0.0, 0.0],
                            zoom_speed=0.5,
                            pan_speed=0.5,
                        )
                    with gr.Column(scale=2):
                        render_video = gr.Video(
                            label="Rendered sweep",
                            autoplay=False,
                            height=preview_size,
                        )
                artifacts = gr.File(label="Download outputs (zip)")
                log = gr.Textbox(label="Log", lines=8)

        # ---------- Event wiring ----------

        if EXAMPLES_FULL:
            examples_gallery.select(
                fn=get_select_index,
                inputs=None,
                outputs=[image_block, selected],
            ).then(
                fn=preprocess_fn,
                inputs=[image_block, selected],
                outputs=[preprocessed, batch_state],
            )

        image_block.upload(
            fn=preprocess_fn,
            inputs=[image_block],
            outputs=[preprocessed, batch_state],
        ).then(
            fn=info_fn,
            inputs=None,
            outputs=None,
        )

        run_inference_btn.click(
            fn=check_img_input,
            inputs=[batch_state],
        ).then(
            fn=run_inference_fn,
            inputs=[batch_state],
            outputs=[output_gallery, output_3d, render_video, artifacts, log],
        )

    return demo


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=DEFAULT_CONFIG)
    pre_args, _ = pre_parser.parse_known_args()
    default_ckpt, default_output_dir = _default_paths_for_config(pre_args.config)
    default_ckpt_help = (
        default_ckpt
        if _maybe_existing_checkpoint(default_ckpt)
        else DEFAULT_CKPT
    )

    parser = argparse.ArgumentParser(
        description="Launch the E-RayZer Gradio demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help="Default config path"
    )
    parser.add_argument(
        "--ckpt",
        default=default_ckpt,
        help=f"Checkpoint path; default downloads {default_ckpt_help} if missing",
    )
    parser.add_argument(
        "--device", default=None, help="Default device override"
    )
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help="Directory for outputs and demos",
    )
    parser.add_argument(
        "--share", action="store_true", help="Enable Gradio public sharing"
    )
    parser.add_argument(
        "--server-name", default="0.0.0.0", help="Host/IP to bind"
    )
    parser.add_argument(
        "--server-port", type=int, default=7860, help="Port to bind"
    )
    args = parser.parse_args()

    demo = build_demo(args)
    demo.queue().launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        theme=gr.themes.Ocean()
    )


if __name__ == "__main__":
    main()
