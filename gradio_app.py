from __future__ import annotations

import argparse
import functools
import os
import shutil
import time
from typing import Dict, List, Optional, Sequence, Tuple

import gradio as gr
import numpy as np
import torch
from PIL import Image

from app_core.engine import get_engine

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(THIS_DIR, "config", "erayzer.yaml")
DEFAULT_CKPT = os.path.join(THIS_DIR, "checkpoints", "erayzer_multi.pt")
DEFAULT_OUTPUT_ROOT = os.path.join(THIS_DIR, "outputs")
EXAMPLES_DIR = os.path.join(THIS_DIR, "examples")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")

EXAMPLES_LIST: List[str] = []
EXAMPLES_FULL: List[List[List[str]]] = []


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

    inferred_device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    defaults = {
        "config": args.config or DEFAULT_CONFIG,
        "ckpt": args.ckpt or DEFAULT_CKPT,
        "device": inferred_device,
        "output_dir": args.output_dir or DEFAULT_OUTPUT_ROOT,
    }

    preprocess_fn = functools.partial(preprocess, defaults["output_dir"])
    run_inference_fn = functools.partial(run_inference, defaults)

    _TITLE = "E-RayZer: Self-supervised 3D Reconstruction as Spatial Visual Pre-training"
    _DESCRIPTION = """
    <div>
    <a style="display:inline-block" href="https://qitaozhao.github.io/DiffusionSfM"><img src='https://img.shields.io/badge/public_website-8A2BE2'></a>
    <a style="display:inline-block; margin-left: .5em" href='https://github.com/QitaoZhao/DiffusionSfM'><img src='https://img.shields.io/github/stars/QitaoZhao/DiffusionSfM?style=social'/></a>
    </div>
    DiffusionSfM learns to predict scene geometry and camera poses as pixel-wise ray origins and endpoints using a denoising diffusion model.
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

                max_examples = 4
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
                    columns=max_examples,
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
    parser = argparse.ArgumentParser(
        description="Launch the E-RayZer Gradio demo"
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG, help="Default config path"
    )
    parser.add_argument(
        "--ckpt", default=DEFAULT_CKPT, help="Default checkpoint path"
    )
    parser.add_argument(
        "--device", default=None, help="Default device override"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_ROOT,
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