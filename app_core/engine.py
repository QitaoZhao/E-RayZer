from __future__ import annotations

import os
import shutil
import time
import trimesh
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import yaml
from easydict import EasyDict as edict
from PIL import Image
from scipy.spatial.transform import Rotation

import erayzer_core  # noqa: F401  # ensures vendored modules register themselves
import imageio.v2 as imageio
import numpy as np


@dataclass(frozen=True)
class EngineKey:
    config_path: str
    ckpt_path: str
    device: str


def _ensure_file(path: str, label: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def _load_config(path: str) -> edict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return edict(data)


def add_scene_cam(scene, c2w, edge_color, image=None, focal=None, imsize=None, screen_width=0.03):
    OPENGL = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    
    if image is not None:
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255*image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if focal is None:
        focal = min(H, W) * 1.1  # default value
    elif isinstance(focal, np.ndarray):
        focal = focal[0]

    # create fake camera
    height = focal * screen_width / H
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W/H
    transform = c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)

    # this is the camera mesh
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(4)).as_matrix()
    vertices = cam.vertices
    vertices_offset = 0.9 * cam.vertices
    vertices = np.r_[vertices, vertices_offset, geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b2, a2))
        faces.append((a2, c, c2))
        faces.append((c2, b2, b))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    for i,face in enumerate(cam.faces):
        if 0 in face:
            continue

        if i == 1 or i == 5:
            a, b, c = face
            faces.append((a, b, c))

    vertices[:, [1, 2]] *= -1
    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color
    
    scene.add_geometry(cam)


def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d+1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim-2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1]+1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


class ERayZerEngine:
    """Thin wrapper around the E-RayZer model for single-scene inference."""

    def __init__(self, config_path: str, ckpt_path: str, device: str, output_root: str) -> None:
        _ensure_file(config_path, "config")
        _ensure_file(ckpt_path, "checkpoint")
        os.makedirs(output_root, exist_ok=True)

        self.output_root = output_root
        self.device_name = device or "auto"
        self.device = torch.device(self.device_name if self.device_name != "auto" else self._default_device())
        self.config = _load_config(config_path)
        self.ckpt_path = ckpt_path
        self._prepare_config()
        self.model = self._load_model()
        self.model.eval()

        training = self.config.training
        tokenizer = self.config.model.image_tokenizer
        self.image_size = int(tokenizer.image_size)
        self.num_views = int(training.num_views)
        self.num_input_views = int(training.num_input_views)
        self.num_target_views = int(training.num_target_views)
        def _central_crop(img: Image.Image) -> Image.Image:
            shorter_side = min(img.size)
            return TF.center_crop(img, shorter_side)

        self.transform = T.Compose(
            [
                T.Lambda(_central_crop),
                T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.ToTensor(),
            ]
        )
        amp_dtype = str(training.get("amp_dtype", "fp16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        self.amp_enabled = bool(training.get("use_amp", True)) and self.device.type == "cuda"

    def _prepare_config(self) -> None:
        cfg = self.config
        cfg.inference = True
        cfg.evaluation = False
        cfg.create_visual = True

        training = cfg.training
        training.batch_size_per_gpu = 1
        training.num_workers = 0
        training.prefetch_factor = training.get("prefetch_factor", 2)
        training.random_inputs = False
        training.random_shuffle = False
        training.force_resume_ckpt = True
        training.resume_ckpt = self.ckpt_path
        training.view_selector = edict(training.get("view_selector", {}))
        training.view_selector.type = training.view_selector.get("type", "even_I_B")
        training.view_selector.use_curriculum = False

        cfg.inference_view_selector_type = cfg.get("inference_view_selector_type", training.view_selector.type)

    def _load_model(self) -> torch.nn.Module:
        module_name, class_name = self.config.model.class_name.rsplit(".", 1)
        ModelClass = __import__(module_name, fromlist=[class_name]).__dict__[class_name]
        model = ModelClass(self.config).to(self.device)
        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        state_dict = checkpoint.get("model", checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"[ERayZerEngine] Missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            print(f"[ERayZerEngine] Unexpected keys: {len(incompatible.unexpected_keys)}")
        print("[ERayZerEngine] Model loaded successfully.")
        return model

    @staticmethod
    def _default_device() -> str:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        array = tensor.permute(1, 2, 0).cpu().numpy()
        array = (array * 255.0).round().astype("uint8")
        return Image.fromarray(array)

    def _prepare_batch(self, image_files: Sequence[str]) -> Dict[str, torch.Tensor]:
        if len(image_files) != self.num_views:
            print(f"Warning: expected {self.num_views} views, but got {len(image_files)}; padding inputs to {self.num_views} views.")

        tensors: List[torch.Tensor] = []
        for path in sorted(image_files, key=os.path.basename):
            img = Image.open(path).convert("RGB")
            tensors.append(self.transform(img))
        images = torch.stack(tensors, dim=0).unsqueeze(0)
        intrinsics = torch.tensor(
            [[[1.0, 1.0, 0.5, 0.5]] * self.num_views], dtype=torch.float32
        )
        return {"image": images, "fxfycxcy": intrinsics}

    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: value.to(self.device, non_blocking=self.device.type == "cuda") if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def run(
        self, image_files: Sequence[str]
    ) -> Tuple[List[str], str, str, Optional[str], Optional[str]]:
        batch = self._prepare_batch(image_files)
        batch_gpu = self._move_to_device(batch)
        autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with torch.no_grad():
            with autocast_ctx:
                result = self.model(batch_gpu)

        run_dir, glb_path, video_path = self._export_outputs(result)
        gallery_paths = sorted(
            [os.path.join(run_dir, name) for name in os.listdir(run_dir) if name.startswith("pred_view_")]
        )
        archive = shutil.make_archive(run_dir, "zip", run_dir)
        log = (
            f"Saved {len(gallery_paths)} predicted views and Gaussian assets to {run_dir}.\n"
            f"Archive: {archive}"
        )
        return gallery_paths, archive, log, glb_path, video_path

    def _export_outputs(self, result) -> Tuple[str, Optional[str], Optional[str]]:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(self.output_root, timestamp)
        os.makedirs(run_dir, exist_ok=True)

        glb_path: Optional[str] = None
        video_path: Optional[str] = None

        if getattr(result, "render") is not None:
            render_tensor = result.render.detach().cpu().clamp(0, 1)
            for idx, frame in enumerate(render_tensor[0]):
                img = self._tensor_to_pil(frame)
                img.save(os.path.join(run_dir, f"pred_view_{idx:02d}.png"))

        if hasattr(result, "pixelalign_xyz") is not None:
            glb_path = os.path.join(run_dir, "point_cloud.glb")

            scene = trimesh.Scene()
            xyzs = result.pixelalign_xyz[0].detach().cpu().permute(0, 2, 3, 1).reshape(-1, 3).numpy()
            xyzs[:, [1, 2]] *= -1
            rgbs = (result.image[0].detach().cpu().permute(0, 2, 3, 1).reshape(-1, 3) * 255.0).round().numpy().astype(np.uint8)
            point_cloud = trimesh.points.PointCloud(vertices=xyzs, colors=rgbs)
            scene.add_geometry(point_cloud)

            c2ws = result.c2w[0].detach().cpu().numpy()
            num_images = c2ws.shape[0]
            cmap = plt.get_cmap("hsv")
            for i, c2w in enumerate(c2ws):
                color_rgb = (np.array(cmap(i / num_images))[:3] * 255).astype(int)
                add_scene_cam(
                    scene=scene,
                    c2w=c2w,
                    edge_color=color_rgb,
                    image=None,
                    focal=None,
                    imsize=(256, 256),
                    screen_width=0.1
                )

            scene.export(glb_path)

        if getattr(result, "render_video") is not None:
            frames_dir = os.path.join(run_dir, "render_video_frames")
            os.makedirs(frames_dir, exist_ok=True)
            frames = result.render_video[0].detach().cpu().clamp(0, 1)
            for idx, frame in enumerate(frames):
                img = self._tensor_to_pil(frame)
                img.save(os.path.join(frames_dir, f"frame_{idx:03d}.png"))

            frames_np = (frames.permute(0, 2, 3, 1).numpy() * 255.0).round().astype(np.uint8)
            video_path = os.path.join(run_dir, "render_video.mp4")
            imageio.mimwrite(video_path, frames_np, fps=24)

        return run_dir, glb_path, video_path


_ENGINE_CACHE: Dict[EngineKey, ERayZerEngine] = {}


def get_engine(config_path: str, ckpt_path: str, device: str, output_root: str) -> ERayZerEngine:
    key = EngineKey(config_path, ckpt_path, device or "auto")
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = ERayZerEngine(config_path, ckpt_path, device, output_root)
    return _ENGINE_CACHE[key]
