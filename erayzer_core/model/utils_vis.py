import os
import math
import torch
import imageio
import trimesh
import pytorch3d

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from einops import rearrange
from typing import Optional, Sequence, Tuple, List, Union
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.renderer import (
    DirectionalLights,
    FoVPerspectiveCameras,
    HardPhongShader,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    TexturesUV,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer.cameras import FoVPerspectiveCameras, PerspectiveCameras, CamerasBase


def cylinder_between_points(
    point1, point2, radius=0.01, color=(0, 0, 0), device="cuda"
):
    height = np.linalg.norm(point1 - point2)
    cylinder = trimesh.creation.cylinder(radius, height)
    cylinder_vs = np.array(cylinder.vertices)
    cylinder_vs[:, 2] += height / 2
    trans = transform_ray(np.array([0, 0, 1]), point2 - point1)
    cylinder_vs = cylinder_vs @ trans.T
    cylinder_vs = cylinder_vs + point1
    m = Meshes(
        torch.tensor(cylinder_vs).float().unsqueeze(0),
        torch.tensor(np.array(cylinder.faces)).long().unsqueeze(0),
    )
    m = constant_uv_texture(m, color=torch.Tensor(color)).to(device)
    return m


def get_wireframe(s=1, f=2, radius=0.03, color=(1, 0, 0), border=False, device="cuda"):
    meshes = []
    for ray_new in [[-s, -s, f], [-s, s, f], [s, s, f], [s, -s, f]]:
        height = np.sqrt(2 * s**2 + f**2)
        cylinder = trimesh.creation.cylinder(radius, height)
        cylinder_vs = np.array(cylinder.vertices)
        cylinder_vs[:, 2] += height / 2
        T = transform_ray([0, 0, s], ray_new)
        cylinder_vs = cylinder_vs @ T
        mesh = Meshes(
            torch.tensor(cylinder_vs).float().unsqueeze(0),
            torch.tensor(np.array(cylinder.faces)).long().unsqueeze(0),
        )
        meshes.append(constant_uv_texture(mesh, color=torch.Tensor(color)))
    if border:
        for ray_new in [[0, s, 0], [s, 0, 0], [0, -s, 0], [-s, 0, 0]]:
            height = 2 * s
            cylinder = trimesh.creation.cylinder(radius, height)
            cylinder_vs = np.array(cylinder.vertices)
            cylinder_vs[:, 2] += height / 2
            T = transform_ray([0, 0, s], ray_new)
            cylinder_vs = cylinder_vs @ T
            cylinder_vs[:, 2] += f
            cylinder_vs[:, :2] += sum(ray_new)
            mesh = Meshes(
                torch.tensor(cylinder_vs).float().unsqueeze(0),
                torch.tensor(np.array(cylinder.faces)).long().unsqueeze(0),
            )
            meshes.append(constant_uv_texture(mesh, color=torch.Tensor(color)))

    return pytorch3d.structures.join_meshes_as_scene(meshes).to(device)


def transform_ray(ray_og, ray_new):
    ray_og = np.array(ray_og)
    ray_og = ray_og / np.linalg.norm(ray_og)
    ray_new = np.array(ray_new)
    ray_new = ray_new / np.linalg.norm(ray_new)
    rotation_magnitude = np.arccos(np.dot(ray_og, ray_new))
    rotation_axis = np.cross(ray_og, ray_new)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    axis_angle = torch.from_numpy(rotation_axis * rotation_magnitude)
    rot_mat = pytorch3d.transforms.axis_angle_to_matrix(axis_angle).float().numpy()
    return rot_mat


def antialias(image, level=1):
    is_numpy = isinstance(image, np.ndarray)
    if is_numpy:
        image = Image.fromarray(image)
    for _ in range(level):
        size = np.array(image.size) // 2
        image = image.resize(size, Image.LANCZOS)
    #     if is_numpy:
    #         image = np.array(image)
    return image


def get_renderer(image_size=2048, device="cuda"):
    cameras = FoVPerspectiveCameras(znear=0.01, fov=10.0, device=device)
    # Note: bin_size=0 is very slow, but needed for rendering so many rays
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
    lights = DirectionalLights(
        device=device,
        ambient_color=((0.8, 0.8, 0.8),),
        diffuse_color=((0.2, 0.2, 0.2),),
        specular_color=((0.2, 0.2, 0.2),),
        direction=((0, 1, 0),),
    )

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    renderer_mesh = MeshRenderer(
        rasterizer=rasterizer,
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights),
    )
    return renderer_mesh


def constant_uv_texture(mesh, color=None):
    faces = mesh.faces_padded()
    device = faces.device
    N = faces.shape[0]
    if color is None:
        color = torch.Tensor([0.5, 0.5, 0.8]).to(device)
    maps = torch.ones((N, 10, 10, 3))
    maps[..., :] = color
    maps = maps.to(device)

    tex_faces = faces * 1
    tex_faces[..., 0] = 0
    tex_faces[..., 1] = 1
    tex_faces[..., 2] = 2

    tex_verts = torch.ones((N, 3, 2)).to(device)
    tex_verts[:, 0, 0] = 0.25
    tex_verts[:, 0, 1] = 0.25
    tex_verts[:, 1, 0] = 0.75
    tex_verts[:, 1, 1] = 0.25
    tex_verts[:, 2, 0] = 0.5
    tex_verts[:, 2, 1] = 0.75

    textures = TexturesUV(maps, tex_faces, tex_verts).to(device)
    mesh = pytorch3d.structures.Meshes(
        verts=mesh.verts_padded(), faces=mesh.faces_padded(), textures=textures
    )
    return mesh


def grid_mesh_from_view(cameras, image, s=1, f=2, rotate=True, device="cuda"):
    view_to_world = cameras.get_world_to_view_transform().inverse()

    grid_mesh_verts = (
        torch.Tensor([[-s, -s, f], [s, -s, f], [s, s, f], [-s, s, f]])
        .float()
        .to(device)
    )
    grid_mesh_faces = torch.Tensor([[0, 1, 3], [1, 2, 3]]).long().to(device)

    if rotate:
        image = np.rot90(image, 2).copy()

    grid_texture_map = torch.from_numpy(image).float().to(device)
    uv_verts = (
        torch.Tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]]).float().to(device)
    ) * 0.5
    uv_verts[..., 0] *= -1
    uv_verts = uv_verts + 0.5
    grid_texture = TexturesUV([grid_texture_map], [grid_mesh_faces], [uv_verts]).to(
        device
    )

    grid_mesh = Meshes(
        [view_to_world.transform_points(grid_mesh_verts)],
        [grid_mesh_faces],
        textures=grid_texture,
    )

    return grid_mesh


def transform_mesh(mesh, transform, scale=1.0):
    mesh = mesh.clone()
    verts = mesh.verts_packed() * scale
    verts = transform.transform_points(verts)
    mesh.offset_verts_(verts - mesh.verts_packed())
    return mesh


@torch.no_grad()
def _step_back_from_cameras(
    cams: CamerasBase, step_back_distance: float, device: torch.device
) -> CamerasBase:
    """
    Returns a camera set where each camera is translated by -Z (local) by `step_back_distance`.
    Camera intrinsics and rotations are preserved.
    """
    cams = cams.to(device)
    N = len(cams)

    # world->view then inverse to get cam->world transform
    w2v = cams.get_world_to_view_transform()
    c2w = w2v.inverse()

    # local point [0,0,-d] moved into world space → new camera center C'
    p_local = torch.tensor([0.0, 0.0, -float(step_back_distance)], device=device)[None, None, :].expand(N, 1, 3)
    C_prime = c2w.transform_points(p_local).squeeze(1)  # (N,3)

    R = cams.R.clone()                                  # (N,3,3)
    # In PyTorch3D: C = -R^T T => T = -(C @ R)
    T_prime = -(C_prime.unsqueeze(1) @ R).squeeze(1)    # (N,3)

    if isinstance(cams, FoVPerspectiveCameras):
        return FoVPerspectiveCameras(
            R=R, T=T_prime, fov=cams.fov, znear=cams.znear, zfar=cams.zfar, device=device
        )
    elif isinstance(cams, PerspectiveCameras):
        return PerspectiveCameras(
            R=R, T=T_prime,
            focal_length=cams.focal_length,
            principal_point=cams.principal_point,
            in_ndc=cams.in_ndc,
            image_size=getattr(cams, "image_size", None),
            device=device,
        )
    else:
        # Best-effort generic recreation for other subclasses
        kwargs = {}
        for k, v in cams.__dict__.items():
            if isinstance(v, torch.Tensor) and v.shape[:1] == (N,) and k not in ("R", "T"):
                kwargs[k] = v
        return type(cams)(R=R, T=T_prime, device=device, **kwargs)


@torch.no_grad()
def render_wireframes_from_stepback(
    render_cameras: CamerasBase,       # cameras we step back from (viewpoints)
    src_cameras: CamerasBase,          # cameras we DRAW as wireframes/grids
    images: Optional[torch.Tensor],    # (V,3,H,W) in [0,1] or None (textures near planes)
    image_size: int,                   # square render size for get_renderer
    step_back_distance: float = 3.0,
    near_scale: float = 0.10,
    far_scale: float = 0.20,
    line_radius_scale: float = 0.005,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Renders src_cameras as frustum wireframes from 'step-back' versions of render_cameras.

    Returns:
        rgba: (V, 4, H, W) float tensor in [0,1] with alpha in channel 3.
    """
    device = torch.device(device)
    render_cameras = render_cameras.to(device)
    src_cameras = src_cameras.to(device)
    V = len(render_cameras)
    assert len(src_cameras) >= 1, "Need at least one source camera to draw."

    # ---- scale scene using src_cameras spread ----
    centers = src_cameras.get_camera_center()                       # (V_src,3)
    dist_max = torch.norm(centers, dim=1).amax().item() / math.sqrt(2)
    dist_max = max(dist_max, 1.0)
    s = near_scale * dist_max
    f = far_scale * dist_max
    line_radius = line_radius_scale * dist_max

    # ---- build scene of src_cameras (grid + wireframe per camera) ----
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("hsv")

    meshes = []
    Vsrc = len(src_cameras)
    for i in range(Vsrc):
        cam_i = src_cameras[i]
        if images is not None:
            img_i = images[i].clamp(0, 1) if i < images.shape[0] else torch.ones(3, 1, 1, device=device)
        else:
            img_i = torch.ones(3, 1, 1, device=device)  # white texture

        grid_mesh = grid_mesh_from_view(cam_i, img_i.cpu().numpy(), s=s, f=f, rotate=False)

        wireframe = get_wireframe(
            s=s, f=f, radius=line_radius, color=cmap(i / max(1, Vsrc))[:3], border=True
        )
        view_to_world = cam_i.get_world_to_view_transform().inverse()
        mesh_wf = transform_mesh(wireframe, view_to_world)

        meshes.append(grid_mesh)
        meshes.append(mesh_wf)

    scene = join_meshes_as_scene(meshes).to(device)

    # ---- step-back views from render_cameras ----
    stepback = _step_back_from_cameras(render_cameras, step_back_distance, device=device)

    # ---- render RGBA for each step-back camera ----
    renderer = get_renderer(image_size=image_size, device=device)    # returns RGBA (…,4)
    imgs = []
    for i in range(V):
        img = renderer(scene, cameras=stepback[i])                   # (1,H,W,4), float in [0,1]
        img = img[0].permute(2, 0, 1).contiguous()                   # (4,H,W)
        imgs.append(img)
    rgba = torch.stack(imgs, dim=0)                                  # (V,4,H,W)
    return rgba


@torch.no_grad()
def build_stepback_c2ws(frame_c2ws: torch.Tensor, step_back_distance: float) -> torch.Tensor:
    """
    frame_c2ws: (..., 4, 4) camera-to-world (OpenCV-style) transforms
    step_back_distance: scalar distance to move along each camera's local -Z axis
    returns: stepback_c2ws with same shape as frame_c2ws
    """
    # Extract rotation (R) and translation (t) from c2w
    R = frame_c2ws[..., :3, :3]                  # (..., 3, 3)
    t = frame_c2ws[..., :3,  3]                  # (..., 3)

    # Local camera +Z is the 3rd column of R in world coords; step-back is along -Z
    z_world = R[..., :, 2]                       # (..., 3)
    t_new = t - step_back_distance * z_world     # move camera center backward

    c2w_step = frame_c2ws.clone()
    c2w_step[..., :3, 3] = t_new
    return c2w_step


def tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    tensor_cpu = tensor.detach().cpu().to(torch.float32)
    if tensor_cpu.numel() == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    if tensor_cpu.min().item() < 0.0:
        tensor_cpu = (tensor_cpu + 1.0) * 0.5
    tensor_cpu = tensor_cpu.clamp(0.0, 1.0)

    array = tensor_cpu.numpy()
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    elif array.shape[0] == 2:
        array = np.concatenate([array, array[:1]], axis=0)
    array = np.transpose(array, (1, 2, 0))
    return (array * 255.0).round().astype(np.uint8)


def compute_top_pca(features_np: np.ndarray, num_components: int = 3) -> np.ndarray:
    if features_np.size == 0:
        return np.zeros((0, num_components), dtype=np.float32)

    features_2d = features_np.reshape(-1, features_np.shape[-1])
    effective_components = min(num_components, features_2d.shape[1])
    if effective_components <= 0:
        return np.zeros((features_2d.shape[0], num_components), dtype=np.float32)

    comps: Optional[np.ndarray] = None
    try:
        from sklearn.decomposition import PCA as SkPCA  # type: ignore

        pca = SkPCA(n_components=effective_components)
        comps = pca.fit_transform(features_2d)
    except Exception:
        centered = features_2d - features_2d.mean(axis=0, keepdims=True)
        try:
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            vt = vt[:effective_components]
            comps = centered @ vt.T if vt.size else np.zeros((features_2d.shape[0], effective_components), dtype=np.float32)
        except np.linalg.LinAlgError:
            comps = np.zeros((features_2d.shape[0], effective_components), dtype=np.float32)

    comps = np.asarray(comps, dtype=np.float32)
    if comps.shape[1] < num_components:
        comps = np.pad(comps, ((0, 0), (0, num_components - comps.shape[1])), mode="constant")
    elif comps.shape[1] > num_components:
        comps = comps[:, :num_components]

    return comps


def normalize_grid_to_uint8(grid: np.ndarray, min_val: Optional[float] = None, max_val: Optional[float] = None) -> np.ndarray:
    if grid.size == 0:
        return np.zeros_like(grid, dtype=np.uint8)

    if min_val is None or max_val is None:
        min_val = float(grid.min())
        max_val = float(grid.max())

    if max_val - min_val < 1e-6:
        return np.zeros_like(grid, dtype=np.uint8)

    scaled = (grid - min_val) / (max_val - min_val)
    return (scaled * 255.0).round().astype(np.uint8)


def plot_pca_components(
    all_components: np.ndarray,
    image_list: List[np.ndarray],
    base_dir: str,
    prefix: str,
    token_h: int,
    token_w: int,
) -> None:
    if all_components.size == 0 or not image_list:
        return

    os.makedirs(base_dir, exist_ok=True)
    aggregated_targets: List[np.ndarray] = []
    aggregated_pca_rgb: List[np.ndarray] = []

    global_min = all_components.min(axis=0)
    global_max = all_components.max(axis=0)

    for view_idx, target_image in enumerate(image_list):
        view_dir = os.path.join(base_dir, f"{prefix}_view_{view_idx:03d}")
        os.makedirs(view_dir, exist_ok=True)

        target_h, target_w = target_image.shape[0], target_image.shape[1]
        start = view_idx * token_h * token_w
        end = start + token_h * token_w
        if end > all_components.shape[0]:
            break
        comp_slice = all_components[start:end]

        component_arrays = []
        for comp_idx in range(3):
            component_grid = comp_slice[:, comp_idx].reshape(token_h, token_w)
            comp_min = float(global_min[min(comp_idx, global_min.shape[0] - 1)])
            comp_max = float(global_max[min(comp_idx, global_max.shape[0] - 1)])
            component_uint8 = normalize_grid_to_uint8(component_grid, comp_min, comp_max)
            comp_image = Image.fromarray(component_uint8, mode="L").resize((target_w, target_h), resample=Image.NEAREST)
            comp_image.save(os.path.join(view_dir, f"{prefix}_pc{comp_idx + 1}.png"))
            component_arrays.append(np.array(comp_image))

        while len(component_arrays) < 3:
            component_arrays.append(np.zeros((target_h, target_w), dtype=np.uint8))

        target_rgb = target_image.astype(np.uint8)
        if target_rgb.ndim == 2:
            target_rgb = np.repeat(target_rgb[..., None], 3, axis=2)
        elif target_rgb.shape[2] == 1:
            target_rgb = np.repeat(target_rgb, 3, axis=2)

        pca_rgb = np.stack(component_arrays[:3], axis=-1).astype(np.uint8)
        pca_rgb_image = Image.fromarray(pca_rgb, mode="RGB")
        pca_rgb_image.save(os.path.join(view_dir, f"{prefix}_pca_rgb.png"))

        overview = np.concatenate([target_rgb, np.array(pca_rgb_image)], axis=1)
        Image.fromarray(overview, mode="RGB").save(os.path.join(view_dir, f"{prefix}_overview.png"))

        aggregated_targets.append(target_rgb)
        aggregated_pca_rgb.append(np.array(pca_rgb_image))

    if aggregated_targets and aggregated_pca_rgb:
        try:
            target_row = np.concatenate(aggregated_targets, axis=1)
            pca_row = np.concatenate(aggregated_pca_rgb, axis=1)
            combined_overview = np.concatenate([target_row, pca_row], axis=0)
            Image.fromarray(combined_overview.astype(np.uint8), mode="RGB").save(
                os.path.join(base_dir, f"{prefix}_overview_grid.png")
            )
        except ValueError:
            pass


def save_pca_set(
    features_tensor: Optional[Union[torch.Tensor, np.ndarray]],
    images_tensor: Optional[Union[torch.Tensor, Sequence[np.ndarray]]],
    base_dir: str,
    prefix: str,
    token_hw_hint: Optional[Tuple[int, int]] = None,
) -> None:
    if features_tensor is None or images_tensor is None:
        return

    if torch.is_tensor(features_tensor):
        features_np = features_tensor.detach().to(torch.float32).cpu().numpy()
    else:
        features_np = np.asarray(features_tensor, dtype=np.float32)

    if features_np.size == 0:
        return

    if features_np.ndim != 3:
        raise ValueError("features_tensor must be shaped as (num_views, num_tokens, dim)")

    num_views, num_tokens, _ = features_np.shape

    token_h = token_w = None
    if token_hw_hint is not None:
        expected_tokens = token_hw_hint[0] * token_hw_hint[1]
        if expected_tokens == num_tokens:
            token_h, token_w = token_hw_hint

    if token_h is None:
        token_h = int(np.sqrt(num_tokens))
        token_h = max(token_h, 1)
        while token_h > 1 and num_tokens % token_h != 0:
            token_h -= 1
        token_w = num_tokens // token_h if token_h > 0 else num_tokens
    if token_w is None or token_w == 0:
        token_w = num_tokens

    if torch.is_tensor(images_tensor):
        iterable = images_tensor.detach().cpu().to(torch.float32)
        if iterable.ndim == 3:
            iterable = iterable.unsqueeze(0)
        image_list = [tensor_to_uint8_image(img) for img in iterable]
    else:
        image_list = []
        for img in images_tensor:
            if isinstance(img, np.ndarray):
                arr = img
            else:
                arr = np.asarray(img)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            image_list.append(arr.astype(np.uint8))

    view_count = min(len(image_list), num_views)
    if view_count == 0:
        return

    features_np = features_np[:view_count]
    image_list = image_list[:view_count]

    flattened = features_np.reshape(view_count * num_tokens, features_np.shape[-1])
    components = compute_top_pca(flattened)
    if components.shape[0] == 0:
        return

    plot_pca_components(components, image_list, base_dir, prefix, token_h, token_w)


def _save_tensor_png(tensor, path):
    tensor_cpu = tensor.detach().cpu().to(torch.float32)
    if tensor_cpu.min().item() < 0.0:
        tensor_cpu = (tensor_cpu + 1.0) * 0.5
    tensor_cpu = tensor_cpu.clamp(0.0, 1.0)
    array = rearrange(
        tensor_cpu,
        "c h w -> h w c",
    )
    array = (array.numpy() * 255.0).clip(0.0, 255.0).astype(np.uint8)
    Image.fromarray(array).save(path, format="PNG", compress_level=0)


def view_color_coded_images_from_tensor(images, save_path=None):
    num_frames = images.shape[0]
    cmap = plt.get_cmap("hsv")

    num_rows = num_frames // 3 + 1
    num_cols = 3
    figsize = (num_cols * 2, num_rows * 2)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()

    for i in range(num_rows * num_cols):
        if i < num_frames:
            # Prepare image
            if images[i].shape[0] == 3:
                image = images[i].permute(1, 2, 0)
            else:
                image = images[i].unsqueeze(-1)

            axs[i].imshow(image)

            # Color-coded frames
            for side in ["bottom", "top", "left", "right"]:
                axs[i].spines[side].set_color(cmap(i / num_frames))
                axs[i].spines[side].set_linewidth(5)

            axs[i].set_xticks([])
            axs[i].set_yticks([])
        else:
            axs[i].axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)