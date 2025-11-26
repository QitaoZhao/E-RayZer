
import torch
import torch.nn as nn
from easydict import EasyDict as edict

from .gaussians_renderer import (
    GaussianModel,
    render_opencv_cam,
    render_opencv_cam_gsplat,
    deferred_gaussian_render,
)


class Renderer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sh_degree = config.model.gaussians.sh_degree
        self.gaussians_model = GaussianModel(config.model.gaussians.sh_degree)

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        self,
        xyz,
        features,
        scaling,
        rotation,
        opacity,
        height,
        width,
        C2W,
        fxfycxcy,
        deferred=True,
    ):
        """
        xyz: [b, n_gaussians, 3]
        features: [b, n_gaussians, (sh_degree+1)^2, 3]
        scaling: [b, n_gaussians, 3]
        rotation: [b, n_gaussians, 4]
        opacity: [b, n_gaussians, 1]

        height: int
        width: int
        C2W: [b, v, 4, 4]
        fxfycxcy: [b, v, 4]

        output: [b, v, 3, height, width]
        """

        # if deferred:
        if self.config.model.get("use_deferred_rendering", False):
            renderings = deferred_gaussian_render(
                xyz, features, scaling, rotation, opacity, height, width, C2W, fxfycxcy
            )
            b, v = C2W.size(0), C2W.size(1)
            depth = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device
            )
            alpha = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device
            )
        else:
            b, v = C2W.size(0), C2W.size(1)
            renderings = torch.zeros(
                b, v, 3, height, width, dtype=torch.float32, device=xyz.device
            )

            depth = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device
            )
            alpha = torch.zeros(
                b, v, 1, height, width, dtype=torch.float32, device=xyz.device
            )

            for i in range(b):
                pc = self.gaussians_model.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i]
                )
                if self.config.model.get("use_gsplat", True): # False
                    near_plane = self.config.model.get("near_plane", 0.2)
                    buffers = render_opencv_cam_gsplat(
                        pc, height, width, C2W[i], fxfycxcy[i], self.sh_degree,
                        near_plane=near_plane
                    )
                    renderings[i] = buffers["render"]
                    if "depth" in buffers and buffers["depth"] is not None:
                        depth[i] = buffers["depth"]
                    if "alpha" in buffers and buffers["alpha"] is not None:
                        alpha[i] = buffers["alpha"]
                else:
                    for j in range(v):
                        # renderings[i, j] = render_opencv_cam(
                        #     pc, height, width, C2W[i, j], fxfycxcy[i, j]
                        # )["render"]
                        buffers = render_opencv_cam(
                            pc, height, width, C2W[i, j], fxfycxcy[i, j]
                        )
                        renderings[i, j] = buffers["render"]
                        if "depth" in buffers and buffers["depth"] is not None: 
                            depth[i, j] = buffers["depth"]
                        if "alpha" in buffers and buffers["alpha"] is not None:
                            alpha[i, j] = buffers["alpha"]

        # return renderings
        return edict(render=renderings, depth=depth, alpha=alpha)


def get_point_range_func(gaussians_config):
    range_setting = gaussians_config.get(
        "range_setting", edict({"type": "object_centric_depth"})
    )

    print("range_setting: ", range_setting)

    if range_setting.type == "object_centric_depth":
        rangefunc = lambda t : (2.0 * torch.sigmoid(t) - 1.0) * 1.5 + 2.7
        return rangefunc
    elif range_setting.type == "linear_depth":
        near = range_setting.get("near", 0.0)
        far = range_setting.get("far", 500.0)
        rangefunc = lambda t : torch.sigmoid(t) * (far - near) + near
        return rangefunc
    elif range_setting.type == "log_depth":
        near = range_setting.get("near", -6.2)
        far = range_setting.get("far", 6.2)
        # rangefunc = lambda t : torch.exp(torch.sigmoid(t) * (far - near) + near)
        rangefunc = lambda t : torch.exp(torch.sigmoid(t) * (far - near) + near)
        return rangefunc
    elif range_setting.type == "disparity":
        near = range_setting.get("near", 0.1)
        far = range_setting.get("far", 500.0)
        rangefunc = lambda t : 1.0 / (torch.sigmoid(t) * (1.0 / near - 1.0 / far) + 1.0 / far) 
        return rangefunc
    else:
        raise NotImplementedError

    return lambda t : t
