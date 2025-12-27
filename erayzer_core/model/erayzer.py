import copy

import torch
import torch.nn as nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

from .utils_transformer import (
    QK_Norm_TransformerBlock,
    _init_weights,
    _init_weights_layerwise
)
from utils import camera_utils
from .utils_pe import get_2d_sincos_pos_embed
from .utils_rot import rot6d2mat, quat2mat
from .utils_gaussian import get_point_range_func, Renderer
from .utils_vis import build_stepback_c2ws


def build_transformer_blocks(
    num_layers: int,
    d: int,
    d_head: int,
    use_qk_norm: bool,
    special_init: bool = False,
    depth_init: bool = False
) -> nn.ModuleList:
    """
    Initializes a list of QK_Norm_TransformerBlock layers with optional special initialization.

    Args:
        num_layers (int): Number of layers.
        d (int): Embedding dimension.
        d_head (int): Head dimension.
        use_qk_norm (bool): Whether to use qk normalization.
        special_init (bool): If True, use special weight initialization.
        depth_init (bool): If True, use layer-depth-aware std deviation.

    Returns:
        nn.ModuleList: Initialized transformer layers.
    """
    layers = [
        QK_Norm_TransformerBlock(d, d_head, use_qk_norm=use_qk_norm)
        for _ in range(num_layers)
    ]

    if special_init:
        for idx, layer in enumerate(layers):
            if depth_init:
                std = 0.02 / (2 * (idx + 1)) ** 0.5
            else:
                std = 0.02 / (2 * num_layers) ** 0.5
            layer.apply(lambda module: _init_weights_layerwise(module, std))

    return nn.ModuleList(layers)


class GaussiansUpsampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # scale and opacity initialization
        self.scaling_bias = self.config.model.get("scaling_bias", -2.3)
        self.scaling_max = self.config.model.get("scaling_max", -1.2)
        self.opacity_bias = self.config.model.get("opacity_bias", -2.0)

        """
        xyz : torch.tensor of shape (n_gaussians, 3)
        features : torch.tensor of shape (n_gaussians, (sh_degree + 1) ** 2, 3)
        scaling : torch.tensor of shape (n_gaussians, 3)
        rotation : torch.tensor of shape (n_gaussians, 4)
        opacity : torch.tensor of shape (n_gaussians, 1)
        """

    def to_gs(self, gaussians):
        """
        gaussians: [b, n_gaussians, d]
        """
        xyz, features, scaling, rotation, opacity = gaussians.split(
            [3, (self.config.model.gaussians.sh_degree + 1) ** 2 * 3, 3, 4, 1], dim=2
        )

        if not self.config.model.hard_pixelalign:
            xyz = xyz.clamp(-500.0, 500.0)

        features = features.reshape(
            features.size(0),
            features.size(1),
            (self.config.model.gaussians.sh_degree + 1) ** 2,
            3,
        )

        scaling = (scaling + self.scaling_bias).clamp(max=self.scaling_max).clamp(min=-10.0)
        opacity = (opacity + self.opacity_bias).clamp(min=-10.0)

        return xyz, features, scaling, rotation, opacity

    def forward(self, gaussians, images):
        """
        triplane: [b, n_gaussians, d]
        images: [b, l, d]
        output: [b, n_gaussians, dd]
        """
        u = self.config.model.gaussians.upsampler.upsample_factor
        if u > 1:
            raise NotImplementedError("GaussiansUpsampler only supports u=1")

        gaussians = self.linear(self.layernorm(gaussians))

        return gaussians


class ERayZer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.d = self.config.model.transformer.d
        self.d_head = self.config.model.transformer.d_head
        self.hh = self.ww = self.config.model.image_tokenizer.image_size // self.config.model.image_tokenizer.patch_size
        self.ph = self.pw = self.config.model.image_tokenizer.patch_size
        self.inference = True
 
        # image tokenizer
        self.image_tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=self.ph,
                pw=self.pw,
            ),
            nn.Linear(
                self.config.model.image_tokenizer.in_channels
                * (self.ph * self.pw),
                self.d,
                bias=False,
            ),
        )
        self.image_tokenizer.apply(_init_weights)

        # image positional embedding embedder
        self.use_pe_embedding_layer = self.config.model.get('input_with_pe', True)
        if self.use_pe_embedding_layer:
            self.pe_embedder = (
                nn.Sequential(
                    nn.Linear(
                        self.d,
                        self.d,
                    ),
                    nn.SiLU(),
                    nn.Linear(
                        self.d,
                        self.d,
                    ),
                )
                if self.use_pe_embedding_layer
                else nn.Identity()
            )
            self.pe_embedder.apply(_init_weights)

            self.pe_embedder_plucker = (
                nn.Sequential(
                    nn.Linear(
                        self.d,
                        self.d,
                    ),
                    nn.SiLU(),
                    nn.Linear(
                        self.d,
                        self.d,
                    ),
                )
                if self.use_pe_embedding_layer
                else nn.Identity()
            )
            self.pe_embedder_plucker.apply(_init_weights)

        # Reference: VGGT (https://github.com/facebookresearch/vggt/blob/main/vggt/models/aggregator.py)
        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.num_register_tokens = 4
        self.camera_token = nn.Parameter(torch.randn(1, 1, self.d))
        self.register_token = nn.Parameter(torch.randn(1, self.num_register_tokens, self.d))

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # qk norm settings
        use_qk_norm = self.config.model.transformer.get("use_qk_norm", False)

        # transformer encoder and init
        self.transformer_encoder = build_transformer_blocks(
            num_layers=config.model.transformer.encoder_n_layer,
            d=self.d,
            d_head=self.d_head,
            use_qk_norm=use_qk_norm,
            special_init=config.model.transformer.get("special_init", False),
            depth_init=config.model.transformer.get("depth_init", False),
        )

        # transformer encoder2 and init
        self.transformer_encoder_geom = build_transformer_blocks(
            num_layers=config.model.transformer.encoder_geom_n_layer,
            d=self.d,
            d_head=self.d_head,
            use_qk_norm=use_qk_norm,
            special_init=config.model.transformer.get("special_init", False),
            depth_init=config.model.transformer.get("depth_init", False),
        )

        # pose predictor
        self.pose_predictor = PoseEstimator(self.config)
        
        # input pose tokenizer
        self.input_pose_tokenizer = nn.Sequential(
            Rearrange(
                "b v (hh ph) (ww pw) c -> (b v) (hh ww) (ph pw c)",
                ph=self.ph,
                pw=self.pw,
            ),
            nn.Linear(
                6
                * (self.ph * self.pw),
                self.d,
                bias=False,
            ),
        )
        self.input_pose_tokenizer.apply(_init_weights)

        # fuse mlp
        self.mlp_fuse = nn.Sequential(
            nn.LayerNorm(self.d*2, bias=False),
            nn.Linear(
                self.d*2,
                self.d,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                self.d,
                self.d,
                bias=True,
            ),
        )
        self.mlp_fuse.apply(_init_weights)

        # 3D gaussian decoder
        self.gs_per_pixel_sqrt = self.config.model.gs_per_pixel_sqrt
        self.image_token_decoder = nn.Sequential(
            nn.LayerNorm(self.d, bias=False),
            nn.Linear(
                self.d,
                (self.ph * self.pw) * (
                    3 + (self.config.model.gaussians.sh_degree + 1) ** 2 * 3 + 3 + 4 + 1
                ),
                bias=False,
            ),
        )
        self.image_token_decoder.apply(_init_weights)

        self.upsampler = GaussiansUpsampler(self.config)
        self.range_func = get_point_range_func(self.config.model.gaussians)
        self.renderer = Renderer(self.config)

        # config backup
        self.config_bk = copy.deepcopy(self.config)
        self.render_interpolate = config.training.get("render_interpolate", False)

        if config.model.transformer.get('fix_decoder', False):
            self.freeze_weights()

        # training settings
        if config.inference or config.get("evaluation", False):
            if config.training.get('random_inputs', False):
                self.random_index = True
            else:
                self.random_index = False
        else:
            self.random_index = config.training.get('random_split', False)
        
        print('Use random index:', self.random_index)

    def forward(self, data):
        # input, target, input_idx, target_idx = self.split_data(data, random_index=self.random_index)
        image_all = data['image'] * 2.0 - 1.0                                     # [b, v_all, c, h, w], range (0,1) to (-1,1)
        b, v, c, h, w = image_all.shape
        device = image_all.device

        # Padding input to 10 views if less than 10 views (repeat the last view)
        if v < 10:
            pad_input = True
            v_all = 10
            pad_views = 10 - v
            last_view = image_all[:, -1:, ...].repeat(1, pad_views, 1, 1, 1)
            image_all = torch.cat([image_all, last_view], dim=1)                   # [b, v_all, c, h, w]
        else:
            pad_input = False
            v_all = v

        '''se3 pose prediction for all views'''
        # tokenize images, add spatial-temporal p.e.
        img_tokens = self.image_tokenizer(image_all)                              # [b*v_all, n, d]
        _, n, d = img_tokens.shape

        # add spatial positional embedding
        if self.use_pe_embedding_layer:
            img_tokens = self.add_spatial_pe(
                img_tokens,
                b, v_all,
                self.hh,
                self.ww,
                embedder=self.pe_embedder,
            )

        # concanate all tokens together
        cam_tokens = repeat(self.camera_token, '1 n d -> bv n d', bv=b*v_all)
        register_tokens = repeat(self.register_token, '1 n d -> bv n d', bv=b*v_all)
        all_tokens = torch.cat([cam_tokens, register_tokens, img_tokens], dim=1)
        _, n2, _ = all_tokens.shape
        all_tokens = rearrange(all_tokens, '(b v) n d -> b (v n) d', b=b)         # [b, v_all*n, d]

        # pose estimation for all views
        all_tokens = self.run_vggt_encoder(all_tokens, b, v_all)
        all_tokens = rearrange(all_tokens, 'b (v n) d -> (b v) n d', v=v_all)
        cam_tokens, _, _ = all_tokens.split([1, self.num_register_tokens, n], dim=1) 

        # get se3 poses and intrinsics
        cam_tokens = cam_tokens[:, 0]                                             # [b*v_all, d]
        cam_info = self.pose_predictor(cam_tokens, v_all)                 # [b*v_all, num_pose_element+3+4], rot, 3d trans, 4d fxfycxcy
        pred_c2w, pred_fxfycxcy = get_cam_se3(cam_info) # [b*v_all, 4, 4], [b*v_all, 4]
        pred_c2w = rearrange(pred_c2w, '(b v) n d -> b v n d', b=b)
        pred_fxfycxcy = rearrange(pred_fxfycxcy, '(b v) d -> b v d', b=b).detach()
        normalized = True

        # get plucker ray and embeddings
        if v < 5:
            v_input = 5
            c2w_input = pred_c2w[:, :5, ...]                                   # [b, v_input, 4, 4]
            fxfycxcy_input = pred_fxfycxcy[:, :5, ...]                         # [b, v_input, 4]
            img_tokens_input = rearrange(img_tokens, '(b v) n d -> b v n d', b=b)[:, :5, ...]
        else:
            v_input = v
            c2w_input = pred_c2w[:, :v, ...]                                   # [b, v_input, 4, 4]
            fxfycxcy_input = pred_fxfycxcy[:, :v, ...]                         # [b, v_input, 4]
            img_tokens_input = rearrange(img_tokens, '(b v) n d -> b v n d', b=b)[:, :v, ...]

        c2w_target = pred_c2w[:, :v]                                           # [b, v_target, 4, 4]
        fxfycxcy_target = pred_fxfycxcy[:, :v]                                 # [b, v_target, 4]
    
        plucker_rays_input = cam_info_to_plucker(c2w_input, fxfycxcy_input, self.config.model.target_image, normalized=normalized, return_moment=True)
        plucker_rays_input = rearrange(plucker_rays_input, '(b v) c h w -> b v h w c', b=b, v=v_input)
        plucker_emb_input = self.input_pose_tokenizer(plucker_rays_input)                                     # [b*v_input, n, d]
        if self.use_pe_embedding_layer:
            plucker_emb_input = self.add_spatial_pe(
                plucker_emb_input,
                b, v_input,
                self.hh, self.ww,
                embedder=self.pe_embedder_plucker,
            )
        plucker_emb_input = rearrange(plucker_emb_input, '(b v) n d -> b (v n) d', v=v_input)                 # [b, v_input*n, d]

        '''predict scene representation using (posed) input views'''
        # get posed image representation
        img_tokens_input = rearrange(img_tokens_input, 'b v n d -> b (v n) d')
        img_tokens_input = torch.cat([img_tokens_input, plucker_emb_input], dim=-1)                           # [b, v_input*n, 2d]
        all_tokens = self.mlp_fuse(img_tokens_input)                                                          # [b, v_input*n, d]

        # encoder layers, predict depths and feature vectors
        all_tokens = self.run_vggt_encoder_geom(all_tokens, b, v_input)                                                   # [b, v_input*n, d]
        img_aligned_gaussians = self.image_token_decoder(all_tokens)
        img_aligned_gaussians = rearrange(
            img_aligned_gaussians,
            'b (v n) d -> b v n d',
            v=v_input,
        )[:, :v]
        img_aligned_gaussians = rearrange(
            img_aligned_gaussians, 
            'b v n (ph pw c) -> b (v n ph pw) c', 
            ph=self.ph,
            pw=self.pw,
        )
        xyz, features, scaling, rotation, opacity = self.upsampler.to_gs(img_aligned_gaussians)
        img_aligned_xyz = rearrange(
            xyz,
            "b (v hh ww ph pw) c -> b v c (hh ph) (ww pw)",
            v=v,
            hh=self.hh,
            ww=self.ww,
            ph=self.ph,
            pw=self.pw,
        )

        if self.config.model.hard_pixelalign:
            img_aligned_xyz = img_aligned_xyz.mean(dim=2, keepdim=True)
            img_aligned_xyz = self.range_func(img_aligned_xyz)
            plucker_rays_input = cam_info_to_plucker(c2w_input[:, :v, ...], fxfycxcy_input[:, :v, ...], self.config.model.target_image, normalized=normalized, return_moment=False)
            plucker_rays_input = rearrange(plucker_rays_input, '(b v) c h w -> b v c h w', b=b)
            ray_o, ray_d = plucker_rays_input.split([3, 3], dim=2)
            img_aligned_xyz = ray_o + img_aligned_xyz * ray_d
            xyz = rearrange(
                img_aligned_xyz,
                "b v c (hh ph) (ww pw) -> b (v hh ww ph pw) c",
                ph=self.ph,
                pw=self.pw,
            )
        else:
            xyz = img_aligned_xyz 

        gaussian_attrs = edict(
            xyz=xyz,
            features=features,
            scaling=scaling,
            rotation=rotation,
            opacity=opacity,
        )

        loss_metrics = None
        render = None
        height, width = h, w
        if normalized:
            fxfycxcy_target[..., 0] *= width
            fxfycxcy_target[..., 1] *= height
            fxfycxcy_target[..., 2] *= width
            fxfycxcy_target[..., 3] *= height
        render = self.renderer(
            xyz,
            features,
            scaling,
            rotation,
            opacity,
            height,
            width,
            C2W=c2w_target,
            fxfycxcy=fxfycxcy_target,
        )
        render_results = edict(
            rendered_images=render.render
        )

        with torch.no_grad():
            vis_only_results = self.render_images_video(gaussian_attrs, c2w_target, fxfycxcy_target, normalized=False, step_back=self.config.get('evaluation_step_back_distance', 0))
            # vis_only_results_input = self.render_images_video(gaussian_attrs, c2w_input, fxfycxcy_input, normalized=False)

        gaussians = []
        pixelalign_xyz = []
        gaussians_usage = []
        gaussians_scale = []
        gaussians_opacity = []
        for b in range(xyz.size(0)):
            self.renderer.gaussians_model.empty()
            gaussians_model = copy.deepcopy(self.renderer.gaussians_model)
            gaussians.append(
                gaussians_model.set_data(
                    xyz[b].detach().float(),
                    features[b].detach().float(),
                    scaling[b].detach().float(),
                    rotation[b].detach().float(),
                    opacity[b].detach().float(),
                )
            )

            threshold = 0.05
            usage_mask = gaussians[-1].get_opacity > threshold
            usage = usage_mask.sum() / usage_mask.numel()
            if torch.is_tensor(usage):
                usage = usage.item()
            gaussians_usage.append(usage)

            mean_scale = gaussians[-1].get_scaling.mean()
            if torch.is_tensor(mean_scale):
                mean_scale = mean_scale.item()
            gaussians_scale.append(mean_scale)

            mean_opacity = gaussians[-1].get_opacity.mean()
            if torch.is_tensor(mean_opacity):
                mean_opacity = mean_opacity.item()
            gaussians_opacity.append(mean_opacity)

            img_aligned_xyz = gaussians[-1].get_xyz
            img_aligned_xyz = rearrange(
                img_aligned_xyz,
                "(v hh ww ph pw) c -> v c (hh ph) (ww pw)",
                v=v,
                hh=self.hh,
                ww=self.ww,
                ph=self.ph,
                pw=self.pw,
            )
            pixelalign_xyz.append(img_aligned_xyz)
        pixelalign_xyz = torch.stack(pixelalign_xyz, dim=0)

        # return results
        result = edict(
            ray_o=ray_o,
            gaussians=gaussians,
            pixelalign_xyz=pixelalign_xyz,
            image=data['image'],
            render=render_results.rendered_images,
            c2w=pred_c2w,
            fxfycxcy=rearrange(pred_fxfycxcy, 'b v d -> (b v) d'),
            c2w_input=c2w_input,
            fxfycxcy_input=fxfycxcy_input,
            c2w_target=c2w_target,
            fxfycxcy_target=fxfycxcy_target,
        )

        result.render_video = vis_only_results.rendered_images_video.detach().clamp(0, 1)

        return result

    def add_spatial_pe(self, tokens, b, v, h_tokens, w_tokens, embedder):
        """
        Add spatial (and optionally temporal) positional encoding to tokens.
        
        Args:
            tokens (Tensor): shape [b*v, n, d]
            b (int): batch size
            v (int): number of views
            h_tokens (int): number of tokens along height
            w_tokens (int): number of tokens along width
            embedder (nn.Module): PE projection layer

        Returns:
            Tensor: tokens with spatial PE added, shape [b*v, n, d]
        """
        bv, n, d = tokens.shape
        assert (h_tokens * w_tokens) == n, f"Token count {n} != h*w {h_tokens}x{w_tokens}"

        spatial_pe = get_2d_sincos_pos_embed(
            embed_dim=d,
            grid_size=(h_tokens, w_tokens),
            device=tokens.device
        ).to(tokens.dtype)  # [n, d]

        spatial_pe = spatial_pe.reshape(1, 1, n, d).repeat(b, v, 1, 1)   # [b, v, n, d]
        spatial_pe = spatial_pe.reshape(bv, n, d)                       # [b*v, n, d]

        pe = embedder(spatial_pe)                                       # [b*v, n, d]
        return tokens + pe

    @staticmethod
    def slice_expand_and_flatten(token_tensor, B, S):
        """
        Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
        1) Uses the first position (index=0) for the first frame only
        2) Uses the second position (index=1) for all remaining frames (S-1 frames)
        3) Expands both to match batch size B
        4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
        followed by (S-1) second-position tokens
        5) Flattens to (B*S, X, C) for processing

        Returns:
            torch.Tensor: Processed tokens with shape (B*S, X, C)
        """

        # Slice out the "query" tokens => shape (1, 1, ...)
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
        # Slice out the "other" tokens => shape (1, S-1, ...)
        others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
        # Concatenate => shape (B, S, ...)
        combined = torch.cat([query, others], dim=1)

        # Finally flatten => shape (B*S, ...)
        combined = combined.view(B * S, *combined.shape[2:])
        return combined

    def run_vggt_encoder(self, all_tokens_encoder, b, v):
        checkpoint_every = self.config.training.grad_checkpoint_every
        for i in range(0, len(self.transformer_encoder), checkpoint_every):
            if i % 2 == 0:
                # frame attention
                all_tokens_encoder = rearrange(all_tokens_encoder, 'b (v n) d -> (b v) n d', v=v)
            else:
                # global attention
                all_tokens_encoder = rearrange(all_tokens_encoder, '(b v) n d -> b (v n) d', b=b)

            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self.run_layers_encoder(i, i+1),
                all_tokens_encoder,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self.run_layers_encoder(i + 1, i + checkpoint_every)(all_tokens_encoder)
        return all_tokens_encoder
    
    def run_vggt_encoder_geom(self, all_tokens_encoder, b, v):
        checkpoint_every = self.config.training.grad_checkpoint_every
        for i in range(0, len(self.transformer_encoder_geom), checkpoint_every):
            if i % 2 == 0:
                # frame attention
                all_tokens_encoder = rearrange(all_tokens_encoder, 'b (v n) d -> (b v) n d', v=v)
            else:
                # global attention
                all_tokens_encoder = rearrange(all_tokens_encoder, '(b v) n d -> b (v n) d', b=b)

            all_tokens_encoder = torch.utils.checkpoint.checkpoint(
                self.run_layers_encoder_geom(i, i+1),
                all_tokens_encoder,
                use_reentrant=False,
            )
            if checkpoint_every > 1:
                all_tokens_encoder = self.run_layers_encoder_geom(i + 1, i + checkpoint_every)(all_tokens_encoder)
        return all_tokens_encoder

    def run_layers_encoder(self, start, end):
        def custom_forward(concat_nerf_img_tokens):
            for i in range(start, min(end, len(self.transformer_encoder))):
                concat_nerf_img_tokens = self.transformer_encoder[i](concat_nerf_img_tokens)
            return concat_nerf_img_tokens
        return custom_forward
    
    def run_layers_encoder_geom(self, start, end):
        def custom_forward(concat_nerf_img_tokens):
            for i in range(start, min(end, len(self.transformer_encoder_geom))):
                concat_nerf_img_tokens = self.transformer_encoder_geom[i](concat_nerf_img_tokens)
            return concat_nerf_img_tokens
        return custom_forward

    def render_images_video(self, gaussian_attrs, c2w_all, fxfycxcy_all, normalized=False, step_back=0):
        '''
        points_input_all  : [b, v_input, h_rend_input, w_rend_input, 3]
        features_input_all: [b, v_input, h_rend_input, w_rend_input, c]
        c2w_all           : [b*v, 4, 4]
        fxfycxcy_all      : [b*v, 4]
        '''
        with torch.no_grad():
            xyz = gaussian_attrs.xyz.detach()
            features = gaussian_attrs.features.detach()
            scaling = gaussian_attrs.scaling.detach()
            rotation = gaussian_attrs.rotation.detach()
            opacity = gaussian_attrs.opacity.detach()
            c2w_all = c2w_all.detach()
            fxfycxcy_all = fxfycxcy_all.detach()

            b, v, _, _ = c2w_all.shape
            device = xyz.device

            all_renderings = []
            num_frames = 30
            traj_type = "interpolate"
            order_poses = False
            
            for i in range(b):
                c2ws = c2w_all[i]                   # [v, 4, 4]
                fxfycxcy = fxfycxcy_all[i]          # [v, 4]
                if traj_type == "interpolate":
                    # build Ks from fxfycxcy
                    Ks = torch.zeros((c2ws.shape[0], 3, 3), device=device)
                    Ks[:, 0, 0] = fxfycxcy[:, 0]
                    Ks[:, 1, 1] = fxfycxcy[:, 1]
                    Ks[:, 0, 2] = fxfycxcy[:, 2]
                    Ks[:, 1, 2] = fxfycxcy[:, 3]
                    c2ws, Ks = camera_utils.get_interpolated_poses_many(
                        c2ws[:, :3, :4], Ks, num_frames, order_poses=order_poses
                    )
                    frame_c2ws = torch.cat([c2ws.to(device), torch.tensor([[[0, 0, 0, 1]]], device=device).repeat(c2ws.shape[0], 1, 1)], dim=1)    # [v',4,4]
                    frame_fxfycxcy = torch.zeros((c2ws.shape[0], 4), device=device)
                    frame_fxfycxcy[:, 0] = Ks[:, 0, 0]      # [v',4]
                    frame_fxfycxcy[:, 1] = Ks[:, 1, 1]
                    frame_fxfycxcy[:, 2] = Ks[:, 0, 2]
                    frame_fxfycxcy[:, 3] = Ks[:, 1, 2]
                elif traj_type == "same":
                    frame_c2ws = c2ws.clone()
                    frame_fxfycxcy = fxfycxcy.clone()
                else:
                    raise NotImplementedError
                
                if step_back > 0:
                    frame_c2ws = build_stepback_c2ws(frame_c2ws, step_back_distance=step_back)

                batch_size = 5
                num_views = frame_c2ws.shape[0]
                renderings = []

                for start in range(0, num_views, batch_size):
                    end = min(start + batch_size, num_views)

                    # Slice views
                    batch_c2w = frame_c2ws[start:end].unsqueeze(0)  # [1, batch_size, 4, 4]
                    batch_fx = frame_fxfycxcy[start:end].unsqueeze(0)  # [1, batch_size, 4]

                    # Call renderer on minibatch
                    rendered_images_batch = self.renderer(
                        xyz,
                        features,
                        scaling,
                        rotation,
                        opacity,
                        self.config.model.image_tokenizer.image_size,
                        self.config.model.image_tokenizer.image_size,
                        C2W=batch_c2w,
                        fxfycxcy=batch_fx,
                    ).render.squeeze(0)  # [batch_size, H, W, 3]

                    renderings.append(rendered_images_batch)

                # Concatenate all rendered minibatches
                rendered_images_all = torch.cat(renderings, dim=0)  # [num_views, H, W, 3]
                all_renderings.append(rendered_images_all)

            all_renderings = torch.stack(all_renderings)    # [b,v',c,h,w]

        render_results = edict(
            rendered_images_video=all_renderings
        )
        return render_results


def get_cam_se3(cam_info):
    '''
    cam_info: [b,num_pose_element+3+4], rot, 3d trans, 4d fxfycxcy
    '''
    b, n = cam_info.shape

    if n == 13:
        rot_6d = cam_info[:,:6]
        R = rot6d2mat(rot_6d)  # [b,3,3]
        t = cam_info[:,6:9].unsqueeze(-1)   # [b,3,1]
        fxfycxcy = cam_info[:,9:]   # normalized by resolution / shift from average, [b,4]
    elif n == 11:
        rot_quat = cam_info[:,:4]
        R = quat2mat(rot_quat)
        t = cam_info[:,4:7].unsqueeze(-1)   # [b,3,1]
        fxfycxcy = cam_info[:,7:]   # normalized by resolution / shift from average, [b,4]
    else:
        raise NotImplementedError

    Rt = torch.cat([R, t], dim=2)  # [b,3,4]
    c2w = torch.cat([Rt, 
                    torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device).view(1, 1, 4).repeat(b, 1, 1)], dim=1)  # [b,4,4]
    return c2w, fxfycxcy


def cam_info_to_plucker(c2w, fxfycxcy, target_imgs_info, normalized=True, return_moment=True):
    '''
    c2w: [b,4,4] or [b,v,4,4]
    fxfycxcy: [b,4] or [b,v,4]
    '''
    if len(c2w.shape) == 3: 
        b = c2w.shape[0]
    elif len(c2w.shape) == 4:
        c2w, fxfycxcy = rearrange(c2w.clone(), "b v n d -> (b v) n d"), \
            rearrange(fxfycxcy.clone(), "b v d -> (b v) d")
        b = c2w.shape[0]

    device = c2w.device
    h, w = target_imgs_info.height, target_imgs_info.width

    fxfycxcy = fxfycxcy.clone()
    if normalized:
        fxfycxcy[:, 0] *= w
        fxfycxcy[:, 1] *= h
        fxfycxcy[:, 2] *= w
        fxfycxcy[:, 3] *= h

    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y, x = y.to(c2w), x.to(c2w)
    x = x[None, :, :].expand(b, -1, -1).reshape(b, -1)
    y = y[None, :, :].expand(b, -1, -1).reshape(b, -1)
    x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
    y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
    z = torch.ones_like(x)
    ray_d = torch.stack([x, y, z], dim=2)  # [b, h*w, 3]
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)  # [b, h*w, 3]
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b, h*w, 3]

    ray_o = ray_o.reshape(b, h, w, 3).permute(0, 3, 1, 2)   # [b,3,h,w]
    ray_d = ray_d.reshape(b, h, w, 3).permute(0, 3, 1, 2)
    
    if return_moment:
        plucker = torch.cat(
            [
                torch.cross(ray_o, ray_d, dim=1),
                ray_d,
            ],
            dim=1,
        )
    else:
        plucker = torch.cat(
            [
                ray_o,
                ray_d,
            ],
            dim=1,
        )
    return plucker     # [b,c=6,h,w]


class CanonicalKHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.model.transformer.d

        self.layers = nn.ModuleList([
            nn.Linear(d, d, bias=True),
            nn.SiLU(),
            nn.Linear(d, 1, bias=True)
        ])

        # Apply weight initialization
        self.apply(_init_weights)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(f"Output after layer {i} ({layer.__class__.__name__}): {x}")
        return x


class PoseEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.canonical = self.config.model.pose_latent.get('canonical', 'first')
        assert self.canonical in ["first", "middle", "unordered"], \
            f"Unknown canonical mode: {self.canonical}"
        self.is_pairwise = self.config.model.pose_latent.get('mode', 'pairwise') == 'pairwise'
        self.rel_head_input = self.config.model.transformer.d * 2 if self.is_pairwise else config.model.transformer.d
        self.pose_consistency_reg_weight = self.config.training.get('pose_consistency_reg_weight', 0.0)

        self.pose_rep = self.config.model.pose_latent.get('representation', '6d')
        print('Pose representation:', self.pose_rep)
        if self.pose_rep == '6d':
            self.num_pose_element = 6
        elif self.pose_rep == 'quat':
            self.num_pose_element = 4
        else:
            raise NotImplementedError

        self.rel_head = nn.Sequential(
            nn.Linear(
                self.rel_head_input,
                config.model.transformer.d,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                config.model.transformer.d,
                self.num_pose_element+3,
                bias=True,
            ),
        )
        self.rel_head.apply(_init_weights)

        self.canonical_k_head = nn.Sequential(
            nn.Linear(
                config.model.transformer.d,
                config.model.transformer.d,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                config.model.transformer.d,
                1,
                bias=False,
            ),
        )
        self.canonical_k_head.apply(_init_weights)

        self.f_bias = 1.25

    def forward(self, x, v):
        '''
        x: [b*v, d] or [b, v, d]
        '''

        if x.ndim == 2:
            x = rearrange(x, '(b v) d -> b v d', v=v)
            return_dim2 = True
        else:
            return_dim2 = False

        b = x.shape[0]

        if self.is_pairwise:
            if v == 1:
                return self.forward_canonical_single_view(x, v, self.canonical, return_dim2)
            else:
                return self.forward_canonical(x, v, self.canonical, return_dim2)
        else:
            return self.forward_global(x, v, self.canonical, return_dim2)

    def forward_global(self, x, v, canonical, return_dim2):
        '''
        Unordered: extrinsics per view, intrinsics averaged across views.
        Adds predicted offsets to canonical identity extrinsics per view.
        '''
        b = x.shape[0]

        # Identity canonical extrinsics per view
        if self.pose_rep == '6d':
            rt_canonical = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0], device=x.device).reshape(1, 1, 9).repeat(b, v, 1)
        elif self.pose_rep == 'quat':
            rt_canonical = torch.tensor([1, 0, 0, 0, 0, 0, 0], device=x.device).reshape(1, 1, 7).repeat(b, v, 1)
        else:
            raise NotImplementedError

        # Predict extrinsics offset (rotation and translation)
        extrinsics_offset = self.rel_head(x)  # [b, v, num_pose_element + 3]

        # Determine canonical index
        if self.canonical == "first":
            cano_idx = 0
        elif self.canonical == "middle":
            cano_idx = v // 2
        elif self.canonical == "unordered":
            cano_idx = None  # no canonical frame; apply residuals to all
        else:
            raise ValueError(f"Unknown canonical mode: {self.canonical}")

        # Apply residuals
        if cano_idx is None:
            rt_final = rt_canonical + extrinsics_offset
        else:
            # Ordered mode: zero out residuals for canonical frame
            mask = torch.ones((1, v, 1), device=x.device)
            mask[:, cano_idx, :] = 0.0
            rt_final = rt_canonical + extrinsics_offset * mask

        # Predict fx/fy per view
        fxfy_per_view = self.canonical_k_head(x) + self.f_bias  # [b, v, 1]
        fxfy_per_view = fxfy_per_view.repeat(1, 1, 2)           # [b, v, 2]

        # Average intrinsics across views
        fxfy_avg = fxfy_per_view.mean(dim=1, keepdim=True)      # [b, 1, 2]
        fxfy_all = fxfy_avg.repeat(1, v, 1)                     # [b, v, 2]

        # Combine extrinsics and intrinsics
        info_all = torch.cat([rt_final, fxfy_all], dim=-1)  # [b, v, num_pose_element + 3 + 2]

        # Add cx, cy (fixed at 0.5)
        cxcy_all = torch.tensor([0.5, 0.5], device=info_all.device).reshape(1, 1, 2).repeat(b, v, 1)
        info_all = torch.cat([info_all, cxcy_all], dim=-1)

        return rearrange(info_all, 'b v d -> (b v) d') if return_dim2 else info_all

    def forward_canonical(self, x, v, canonical, return_dim2):
        '''
        Canonical-based modes: "first" or "middle"
        '''
        b = x.shape[0]

        if canonical == 'first':
            x_canonical = x[:, 0:1]     # [b, 1, d]
            x_rel = x[:, 1:]            # [b, v-1, d]
        elif canonical == 'middle':
            cano_idx = v // 2
            rel_indices = torch.cat([torch.arange(cano_idx), torch.arange(cano_idx + 1, v)], dim=0).to(x.device)
            x_canonical = x[:, cano_idx:cano_idx+1]  # [b, 1, d]
            x_rel = x[:, rel_indices]                # [b, v-1, d]
        else:
            raise NotImplementedError

        # Predict canonical intrinsics
        fxfy_canonical = self.canonical_k_head(x_canonical[:, 0]) + self.f_bias  # [b, 1]
        fxfy_canonical = fxfy_canonical.unsqueeze(1).repeat(1, 1, 2)             # [b, 1, 2]

        # Identity canonical extrinsics
        if self.pose_rep == '6d':
            rt_canonical = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0], device=fxfy_canonical.device).reshape(1, 1, 9).repeat(b, 1, 1)
        elif self.pose_rep == 'quat':
            rt_canonical = torch.tensor([1, 0, 0, 0, 0, 0, 0], device=fxfy_canonical.device).reshape(1, 1, 7).repeat(b, 1, 1)

        info_canonical = torch.cat([rt_canonical, fxfy_canonical], dim=-1)  # [b, 1, num_pose_element + 3 + 2]

        # Relative extrinsics
        feat_rel = torch.cat([x_canonical.repeat(1, v - 1, 1), x_rel], dim=-1)  # [b, v-1, 2*d]
        info_rel = self.rel_head(feat_rel)                                     # [b, v-1, num_pose_element + 3]
        info_all = info_canonical.repeat(1, v, 1)                             # [b, v, num_pose_element + 3 + 2]

        if canonical == 'first':
            info_all[:, 1:, :self.num_pose_element + 3] += info_rel
        elif canonical == 'middle':
            info_all[:, rel_indices, :self.num_pose_element + 3] += info_rel

        # Add cx, cy (fixed at 0.5)
        cxcy_all = torch.tensor([0.5, 0.5], device=info_all.device).reshape(1, 1, 2).repeat(b, v, 1)
        info_all = torch.cat([info_all, cxcy_all], dim=-1)

        if self.pose_consistency_reg_weight > 0:
            info_reg = self.forward_canonical_reg(x, info_all[:, :, :self.num_pose_element + 3], canonical)
            return rearrange(info_all, 'b v d -> (b v) d') if return_dim2 else info_all, info_reg
        else:
            return rearrange(info_all, 'b v d -> (b v) d') if return_dim2 else info_all

    def forward_canonical_single_view(self, x, v, canonical, return_dim2):
        '''
        Canonical-based modes: "first" or "middle"
        '''
        b = x.shape[0]

        x_canonical = x[:, 0:1]     # [b, 1, d]

        # Predict canonical intrinsics
        fxfy_canonical = self.canonical_k_head(x_canonical[:, 0]) + self.f_bias  # [b, 1]
        fxfy_canonical = fxfy_canonical.unsqueeze(1).repeat(1, 1, 2)             # [b, 1, 2]

        # Identity canonical extrinsics
        if self.pose_rep == '6d':
            rt_canonical = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 0], device=fxfy_canonical.device).reshape(1, 1, 9).repeat(b, 1, 1)
        elif self.pose_rep == 'quat':
            rt_canonical = torch.tensor([1, 0, 0, 0, 0, 0, 0], device=fxfy_canonical.device).reshape(1, 1, 7).repeat(b, 1, 1)

        info_canonical = torch.cat([rt_canonical, fxfy_canonical], dim=-1)    # [b, 1, num_pose_element + 3 + 2]                                    # [b, v-1, num_pose_element + 3]
        info_all = info_canonical.repeat(1, v, 1)                             # [b, v, num_pose_element + 3 + 2]

        # Add cx, cy (fixed at 0.5)
        cxcy_all = torch.tensor([0.5, 0.5], device=info_all.device).reshape(1, 1, 2).repeat(b, v, 1)
        info_all = torch.cat([info_all, cxcy_all], dim=-1)

        return rearrange(info_all, 'b v d -> (b v) d') if return_dim2 else info_all