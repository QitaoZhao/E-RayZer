from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import scipy.io
import torch
import torch.nn as nn


class VGG19(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.max1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.max2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.relu6 = nn.ReLU(inplace=True)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.relu8 = nn.ReLU(inplace=True)
        self.max3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu10 = nn.ReLU(inplace=True)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu12 = nn.ReLU(inplace=True)
        self.max4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu13 = nn.ReLU(inplace=True)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu14 = nn.ReLU(inplace=True)
        self.conv15 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu15 = nn.ReLU(inplace=True)
        self.conv16 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.relu16 = nn.ReLU(inplace=True)
        self.max5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out1 = self.conv1(x)
        out2 = self.relu1(out1)
        out3 = self.conv2(out2)
        out4 = self.relu2(out3)
        out5 = self.max1(out4)
        out6 = self.conv3(out5)
        out7 = self.relu3(out6)
        out8 = self.conv4(out7)
        out9 = self.relu4(out8)
        out10 = self.max2(out9)
        out11 = self.conv5(out10)
        out12 = self.relu5(out11)
        out13 = self.conv6(out12)
        out14 = self.relu6(out13)
        out15 = self.conv7(out14)
        out16 = self.relu7(out15)
        out17 = self.conv8(out16)
        out18 = self.relu8(out17)
        out19 = self.max3(out18)
        out20 = self.conv9(out19)
        out21 = self.relu9(out20)
        out22 = self.conv10(out21)
        out23 = self.relu10(out22)
        out24 = self.conv11(out23)
        out25 = self.relu11(out24)
        out26 = self.conv12(out25)
        out27 = self.relu12(out26)
        out28 = self.max4(out27)
        out29 = self.conv13(out28)
        out30 = self.relu13(out29)
        out31 = self.conv14(out30)
        out32 = self.relu14(out31)
        return out4, out9, out14, out23, out32


class PerceptualLoss(nn.Module):
    """MatConvNet VGG19 perceptual loss used by the training setup."""

    def __init__(self, weight_path: Optional[str] = None) -> None:
        super().__init__()
        resolved_weight_path = self._resolve_weight_path(weight_path)
        self.net = VGG19()
        self._load_matconvnet_weights(resolved_weight_path)
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

    @staticmethod
    def _resolve_weight_path(weight_path: Optional[str]) -> str:
        candidates = [
            weight_path,
            os.environ.get("ERAYZER_VGG19_WEIGHT_PATH"),
            "weights/imagenet-vgg-verydeep-19.mat",
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return candidate
        raise FileNotFoundError(
            "VGG19 perceptual loss weights not found. Set training.perceptual_loss_weight_path "
            "or PERCEPTUAL_LOSS_WEIGHT_PATH/ERAYZER_VGG19_WEIGHT_PATH to "
            "imagenet-vgg-verydeep-19.mat."
        )

    def _load_matconvnet_weights(self, weight_path: str) -> None:
        vgg_rawnet = scipy.io.loadmat(weight_path)
        vgg_layers = vgg_rawnet["layers"][0]
        layers = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        names = [
            "conv1",
            "conv2",
            "conv3",
            "conv4",
            "conv5",
            "conv6",
            "conv7",
            "conv8",
            "conv9",
            "conv10",
            "conv11",
            "conv12",
            "conv13",
            "conv14",
            "conv15",
            "conv16",
        ]
        sizes = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
        for layer_idx, name in enumerate(names):
            layer = getattr(self.net, name)
            layer.weight = nn.Parameter(
                torch.from_numpy(vgg_layers[layers[layer_idx]][0][0][2][0][0]).permute(3, 2, 0, 1)
            )
            layer.bias = nn.Parameter(
                torch.from_numpy(vgg_layers[layers[layer_idx]][0][0][2][0][1]).view(sizes[layer_idx])
            )

    @staticmethod
    def compute_error(truth: torch.Tensor, pred: torch.Tensor, conf: Optional[torch.Tensor] = None) -> torch.Tensor:
        if conf is not None:
            return torch.mean(torch.abs(truth - pred) * conf)
        return torch.mean(torch.abs(truth - pred))

    def forward(self, pred_img: torch.Tensor, real_img: torch.Tensor, conf: Optional[torch.Tensor] = None) -> torch.Tensor:
        bb = pred_img.new_tensor([123.6800, 116.7790, 103.9390]).reshape(1, 3, 1, 1)
        real_img_sb = real_img * 255.0 - bb
        pred_img_sb = pred_img * 255.0 - bb

        out3_r, out8_r, out13_r, out22_r, out33_r = self.net(real_img_sb)
        out3_f, out8_f, out13_f, out22_f, out33_f = self.net(pred_img_sb)

        e0 = self.compute_error(real_img_sb, pred_img_sb, conf=conf)
        e1 = self.compute_error(out3_r, out3_f) / 2.6
        e2 = self.compute_error(out8_r, out8_f) / 4.8
        e3 = self.compute_error(out13_r, out13_f) / 3.7
        e4 = self.compute_error(out22_r, out22_f) / 5.6
        e5 = self.compute_error(out33_r, out33_f) * 10.0 / 1.5
        return (e0 + e1 + e2 + e3 + e4 + e5) / 255.0
