"""
Dilated U-Net + Global & Local PatchGAN (Baseline chuan)
========================================================
Nghien cuu nen tang:
- Context Encoders (CVPR 2016)
- Globally and Locally Consistent Image Completion (SIGGRAPH 2017, Iizuka et al.)
- PatchGAN (Pix2Pix, CVPR 2017) & WGAN-GP (NeurIPS 2017)

Muc tieu nghien cuu (theo HIHI.pdf):
- La mo hinh doi chung (Baseline) dung tich chap thong thuong (standard nn.Conv2d),
  KHONG dung PartialConv / GatedConv / Contextual Attention / FFC, KHONG pretrained.
- Chen chuoi Dilated Convolution (dilation = 2, 4, 8, 16) tai bottleneck 64x64 de mo rong
  vung tiep nhan ma khong giam do phan giai dac trung.
- Ket hop 2 bo phan biet (Discriminator):
    1) GlobalDiscriminator: danh gia tinh nhat quan toan cuc tren toan anh 256x256.
    2) LocalPatchGANDiscriminator: PatchGAN 70x70 danh gia do sac net cuc bo tai
       vung khuyet 128x128 (cat quanh tam lo khuyet).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_norm(norm_type: str, num_channels: int) -> nn.Module:
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class ConvBlock(nn.Module):
    """Standard Conv2d + Norm + ReLU block (no gating/partial mask)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        norm_type: str = "instance",
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=(not use_norm or norm_type == "instance"),
            )
        ]
        if use_norm and norm_type != "none":
            layers.append(_make_norm(norm_type, out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DilatedUNetGenerator(nn.Module):
    """
    Encoder-Decoder U-Net voi chuoi Dilated Convolution (dilation = 2, 4, 8, 16)
    tai bottleneck 64x64.

    Input:
        masked_img: (B, 3, 256, 256) float in [0, 1] (vung khuyet = 0)
        hole_mask:  (B, 1, 256, 256) float in {0, 1} (1 = vung khuyet)
    Output:
        pred_img:   (B, 3, 256, 256) float in [0, 1]
    """

    def __init__(self, base_ch: int = 64, norm_type: str = "instance") -> None:
        super().__init__()
        c1 = base_ch          # 64  at 256x256
        c2 = base_ch * 2      # 128 at 128x128
        c3 = base_ch * 4      # 256 at 64x64

        # Encoder (256 -> 128 -> 64)
        self.enc1 = nn.Sequential(
            ConvBlock(4, c1, kernel_size=5, stride=1, padding=2, norm_type=norm_type, use_norm=False),
            ConvBlock(c1, c1, kernel_size=3, stride=1, padding=1, norm_type=norm_type, use_norm=True),
        )
        self.enc2 = nn.Sequential(
            ConvBlock(c1, c2, kernel_size=3, stride=2, padding=1, norm_type=norm_type, use_norm=True),
            ConvBlock(c2, c2, kernel_size=3, stride=1, padding=1, norm_type=norm_type, use_norm=True),
        )
        self.enc3 = nn.Sequential(
            ConvBlock(c2, c3, kernel_size=3, stride=2, padding=1, norm_type=norm_type, use_norm=True),
            ConvBlock(c3, c3, kernel_size=3, stride=1, padding=1, norm_type=norm_type, use_norm=True),
        )

        # Bottleneck at 64x64: chuoi Dilated Convolution (dilation = 2, 4, 8, 16)
        # Span cua dilation=16 tren 64x64 la 33px (>50% anh), tong receptive field ~63/64.
        self.dilated_bottleneck = nn.Sequential(
            ConvBlock(c3, c3, kernel_size=3, stride=1, padding=2, dilation=2, norm_type=norm_type),
            ConvBlock(c3, c3, kernel_size=3, stride=1, padding=4, dilation=4, norm_type=norm_type),
            ConvBlock(c3, c3, kernel_size=3, stride=1, padding=8, dilation=8, norm_type=norm_type),
            ConvBlock(c3, c3, kernel_size=3, stride=1, padding=16, dilation=16, norm_type=norm_type),
            # Conv 3x3 dilation=1 giup lam min gridding artifacts cua dilated conv
            ConvBlock(c3, c3, kernel_size=3, stride=1, padding=1, dilation=1, norm_type=norm_type),
        )

        # Decoder (64 -> 128 -> 256) dung Nearest Upsample + Conv3x3 + U-Net Skip Connections
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(c3, c2, kernel_size=3, stride=1, padding=1, norm_type=norm_type),
        )
        self.dec2 = nn.Sequential(
            ConvBlock(c2 * 2, c2, kernel_size=3, stride=1, padding=1, norm_type=norm_type),
            ConvBlock(c2, c2, kernel_size=3, stride=1, padding=1, norm_type=norm_type),
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(c2, c1, kernel_size=3, stride=1, padding=1, norm_type=norm_type),
        )
        self.dec1 = nn.Sequential(
            ConvBlock(c1 * 2, c1, kernel_size=3, stride=1, padding=1, norm_type=norm_type),
            ConvBlock(c1, c1 // 2, kernel_size=3, stride=1, padding=1, norm_type=norm_type),
            nn.Conv2d(c1 // 2, 3, kernel_size=3, stride=1, padding=1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.0, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, masked_img: torch.Tensor, hole_mask: torch.Tensor) -> torch.Tensor:
        # Chuyen pixel hop le ve [-1, 1], giu vung khuyet = 0 (standard conv input)
        x_in = (masked_img * 2.0 - 1.0) * (1.0 - hole_mask)
        inp = torch.cat([x_in, hole_mask], dim=1)

        s1 = self.enc1(inp)          # (B, 64, 256, 256)
        s2 = self.enc2(s1)           # (B, 128, 128, 128)
        b0 = self.enc3(s2)           # (B, 256, 64, 64)

        # Residual connection quanh chuoi dilated giup on dinh gradient khi train tu dau
        b = b0 + self.dilated_bottleneck(b0)

        d2 = self.up2(b)             # (B, 128, 128, 128)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = self.up1(d2)            # (B, 64, 256, 256)
        out = self.dec1(torch.cat([d1, s1], dim=1))
        return (torch.tanh(out) + 1.0) * 0.5


def _maybe_sn(conv: nn.Conv2d, use_sn: bool) -> nn.Module:
    return nn.utils.spectral_norm(conv) if use_sn else conv


class GlobalDiscriminator(nn.Module):
    """
    Global Discriminator (256x256 -> 1x1 score):
    Danh gia tinh nhat quan ngu canh toan cuc tren toan bo anh.
    Khong dung BatchNorm de tuong thich hoan toan voi WGAN-GP.
    """

    def __init__(self, base_ch: int = 64, use_sn: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _maybe_sn(nn.Conv2d(3, base_ch, 4, stride=2, padding=1), use_sn),             # 128
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1), use_sn),   # 64
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1), use_sn),  # 32
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1), use_sn),  # 16
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1), use_sn),  # 8
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(base_ch * 4, base_ch * 4, 4, stride=2, padding=1), use_sn),  # 4
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(base_ch * 4, 1, 4, stride=1, padding=0), use_sn),         # 1x1
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img_norm: torch.Tensor) -> torch.Tensor:
        # img_norm: (B, 3, 256, 256) in [-1, 1] -> (B, 1, 1, 1)
        return self.net(img_norm)


class LocalPatchGANDiscriminator(nn.Module):
    """
    Local 70x70 PatchGAN Discriminator (128x128 -> 14x14 patch score map):
    Danh gia do sac net ket cau cuc bo tai vung khuyet 128x128.
    Khong dung BatchNorm de tuong thich hoan toan voi WGAN-GP.
    """

    def __init__(self, base_ch: int = 64, use_sn: bool = False) -> None:
        super().__init__()
        self.net = nn.Sequential(
            _maybe_sn(nn.Conv2d(3, base_ch, 4, stride=2, padding=1), use_sn),              # 64
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(base_ch, base_ch * 2, 4, stride=2, padding=1), use_sn),    # 32
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(base_ch * 2, base_ch * 4, 4, stride=2, padding=1), use_sn),  # 16
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(base_ch * 4, base_ch * 8, 4, stride=1, padding=1), use_sn),  # 15
            nn.LeakyReLU(0.2, inplace=True),
            _maybe_sn(nn.Conv2d(base_ch * 8, 1, 4, stride=1, padding=1), use_sn),          # 14x14
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, local_img_norm: torch.Tensor) -> torch.Tensor:
        # local_img_norm: (B, 3, 128, 128) in [-1, 1] -> (B, 1, 14, 14)
        return self.net(local_img_norm)


def compute_local_crop_boxes(hole_mask: torch.Tensor, crop_size: int = 128) -> List[Tuple[int, int]]:
    """
    Tinh toa do goc tren-trai (y0, x0) cho crop 128x128 quanh trong tam lo khuyet
    cua tung anh trong batch. Chay tren CPU tensor (1 lan copy nho) de tranh 32 lan
    CUDA stream sync (.item()) tren GPU moi batch.
    """
    b, _, h, w = hole_mask.shape
    half = crop_size // 2
    max_y = max(0, h - crop_size)
    max_x = max(0, w - crop_size)
    boxes: List[Tuple[int, int]] = []

    with torch.no_grad():
        mask_cpu = (hole_mask[:, 0].detach().cpu().numpy() > 0.5)
        for i in range(b):
            ys, xs = np.nonzero(mask_cpu[i])
            if ys.size == 0:
                cy, cx = h // 2, w // 2
            else:
                cy = int(ys.mean())
                cx = int(xs.mean())
            y0 = max(0, min(cy - half, max_y))
            x0 = max(0, min(cx - half, max_x))
            boxes.append((y0, x0))
    return boxes


def crop_by_boxes(
    tensor: torch.Tensor,
    boxes: List[Tuple[int, int]],
    crop_size: int = 128,
) -> torch.Tensor:
    """Cat tensor (B, C, H, W) theo danh sach box (y0, x0) — bao toan autograd."""
    crops = [
        tensor[i : i + 1, :, y0 : y0 + crop_size, x0 : x0 + crop_size]
        for i, (y0, x0) in enumerate(boxes)
    ]
    return torch.cat(crops, dim=0)


def spatial_discounted_l1_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    weight_map: torch.Tensor,
) -> torch.Tensor:
    """
    Spatial Discounted L1 Loss (gamma^distance-to-hole trong lo, 1.0 ngoai lo),
    reduction mean-theo-pixel theo dung recipe chung cua nhom.
    """
    return (torch.abs(pred - gt) * weight_map).mean()


def compute_hole_gradient_penalty(
    critic: nn.Module,
    real_norm: torch.Tensor,
    fake_norm: torch.Tensor,
    hole_mask: torch.Tensor,
) -> torch.Tensor:
    """
    WGAN-GP Gradient Penalty chi nhan trong vung lo (hole_mask), dung theo
    HUONG_DAN_TRAIN.docx va paper DeepFillv1 / Iizuka.
    """
    b = real_norm.size(0)
    alpha = torch.rand(b, 1, 1, 1, device=real_norm.device, dtype=real_norm.dtype)
    interp = (alpha * real_norm + (1.0 - alpha) * fake_norm).requires_grad_(True)

    out = critic(interp)
    # Trung binh theo khong gian de PatchGAN va Global co cung thang diem moi anh
    scores = out.reshape(b, -1).mean(dim=1)
    grads = torch.autograd.grad(
        outputs=scores.sum(),
        inputs=interp,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grads_hole = grads * hole_mask
    grad_norm = torch.sqrt((grads_hole.reshape(b, -1) ** 2).sum(dim=1) + 1e-12)
    return ((grad_norm - 1.0) ** 2).mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
