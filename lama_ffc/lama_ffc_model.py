"""
Kien truc LaMa rut gon: Fast Fourier Convolution (FFC)
Cau hinh khop BAO_CAO_LAMA_FFC.docx:
  - Input: 4 kenh (RGB masked + hole_mask)
  - Encoder (256 -> 128 -> 64 -> 32): base_ch = 64 -> 128 -> 256 -> bottleneck_ch = 384
  - Bottleneck: 9 x FFC-ResBlock (ratio_g = 0.75 -> 25% Local Conv3x3 (96 ch) + 75% Global 2D Real FFT rfft2/irfft2 (288 ch))
  - Decoder: Nearest Upsample + ReflectionPad2d Conv3x3 (32 -> 64 -> 128 -> 256)
  - San sang nap file trong so ckpt_epoch22.pt / ckpt_best.pt khi cap nhat.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FourierUnit(nn.Module):
    """Global Spectral Transform via 2D Real FFT (rfft2 / irfft2)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch * 2, out_ch * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch * 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # fp32 rfft2 for numerical stability
        ffted = torch.fft.rfft2(x.float(), norm="ortho")
        # Stack real & imaginary parts along channel dim: (B, 2*C, H, W//2 + 1)
        ffted_ri = torch.cat([ffted.real, ffted.imag], dim=1)
        ffted_ri = self.act(self.bn(self.conv(ffted_ri)))
        real, imag = torch.chunk(ffted_ri, 2, dim=1)
        ffted_out = torch.complex(real, imag)
        output = torch.fft.irfft2(ffted_out, s=(h, w), norm="ortho")
        return output.to(x.dtype)


class SpectralTransform(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        half_ch = max(1, out_ch // 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, half_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(half_ch),
            nn.ReLU(inplace=True),
        )
        self.fu = FourierUnit(half_ch, half_ch)
        self.conv2 = nn.Conv2d(half_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        out = self.fu(x)
        return self.conv2(x + out)


class FFCBlock(nn.Module):
    """Fast Fourier Convolution mixing Local (25%) and Global (75%) branches."""

    def __init__(self, ch: int = 384, ratio_g: float = 0.75) -> None:
        super().__init__()
        self.ch_g = int(ch * ratio_g)       # 288
        self.ch_l = ch - self.ch_g          # 96

        self.conv_l2l = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.ch_l, self.ch_l, kernel_size=3, bias=False),
        )
        self.conv_l2g = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.ch_l, self.ch_g, kernel_size=3, bias=False),
        )
        self.conv_g2l = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.ch_g, self.ch_l, kernel_size=3, bias=False),
        )
        self.conv_g2g = SpectralTransform(self.ch_g, self.ch_g)

        self.bn_l = nn.BatchNorm2d(self.ch_l)
        self.bn_g = nn.BatchNorm2d(self.ch_g)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = x
        out_l = self.act(self.bn_l(self.conv_l2l(x_l) + self.conv_g2l(x_g)))
        out_g = self.act(self.bn_g(self.conv_l2g(x_l) + self.conv_g2g(x_g)))
        return out_l, out_g


class FFCResnetBlock(nn.Module):
    def __init__(self, ch: int = 384, ratio_g: float = 0.75) -> None:
        super().__init__()
        self.ffc1 = FFCBlock(ch, ratio_g)
        self.ffc2 = FFCBlock(ch, ratio_g)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x_l, x_g = x
        y_l, y_g = self.ffc1((x_l, x_g))
        y_l, y_g = self.ffc2((y_l, y_g))
        return x_l + y_l, x_g + y_g


class LaMaFFCGenerator(nn.Module):
    """
    LaMa rut gon: Encoder (256->32) + 9x FFC-ResBlock (384 ch, ratio_g=0.75) + Decoder (32->256).
    """

    def __init__(self, base_ch: int = 64, bottleneck_ch: int = 384, n_blocks: int = 9, ratio_g: float = 0.75) -> None:
        super().__init__()
        self.ch_g = int(bottleneck_ch * ratio_g)
        self.ch_l = bottleneck_ch - self.ch_g

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, base_ch, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            # 256 -> 128
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            # 128 -> 64
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            # 64 -> 32
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch * 4, bottleneck_ch, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.ModuleList([FFCResnetBlock(bottleneck_ch, ratio_g) for _ in range(n_blocks)])

        self.decoder = nn.Sequential(
            # 32 -> 64
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(bottleneck_ch, base_ch * 4, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            # 64 -> 128
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch * 4, base_ch * 2, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            # 128 -> 256
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(base_ch * 2, base_ch, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_ch, 3, kernel_size=7, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, masked_img: torch.Tensor, hole_mask: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([masked_img * (1.0 - hole_mask), hole_mask], dim=1)
        feat = self.encoder(inp)
        x_l, x_g = torch.split(feat, [self.ch_l, self.ch_g], dim=1)
        for blk in self.blocks:
            x_l, x_g = blk((x_l, x_g))
        feat_out = torch.cat([x_l, x_g], dim=1)
        return self.decoder(feat_out)
