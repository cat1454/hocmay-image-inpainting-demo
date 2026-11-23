"""
Kien truc DeepFillv1: Coarse-to-Fine + Contextual Attention
Trich xuat nguyen ban tu Kaggle notebook: https://www.kaggle.com/code/ducminh05/ml-train (Cell 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def conv(i, o, k=3, s=1, d=1):
    return nn.Sequential(
        nn.ReflectionPad2d((k // 2) * d),
        nn.Conv2d(i, o, k, stride=s, dilation=d),
    )


def elu():
    return nn.ELU(inplace=True)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            conv(4, 32, 5), elu(),
            conv(32, 64, 3, 2), elu(), conv(64, 64), elu(),
            conv(64, 128, 3, 2), elu(), conv(128, 128), elu(), conv(128, 128), elu(),
        )

    def forward(self, x):
        return self.net(x)


def middle():
    return nn.Sequential(
        conv(128, 128, d=2), elu(), conv(128, 128, d=4), elu(),
        conv(128, 128, d=8), elu(), conv(128, 128, d=16), elu(),
        conv(128, 128), elu(), conv(128, 128), elu(),
    )


class Decoder(nn.Module):
    def __init__(self, in_c=128):
        super().__init__()
        self.net = nn.Sequential(
            conv(in_c, 128), elu(), conv(128, 128), elu(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv(128, 64), elu(), conv(64, 64), elu(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv(64, 32), elu(), conv(32, 16), elu(), conv(16, 3),
        )

    def forward(self, x):
        return torch.clamp(self.net(x), -1, 1)


class ContextualAttention(nn.Module):
    def __init__(self, scale=10.0, prop_k=3):
        super().__init__()
        self.scale = scale
        self.prop_k = prop_k
        k_size = 2 * prop_k + 1
        self.register_buffer("prop_kernel", torch.ones(1, 1, k_size, k_size))

    def _prop(self, d):
        return F.conv2d(d.unsqueeze(1), self.prop_kernel, padding=self.prop_k).squeeze(1)

    def forward(self, feat, hole):
        h64, hs = F.max_pool2d(hole, 4), F.max_pool2d(hole, 8)
        B, C, H, W = feat.shape
        fs = F.interpolate(feat, scale_factor=0.5, mode="bilinear", align_corners=False)
        bgP = F.unfold(feat, 3, padding=1, stride=2)
        fgP = F.unfold(fs, 3, padding=1)
        known = (1 - h64)[:, 0, ::2, ::2] > 0.5
        out = feat.clone()

        for b in range(B):
            fi = (hs[b, 0] > 0.5).flatten().nonzero().flatten()
            bi = known[b].flatten().nonzero().flatten()
            if fi.numel() == 0 or bi.numel() == 0:
                continue
            f = F.normalize(fgP[b][:, fi].T, dim=1)
            g = F.normalize(bgP[b][:, bi].T, dim=1)
            w = (f @ g.T * self.scale).softmax(dim=1)

            dense = torch.zeros(bi.numel(), 32, 32, device=feat.device, dtype=feat.dtype)
            dense[:, fi // 32, fi % 32] = w.T
            dense = self._prop(dense)
            dense = dense / dense.sum(0, keepdim=True).clamp_min(1e-8)

            rec = dense[:, fi // 32, fi % 32].T @ g
            canvas = torch.zeros(1, C * 9, 32 * 32, device=feat.device, dtype=feat.dtype)
            canvas[0][:, fi] = rec.T
            cnt = F.fold(torch.ones_like(canvas), (32, 32), 3, padding=1).clamp_min(1)
            rm = (F.fold(canvas, (32, 32), 3, padding=1) / cnt)[0].view(C, 32, 32)
            up = F.interpolate(rm[None], scale_factor=2, mode="bilinear", align_corners=False)[0]
            m = h64[b, 0] > 0.5
            out[b][:, m] = up[:, m]
        return out


class DeepFillV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_c, self.mid_c, self.dec_c = Encoder(), middle(), Decoder()
        self.enc_r, self.mid_r = Encoder(), middle()
        self.ca = ContextualAttention()
        self.ca_conv = nn.Sequential(conv(128, 128), elu(), conv(128, 128), elu())
        self.dec_r = Decoder(in_c=256)

    def forward(self, z, m):  # z: masked image [-1, 1] with hole=0; m: 1=valid, 0=hole
        z_in = z * m
        coarse = self.dec_c(self.mid_c(self.enc_c(torch.cat([z_in, m], 1))))
        coarse_comp = z_in + coarse * (1 - m)
        f = self.enc_r(torch.cat([coarse_comp, m], 1))
        top = self.ca_conv(self.ca(f, 1 - m))
        refine = self.dec_r(torch.cat([self.mid_r(f), top], 1))
        return coarse, refine


def critic(chs, size):
    layers, c = [], 3
    for o in chs:
        layers += [conv(c, o, 5, 2), nn.LeakyReLU(0.2, inplace=True)]
        c = o
    layers += [nn.Flatten(), nn.Linear(c * (size // 16) ** 2, 1)]
    return nn.Sequential(*layers)


@torch.no_grad()
def infer_deepfillv1(model: DeepFillV1, gt_hwc: "torch.Tensor | any", hole_hw: "torch.Tensor | any", device: torch.device):
    """
    Helper chay inference DeepFillV1 tren anh gt_hwc [0, 1] (256,256,3) va hole_hw bool (256,256).
    Tra ve (coarse_comp_hwc, refine_comp_hwc) trong [0, 1].
    """
    import numpy as np
    gt = np.asarray(gt_hwc, dtype=np.float32)
    hole = np.asarray(hole_hw, dtype=bool)
    valid_m = (~hole).astype(np.float32)  # 1 = valid, 0 = hole

    # DeepFillV1 trong ml-train dung z thuoc [-1, 1] voi vung hole = 0
    z_np = (gt * 2.0 - 1.0) * valid_m[..., None]
    z_t = torch.from_numpy(np.transpose(z_np, (2, 0, 1))[None]).to(device, dtype=torch.float32)
    m_t = torch.from_numpy(valid_m[None, None]).to(device, dtype=torch.float32)

    coarse, refine = model(z_t, m_t)
    coarse_01 = ((coarse[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 0.5).clip(0.0, 1.0)
    refine_01 = ((refine[0].permute(1, 2, 0).cpu().numpy() + 1.0) * 0.5).clip(0.0, 1.0)

    hole_3c = hole[..., None].astype(np.float32)
    coarse_comp = np.clip(gt * (1.0 - hole_3c) + coarse_01 * hole_3c, 0.0, 1.0)
    refine_comp = np.clip(gt * (1.0 - hole_3c) + refine_01 * hole_3c, 0.0, 1.0)
    return coarse_comp, refine_comp
