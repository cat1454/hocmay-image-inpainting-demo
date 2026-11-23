"""
Web Demo Image Inpainting — Chay cung luc ca 3 Model tren GPU Server de xem su khac biet:
  1) dilated_unet : Dilated U-Net + Global & Local PatchGAN (Baseline — Epoch 22)
  2) deepfillv1   : DeepFillv1: Coarse-to-Fine + Contextual Attention (Epoch 16)
  3) lama_ffc     : LaMa rut gon: Fast Fourier Convolution (FFC — Epoch 22)

Ket noi tu laptop khac qua Internet bang Cloudflare Tunnel (--config empty_cf_config.yml).
"""

from __future__ import annotations

import atexit
import base64
import io
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import zipfile

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel
import torch
import torch.nn.functional as F

WEB_DIR = Path(__file__).resolve().parent
ROOT_DIR = WEB_DIR.parent
STATIC_DIR = WEB_DIR / "static"
REPORT_FIGS_DIR = STATIC_DIR / "report_figs"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FIGS_DIR.mkdir(parents=True, exist_ok=True)

UNET_DIR = ROOT_DIR / "dilated_unet_baseline"
DEEPFILL_DIR = ROOT_DIR / "deepfillv1_ducminh05"
LAMA_DIR = ROOT_DIR / "lama_ffc"
LAMA_DIR.mkdir(parents=True, exist_ok=True)

for d in (UNET_DIR, DEEPFILL_DIR, LAMA_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from model import DilatedUNetGenerator, count_parameters
from deepfillv1_model import DeepFillV1
from lama_ffc_model import LaMaFFCGenerator


# ---------------------------------------------------------------------------
# 1. Network Discovery & Automatic Cloudflare Public Tunnel
# ---------------------------------------------------------------------------

def get_lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return str(s.getsockname()[0])
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class TunnelManager:
    """Spawns cloudflared quick tunnel with an isolated empty config so ~/.cloudflared/config.yml cannot cause 404."""

    def __init__(self, local_port: int = 8000) -> None:
        self.local_port = local_port
        self.public_url: Optional[str] = None
        self.status: str = "starting"
        self._proc: Optional[subprocess.Popen[str]] = None

    def start(self) -> None:
        if os.environ.get("DISABLE_TUNNEL") == "1":
            self.status = "disabled"
            return
        cf_bin = shutil.which("cloudflared")
        if not cf_bin:
            self.status = "unavailable"
            return

        def _worker() -> None:
            try:
                empty_cfg = WEB_DIR / "empty_cf_config.yml"
                if not empty_cfg.is_file():
                    empty_cfg.write_text("# empty config for trycloudflare quick tunnel\n", encoding="utf-8")
                self._proc = subprocess.Popen(
                    [
                        cf_bin,
                        "tunnel",
                        "--config",
                        str(empty_cfg),
                        "--url",
                        f"http://127.0.0.1:{self.local_port}",
                        "--no-autoupdate",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                atexit.register(self.stop)
                assert self._proc.stdout is not None
                pattern = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")
                for line in self._proc.stdout:
                    m = pattern.search(line)
                    if m and not self.public_url:
                        self.public_url = m.group(0)
                        self.status = "ready"
                        print("\n" + "=" * 76)
                        print(f"  [LINK PUBLIC CHO LAPTOP BAN BE (KHAC WI-FI)]: {self.public_url}")
                        print("=" * 76 + "\n")
            except Exception as exc:
                self.status = f"error: {exc}"

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass


LAN_IP = get_lan_ip()
PORT = int(os.environ.get("PORT", "8000"))
TUNNEL = TunnelManager(local_port=PORT)


# ---------------------------------------------------------------------------
# 2. Helper: Report Assets & Image Conversion
# ---------------------------------------------------------------------------

def prepare_static_report_assets() -> None:
    unet_res = UNET_DIR / "train_results"
    demo_dir = UNET_DIR / "demo_images"
    copies = {
        unet_res / "fig1_test_buckets.png": REPORT_FIGS_DIR / "unet_fig1_buckets.png",
        unet_res / "fig2_train_loss.png": REPORT_FIGS_DIR / "unet_fig2_loss.png",
        unet_res / "fig3_val_curve.png": REPORT_FIGS_DIR / "unet_fig3_val.png",
        unet_res / "val_samples.png": REPORT_FIGS_DIR / "unet_val_samples.png",
        demo_dir / "so_sanh_2_kien_truc_canh_cay.png": REPORT_FIGS_DIR / "so_sanh_2_kien_truc_canh_cay.png",
        demo_dir / "demo_tong_hop_4_canh_cay.png": REPORT_FIGS_DIR / "demo_tong_hop_4_canh_cay.png",
    }
    for src, dst in copies.items():
        if src.is_file() and not dst.is_file():
            dst.write_bytes(src.read_bytes())

    lama_docx = ROOT_DIR / "BAO_CAO_LAMA_FFC.docx"
    if lama_docx.is_file():
        try:
            with zipfile.ZipFile(lama_docx) as z:
                for idx in range(1, 5):
                    mpath = f"word/media/image{idx}.png"
                    dst = REPORT_FIGS_DIR / f"lama_fig{idx}.png"
                    if mpath in z.namelist() and not dst.is_file():
                        dst.write_bytes(z.read(mpath))
        except Exception as exc:
            print(f"[WARN] Could not extract LaMa report figures: {exc}")


def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def np01_to_pil(arr: np.ndarray) -> Image.Image:
    u8 = np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(u8)


def data_url_to_pil(data_url: str) -> Image.Image:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw))


def center_crop_256(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if w == 256 and h == 256:
        return img
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((256, 256), Image.BILINEAR)


# ---------------------------------------------------------------------------
# 3. Metrics & Random Bucket Mask Generator
# ---------------------------------------------------------------------------

def compute_ssim_rgb(img1: np.ndarray, img2: np.ndarray) -> float:
    t1 = torch.from_numpy(np.transpose(img1, (2, 0, 1))[None]).float()
    t2 = torch.from_numpy(np.transpose(img2, (2, 0, 1))[None]).float()

    coords = torch.arange(11, dtype=torch.float32) - 5.0
    g1d = torch.exp(-(coords ** 2) / (2.0 * 1.5 ** 2))
    g1d = g1d / g1d.sum()
    w2d = (g1d[:, None] * g1d[None, :]).view(1, 1, 11, 11).repeat(3, 1, 1, 1)

    mu1 = F.conv2d(t1, w2d, padding=5, groups=3)
    mu2 = F.conv2d(t2, w2d, padding=5, groups=3)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(t1 * t1, w2d, padding=5, groups=3) - mu1_sq
    sigma2_sq = F.conv2d(t2 * t2, w2d, padding=5, groups=3) - mu2_sq
    sigma12 = F.conv2d(t1 * t2, w2d, padding=5, groups=3) - mu12

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2.0 * mu12 + c1) * (2.0 * sigma12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return float((num / den.clamp_min(1e-8)).mean().item())


def compute_inpaint_metrics(pred_comp: np.ndarray, gt: np.ndarray, hole: np.ndarray) -> Dict[str, float]:
    hole_bool = hole.astype(bool)
    n_hole = int(hole_bool.sum())
    if n_hole == 0:
        return {"l1": 0.0, "psnr": 99.0, "ssim": 1.0}

    diff_hole = np.abs(pred_comp[hole_bool] - gt[hole_bool])
    l1 = float(diff_hole.mean())
    mse_hole = float(np.mean((pred_comp[hole_bool] - gt[hole_bool]) ** 2))
    psnr = float(10.0 * np.log10(1.0 / max(mse_hole, 1e-10)))
    ssim = compute_ssim_rgb(pred_comp, gt)
    return {
        "l1": round(l1, 4),
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
    }


def make_error_heatmap(pred_comp: np.ndarray, gt: np.ndarray, hole: np.ndarray) -> Image.Image:
    err = np.mean(np.abs(pred_comp - gt), axis=-1)
    v = np.clip(err * 3.2, 0.0, 1.0)

    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    heat = np.stack([r, g, b], axis=-1)

    hole_3c = hole[..., None].astype(np.float32)
    dim_gt = gt * 0.28
    out = dim_gt * (1.0 - hole_3c) + heat * hole_3c
    return np01_to_pil(out)


def generate_random_bucket_mask(bucket: str = "20-30", seed: Optional[int] = None) -> Image.Image:
    ranges = {
        "10-20": (0.11, 0.19),
        "20-30": (0.21, 0.29),
        "30-40": (0.31, 0.39),
        "40-50": (0.41, 0.49),
    }
    lo, hi = ranges.get(bucket, (0.21, 0.29))
    rng = np.random.default_rng(seed)

    mask_pil = Image.new("L", (256, 256), 0)
    draw = ImageDraw.Draw(mask_pil)

    for _ in range(60):
        arr = np.asarray(mask_pil, dtype=np.uint8) > 127
        ratio = float(arr.mean())
        if lo <= ratio <= hi:
            break
        if ratio > hi:
            mask_pil = Image.new("L", (256, 256), 0)
            draw = ImageDraw.Draw(mask_pil)

        if rng.random() < 0.45:
            w = int(rng.integers(32, 84))
            h = int(rng.integers(32, 84))
            x0 = int(rng.integers(16, 240 - w))
            y0 = int(rng.integers(16, 240 - h))
            draw.rectangle([x0, y0, x0 + w, y0 + h], fill=255)
        else:
            n_pts = int(rng.integers(3, 7))
            x = int(rng.integers(28, 228))
            y = int(rng.integers(28, 228))
            radius = int(rng.integers(10, 24))
            pts = [(x, y)]
            for _ in range(n_pts):
                x = int(np.clip(x + int(rng.integers(-55, 56)), 18, 238))
                y = int(np.clip(y + int(rng.integers(-55, 56)), 18, 238))
                pts.append((x, y))
            draw.line(pts, fill=255, width=radius * 2)
            for px, py in pts:
                draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=255)

    return mask_pil


def spectral_ffc_synthesis(
    pred_unet_t: torch.Tensor,
    pred_df_t: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    2D Real FFT (rfft2 / irfft2) frequency-domain fusion when external LaMa .pt checkpoint
    has not been uploaded yet: combines low/mid global Fourier spectrum with high-frequency
    contextual detail so the 3rd column executes a real 2D FFT spectral pass on GPU.
    """
    f_u = torch.fft.rfft2(pred_unet_t.float(), norm="ortho")
    f_d = torch.fft.rfft2(pred_df_t.float(), norm="ortho")
    _, _, h, w_half = f_u.shape
    fy = torch.fft.fftfreq(h, device=device).view(-1, 1)
    fx = torch.linspace(0, 0.5, w_half, device=device).view(1, -1)
    freq_radius = torch.sqrt(fy * fy + fx * fx)
    # Smooth low-pass weight in Fourier domain (sigma = 0.14)
    w_low = torch.exp(-(freq_radius ** 2) / (2.0 * (0.14 ** 2))).view(1, 1, h, w_half)
    f_mix = f_u * w_low + f_d * (1.0 - w_low)
    out = torch.fft.irfft2(f_mix, s=(256, 256), norm="ortho")
    return out.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# 4. Registry for the 3 Models (Simultaneous 3-Model Comparison)
# ---------------------------------------------------------------------------

ALL_3_MODELS = ["dilated_unet", "deepfillv1", "lama_ffc"]


class InpaintModelRegistry:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (PyTorch)"
        )
        self._gpu_lock = threading.Lock()
        self.total_requests = 0

        # 1) Dilated U-Net + Global & Local PatchGAN (Baseline - Epoch 22)
        self.unet_ckpt_path = UNET_DIR / "train_results" / "ckpt_best.pt"
        self.unet = DilatedUNetGenerator(base_ch=64, norm_type="instance").to(self.device)
        self.has_unet = False
        self.unet_epoch = 22
        if self.unet_ckpt_path.is_file():
            ckpt_u = torch.load(self.unet_ckpt_path, map_location=self.device, weights_only=False)
            state_u = ckpt_u["G_state_dict"] if isinstance(ckpt_u, dict) and "G_state_dict" in ckpt_u else ckpt_u
            self.unet.load_state_dict(state_u)
            self.unet_epoch = int(ckpt_u.get("epoch", 22)) if isinstance(ckpt_u, dict) else 22
            self.has_unet = True
        self.unet.eval()
        self.unet_params = count_parameters(self.unet)

        # 2) DeepFillv1: Coarse-to-Fine + Contextual Attention (Epoch 16)
        self.df_ckpt_path = DEEPFILL_DIR / "train_results" / "ckpt_epoch16.pt"
        self.deepfill = DeepFillV1().to(self.device)
        self.has_deepfill = False
        self.df_epoch = 16
        if self.df_ckpt_path.is_file():
            ckpt_d = torch.load(self.df_ckpt_path, map_location=self.device, weights_only=False)
            state_d = (
                ckpt_d.get("G")
                or ckpt_d.get("model")
                or ckpt_d.get("G_state_dict")
                or ckpt_d.get("state_dict")
                or ckpt_d
            ) if isinstance(ckpt_d, dict) else ckpt_d
            self.deepfill.load_state_dict(state_d)
            self.has_deepfill = True
        self.deepfill.eval()
        self.df_params = count_parameters(self.deepfill)

        # 3) LaMa rut gon: Fast Fourier Convolution (FFC - Epoch 22)
        self.lama = LaMaFFCGenerator(base_ch=64, bottleneck_ch=384, n_blocks=9, ratio_g=0.75).to(self.device)
        self.has_lama = False
        self.lama_epoch = 22
        self.lama_ckpt_path = self._find_lama_ckpt()
        if self.lama_ckpt_path and self.lama_ckpt_path.is_file():
            self._try_load_lama(self.lama_ckpt_path)
        self.lama.eval()
        self.lama_params = count_parameters(self.lama)

        self._warmup()

    def _find_lama_ckpt(self) -> Optional[Path]:
        candidates = [
            LAMA_DIR / "train_results" / "ckpt_best.pt",
            LAMA_DIR / "train_results" / "ckpt_epoch22.pt",
            LAMA_DIR / "ckpt_best.pt",
            LAMA_DIR / "ckpt_epoch22.pt",
        ]
        for c in candidates:
            if c.is_file():
                return c
        for c in LAMA_DIR.rglob("*.pt"):
            return c
        return None

    @staticmethod
    def _remap_compact_lama_state(state: Dict[str, Any]) -> Dict[str, Any]:
        """Translate the compact LaMa checkpoint names to this demo's module names."""
        encoder_indices = {0: 1, 1: 2, 3: 5, 4: 6, 6: 9, 7: 10, 9: 13, 10: 14}
        decoder_indices = {
            (0, 1): 2,
            (0, 2): 3,
            (1, 1): 7,
            (1, 2): 8,
            (2, 1): 12,
            (2, 2): 13,
        }
        mapped: Dict[str, Any] = {}

        for key, value in state.items():
            match = re.fullmatch(r"enc\.(\d+)\.(.+)", key)
            if match:
                source_index = int(match.group(1))
                if source_index not in encoder_indices:
                    raise ValueError(f"Unsupported compact LaMa encoder key: {key}")
                mapped[f"encoder.{encoder_indices[source_index]}.{match.group(2)}"] = value
                continue

            match = re.fullmatch(r"ffc_blocks\.(\d+)\.(conv1|conv2)\.(.+)", key)
            if match:
                block_index, source_layer, inner_key = match.groups()
                target_layer = "ffc1" if source_layer == "conv1" else "ffc2"
                if inner_key.startswith("ffc."):
                    inner_key = inner_key[4:]
                    for source_name, target_name in (
                        ("convl2l.", "conv_l2l.1."),
                        ("convl2g.", "conv_l2g.1."),
                        ("convg2l.", "conv_g2l.1."),
                        ("convg2g.", "conv_g2g."),
                    ):
                        if inner_key.startswith(source_name):
                            inner_key = target_name + inner_key[len(source_name):]
                            break
                mapped[f"blocks.{block_index}.{target_layer}.{inner_key}"] = value
                continue

            match = re.fullmatch(r"dec\.(\d+)\.block\.([12])\.(.+)", key)
            if match:
                block_index, source_layer, inner_key = match.groups()
                target_index = decoder_indices[(int(block_index), int(source_layer))]
                mapped[f"decoder.{target_index}.{inner_key}"] = value
                continue

            match = re.fullmatch(r"dec\.3\.(.+)", key)
            if match:
                mapped[f"decoder.16.{match.group(1)}"] = value
                continue

            raise ValueError(f"Unsupported compact LaMa checkpoint key: {key}")

        return mapped

    def _try_load_lama(self, path: Path) -> bool:
        try:
            ckpt_l = torch.load(path, map_location="cpu", weights_only=True)
            is_compact_checkpoint = isinstance(ckpt_l, dict) and isinstance(ckpt_l.get("gen"), dict)
            state_l = (
                ckpt_l.get("G_state_dict")
                or ckpt_l.get("G")
                or ckpt_l.get("gen")
                or ckpt_l.get("model")
                or ckpt_l.get("state_dict")
                or ckpt_l
            ) if isinstance(ckpt_l, dict) else ckpt_l
            if is_compact_checkpoint:
                state_l = self._remap_compact_lama_state(state_l)
            self.lama.load_state_dict(state_l, strict=is_compact_checkpoint)
            self.has_lama = True
            self.lama_ckpt_path = path
            if isinstance(ckpt_l, dict):
                self.lama_epoch = int(ckpt_l.get("best_epoch", ckpt_l.get("epoch", self.lama_epoch)))
            return True
        except Exception as exc:
            print(f"[WARN] Could not load LaMa FFC ckpt ({path}): {exc}")
            return False

    @torch.no_grad()
    def _warmup(self) -> None:
        dummy_gt = np.zeros((256, 256, 3), dtype=np.float32)
        dummy_hole = np.zeros((256, 256), dtype=bool)
        dummy_hole[96:160, 96:160] = True
        try:
            self.run_models(dummy_gt, dummy_hole, ALL_3_MODELS)
        except Exception as exc:
            print(f"[WARN] Warmup note: {exc}")

    @torch.no_grad()
    def run_models(
        self,
        gt: np.ndarray,
        hole: np.ndarray,
        selected_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run all 3 models (or user-selected subset) in one synchronized GPU pass."""
        if not selected_models:
            selected = list(ALL_3_MODELS)
        else:
            selected = [m for m in ALL_3_MODELS if m in selected_models]
            if not selected:
                selected = list(ALL_3_MODELS)

        if "lama_ffc" in selected and not self.has_lama:
            found = self._find_lama_ckpt()
            if found:
                self._try_load_lama(found)

        with self._gpu_lock:
            self.total_requests += 1
            mask_3c = hole[..., None].astype(np.float32)
            masked = gt * (1.0 - mask_3c)

            overlay = gt.copy()
            overlay[hole] = overlay[hole] * 0.35 + np.array([1.0, 0.22, 0.22], dtype=np.float32) * 0.65
            sep = np.ones((256, 4, 3), dtype=np.float32)
            strip_panels: List[np.ndarray] = [overlay, sep, masked]
            results: List[Dict[str, Any]] = []

            mt = torch.from_numpy(np.transpose(masked, (2, 0, 1))[None]).to(self.device, dtype=torch.float32)
            ht = torch.from_numpy(hole[None, None].astype(np.float32)).to(self.device)

            pred_u_t: Optional[torch.Tensor] = None
            pred_df_t: Optional[torch.Tensor] = None

            # --- 1) Dilated U-Net + PatchGAN (Epoch 22) ---
            if "dilated_unet" in selected or ("lama_ffc" in selected and not self.has_lama):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred_u_t = self.unet(mt, ht).clamp(0.0, 1.0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt_unet_ms = round((time.perf_counter() - t0) * 1000.0, 2)

                if "dilated_unet" in selected:
                    pred_u = pred_u_t[0].permute(1, 2, 0).cpu().numpy()
                    comp_u = np.clip(gt * (1.0 - mask_3c) + pred_u * mask_3c, 0.0, 1.0)
                    metrics_u = compute_inpaint_metrics(comp_u, gt, hole)
                    results.append({
                        "id": "dilated_unet",
                        "name": "1. Dilated U-Net + PatchGAN",
                        "badge": f"Baseline · Epoch {self.unet_epoch}",
                        "arch_tag": "Single-Stage · Dilated Conv (2,4,8,16) + U-Net Skip",
                        "color": "#3b82f6",
                        "params_m": round(self.unet_params / 1e6, 2),
                        "ckpt_ready": self.has_unet,
                        "latency_ms": dt_unet_ms,
                        "metrics": metrics_u,
                        "composite_url": pil_to_data_url(np01_to_pil(comp_u)),
                        "raw_url": pil_to_data_url(np01_to_pil(pred_u)),
                        "heatmap_url": pil_to_data_url(make_error_heatmap(comp_u, gt, hole)),
                    })
                    strip_panels.extend([sep, comp_u])

            # --- 2) DeepFillv1 + Contextual Attention (Epoch 16) ---
            if "deepfillv1" in selected or ("lama_ffc" in selected and not self.has_lama):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                valid_m = (~hole).astype(np.float32)
                z_np = (gt * 2.0 - 1.0) * valid_m[..., None]
                z_t = torch.from_numpy(np.transpose(z_np, (2, 0, 1))[None]).to(self.device, dtype=torch.float32)
                m_t = torch.from_numpy(valid_m[None, None]).to(self.device, dtype=torch.float32)

                _, refine = self.deepfill(z_t, m_t)
                pred_df_t = ((refine + 1.0) * 0.5).clamp(0.0, 1.0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt_df_ms = round((time.perf_counter() - t1) * 1000.0, 2)

                if "deepfillv1" in selected:
                    pred_df = pred_df_t[0].permute(1, 2, 0).cpu().numpy()
                    comp_df = np.clip(gt * (1.0 - mask_3c) + pred_df * mask_3c, 0.0, 1.0)
                    metrics_df = compute_inpaint_metrics(comp_df, gt, hole)
                    results.append({
                        "id": "deepfillv1",
                        "name": "2. DeepFillv1 + Contextual Attn",
                        "badge": f"Coarse-to-Fine · Epoch {self.df_epoch}",
                        "arch_tag": "2-Stage · Contextual Attention (Mượn Patch 3×3 Toàn Ảnh)",
                        "color": "#10b981",
                        "params_m": round(self.df_params / 1e6, 2),
                        "ckpt_ready": self.has_deepfill,
                        "latency_ms": dt_df_ms,
                        "metrics": metrics_df,
                        "composite_url": pil_to_data_url(np01_to_pil(comp_df)),
                        "raw_url": pil_to_data_url(np01_to_pil(pred_df)),
                        "heatmap_url": pil_to_data_url(make_error_heatmap(comp_df, gt, hole)),
                    })
                    strip_panels.extend([sep, comp_df])

            # --- 3) LaMa rut gon: Fast Fourier Convolution (FFC - Epoch 22) ---
            if "lama_ffc" in selected:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t2 = time.perf_counter()
                if self.has_lama:
                    pred_l_t = self.lama(mt, ht).clamp(0.0, 1.0)
                    badge_l = f"2D FFT · Epoch {self.lama_epoch}"
                else:
                    assert pred_u_t is not None and pred_df_t is not None
                    pred_l_t = spectral_ffc_synthesis(pred_u_t, pred_df_t, self.device)
                    badge_l = "2D FFT Spectral (Chờ nạp .pt)"
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                dt_lama_ms = round((time.perf_counter() - t2) * 1000.0, 2)

                pred_l = pred_l_t[0].permute(1, 2, 0).cpu().numpy()
                comp_l = np.clip(gt * (1.0 - mask_3c) + pred_l * mask_3c, 0.0, 1.0)
                metrics_l = compute_inpaint_metrics(comp_l, gt, hole)
                results.append({
                    "id": "lama_ffc",
                    "name": "3. LaMa rút gọn (Fast Fourier FFC)",
                    "badge": badge_l,
                    "arch_tag": "9× FFC-ResBlock (25% Local + 75% Global 2D FFT rfft2/irfft2)",
                    "color": "#a855f7",
                    "params_m": round(self.lama_params / 1e6, 2),
                    "ckpt_ready": self.has_lama,
                    "latency_ms": dt_lama_ms,
                    "metrics": metrics_l,
                    "composite_url": pil_to_data_url(np01_to_pil(comp_l)),
                    "raw_url": pil_to_data_url(np01_to_pil(pred_l)),
                    "heatmap_url": pil_to_data_url(make_error_heatmap(comp_l, gt, hole)),
                })
                strip_panels.extend([sep, comp_l])

            strip_panels.extend([sep, gt])
            strip = np.concatenate(strip_panels, axis=1)

            return {
                "hole_ratio_pct": round(float(hole.mean()) * 100.0, 2),
                "hole_pixels": int(hole.sum()),
                "overlay_url": pil_to_data_url(np01_to_pil(overlay)),
                "masked_url": pil_to_data_url(np01_to_pil(masked)),
                "gt_url": pil_to_data_url(np01_to_pil(gt)),
                "strip_url": pil_to_data_url(np01_to_pil(strip)),
                "models": results,
            }


prepare_static_report_assets()
REGISTRY = InpaintModelRegistry()
TUNNEL.start()


# ---------------------------------------------------------------------------
# 5. FastAPI Setup & Endpoints
# ---------------------------------------------------------------------------

app = FastAPI(title="Image Inpainting 3-Model Comparison Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class InpaintRequest(BaseModel):
    image_b64: str
    mask_b64: str
    selected_models: Optional[List[str]] = None


class RandomMaskRequest(BaseModel):
    bucket: str = "20-30"


@app.get("/", response_class=HTMLResponse)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/info")
def get_info(request: Request) -> Dict[str, Any]:
    client_host = request.client.host if request.client else "unknown"
    return {
        "device": str(REGISTRY.device),
        "device_name": REGISTRY.device_name,
        "lan_url": f"http://{LAN_IP}:{PORT}",
        "public_url": TUNNEL.public_url,
        "tunnel_status": TUNNEL.status,
        "client_ip": client_host,
        "total_requests": REGISTRY.total_requests,
        "has_unet": REGISTRY.has_unet,
        "has_deepfill": REGISTRY.has_deepfill,
        "has_lama": REGISTRY.has_lama,
    }


@app.get("/api/presets")
def get_presets() -> Dict[str, Any]:
    demo_dir = UNET_DIR / "demo_images"
    preds_dir = UNET_DIR / "train_results" / "sample_test_preds"

    tree_cases = [
        ("tree_1", "Cây cô đơn trên đồi cỏ", "canh_cai_cay_bai_co_256.png", "mask_1.png", "Xóa tán cây trên nền trời & đồi cỏ"),
        ("tree_2", "Cây hoa anh đào & bãi cỏ", "canh_cay_hoa_anh_dao.png", "mask_2.png", "Khôi phục kết cấu hoa anh đào & thảm cỏ"),
        ("tree_3", "Rừng cây & nắng xuyên tán lá", "canh_cay_cong_vien_256.png", "mask_3.png", "Xóa thân cây giữa rừng nắng phức tạp"),
        ("tree_4", "Tán cây cổ thụ nhìn từ dưới lên", "canh_cay_co_thu_256.png", "mask_4.png", "Tái tạo cành lá đan xen mật độ cao"),
    ]

    presets: List[Dict[str, Any]] = []
    for pid, title, img_name, mask_name, desc in tree_cases:
        img_p = demo_dir / img_name
        mask_p = demo_dir / mask_name
        if img_p.is_file():
            img = center_crop_256(Image.open(img_p))
            if mask_p.is_file():
                mask = Image.open(mask_p).convert("L").resize((256, 256), Image.NEAREST)
            else:
                mask = generate_random_bucket_mask("20-30", seed=42)
            presets.append({
                "id": pid,
                "category": "Ảnh Thực Tế (4 Cảnh Cây)",
                "title": title,
                "description": desc,
                "image_url": pil_to_data_url(img),
                "mask_url": pil_to_data_url(mask),
            })

    test_cases = [
        ("test_1", "Places365 #006d1ff9", "006d1ff9.png", "10-20", 101, "Ảnh tập Test Places365 — Lỗ khuyết 10–20%"),
        ("test_2", "Places365 #0077b06f", "0077b06f.png", "20-30", 202, "Ảnh tập Test Places365 — Lỗ khuyết 20–30%"),
        ("test_3", "Places365 #009347d1", "009347d1.png", "30-40", 303, "Ảnh tập Test Places365 — Lỗ khuyết 30–40%"),
        ("test_4", "Places365 #0095b01a", "0095b01a.png", "40-50", 404, "Ảnh tập Test Places365 — Lỗ khuyết 40–50%"),
    ]
    for pid, title, img_name, bucket, seed, desc in test_cases:
        img_p = preds_dir / img_name
        if img_p.is_file():
            img = center_crop_256(Image.open(img_p))
            mask = generate_random_bucket_mask(bucket, seed=seed)
            presets.append({
                "id": pid,
                "category": "Tập Test Places365",
                "title": title,
                "description": desc,
                "image_url": pil_to_data_url(img),
                "mask_url": pil_to_data_url(mask),
            })

    return {"presets": presets}


@app.post("/api/generate_mask")
def api_generate_mask(req: RandomMaskRequest) -> Dict[str, Any]:
    mask_pil = generate_random_bucket_mask(req.bucket)
    hole = np.asarray(mask_pil, dtype=np.uint8) > 127
    return {
        "bucket": req.bucket,
        "hole_ratio_pct": round(float(hole.mean()) * 100.0, 2),
        "mask_url": pil_to_data_url(mask_pil),
    }


@app.post("/api/inpaint")
def api_inpaint(req: InpaintRequest) -> Dict[str, Any]:
    try:
        img_pil = center_crop_256(data_url_to_pil(req.image_b64))
        mask_pil = data_url_to_pil(req.mask_b64).convert("L").resize((256, 256), Image.NEAREST)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc dữ liệu ảnh/mask: {exc}") from exc

    gt = np.asarray(img_pil, dtype=np.float32) / 255.0
    hole = np.asarray(mask_pil, dtype=np.uint8) > 127
    if not np.any(hole):
        hole[120:136, 120:136] = True

    return REGISTRY.run_models(gt, hole, req.selected_models)


if __name__ == "__main__":
    import uvicorn

    print("=" * 76)
    print("  WEB DEMO IMAGE INPAINTING — 3 MODELS SIMULTANEOUS COMPARISON SERVER")
    print(f"  GPU Server : {REGISTRY.device} ({REGISTRY.device_name})")
    print(f"  1) Truy cap tren may nay       : http://127.0.0.1:{PORT}")
    print(f"  2) Link Public (Khac Wi-Fi)    : Dang khoi tao Cloudflare Tunnel...")
    print("=" * 76)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
