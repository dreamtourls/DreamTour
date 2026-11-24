"""Depth backends with a shared metric-alignment contract. MonoDepth: Depth Anything V2 (small, Apache 2.0) relative depth, turned metric by a
least-squares fit of scale and shift in inverse depth against an anchor depth where one is given (else the median is set to a nominal distance).
DA3Depth: Depth Anything 3 multi-view depth with the exact cameras. Helpers: camera_baseline_scale, reliable_anchors.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pose_scale(extrinsics: torch.Tensor) -> torch.Tensor:
    """Use the same lower median and clamp as DA3's extrinsics normalization (float64, outside autocast: linalg.inv has no bf16 kernel)."""
    with torch.autocast(extrinsics.device.type, enabled=False):
        ex = extrinsics.detach().double()
        relative = ex @ torch.linalg.inv(ex[:, :1])
        centers = torch.linalg.inv(relative)[..., :3, 3]
        return centers.norm(dim=-1).median().clamp_min(0.1)


def camera_baseline_scale(predicted_w2c, known_w2c) -> float:
    """Align depth scale from camera distances, including two-view or collinear histories."""
    centers = []
    for poses in (predicted_w2c, known_w2c):
        poses = torch.as_tensor(poses, dtype=torch.float64)
        if poses.ndim != 3 or poses.shape[-2:] not in ((3, 4), (4, 4)):
            raise ValueError("Expected one 3x4 or 4x4 world-to-camera matrix per image")
        if poses.shape[-2] == 3:
            row = poses.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(len(poses), 1, 4)
            poses = torch.cat((poses, row), dim=1)
        centers.append(torch.linalg.inv(poses)[:, :3, 3])
    if len(centers[0]) != len(centers[1]):
        raise ValueError("Predicted and supplied cameras must correspond")
    predicted, known = (torch.pdist(value) for value in centers)
    valid = torch.isfinite(predicted) & torch.isfinite(known) & (predicted > 1e-8) & (known > 1e-8)
    if not valid.any():
        raise RuntimeError("No translated camera baseline available for depth scale")
    return float((known[valid] / predicted[valid]).median())


def reliable_anchors(depth, mask):
    if depth is None or mask is None:
        return None
    valid = mask & torch.isfinite(depth) & (depth > 0)
    d = torch.where(valid, depth, torch.zeros_like(depth))
    near = -F.max_pool2d(-torch.where(valid, d, torch.full_like(d, float("inf")))[None, None], 3, 1, 1)[0, 0]
    far = F.max_pool2d(d[None, None], 3, 1, 1)[0, 0]
    interior = -F.max_pool2d(-valid.float()[None, None], 3, 1, 1)[0, 0] > 0.5
    return valid & interior & ((far - near) < 0.1 * d)


class MonoDepth:
    def __init__(self, device="cuda", model="depth-anything/Depth-Anything-V2-Small-hf"):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.dev = device
        self.proc = AutoImageProcessor.from_pretrained(model)
        self.net = AutoModelForDepthEstimation.from_pretrained(model).to(device).eval()

    @torch.no_grad()
    def relative_inverse(self, img01: torch.Tensor) -> torch.Tensor:
        """(3, H, W) in [0, 1] -> relative inverse depth (H, W) (larger = nearer), resized to the image."""
        import numpy as np
        from PIL import Image

        pil = Image.fromarray((img01.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        inp = self.proc(images=pil, return_tensors="pt").to(self.dev)
        out = self.net(**inp).predicted_depth[None]
        return (
            F.interpolate(out, size=img01.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            .float()
            .clamp_min(1e-6)
        )

    def metric(
        self,
        img01: torch.Tensor,
        anchor_depth: torch.Tensor | None,
        anchor_mask: torch.Tensor | None,
        nominal: float = 3.0,
    ) -> torch.Tensor:
        """Metric depth (H, W): fit inv_metric = a * rel + b on anchor pixels (>= 200 of them) else map the median relative depth to ``nominal`` metres."""
        rel = self.relative_inverse(img01)
        anchor_mask = reliable_anchors(anchor_depth, anchor_mask)
        if anchor_depth is not None and anchor_mask is not None and int(anchor_mask.sum()) >= 200:
            x = rel[anchor_mask]
            y = 1.0 / anchor_depth[anchor_mask].clamp_min(1e-3)
            A = torch.stack((x, torch.ones_like(x)), 1)
            sol = torch.linalg.lstsq(A, y[:, None]).solution[:, 0]
            a, b = float(sol[0]), float(sol[1])
            a = max(a, 1e-6)
        else:
            a, b = (1.0 / nominal) / float(rel.median()), 0.0
        result = 1.0 / (a * rel + b).clamp_min(1e-3)
        self.last_confidence = torch.isfinite(result).float() * 0.75
        self.last_sky = torch.zeros_like(result, dtype=torch.bool)
        return result


class DA3Depth:
    """Depth Anything 3 (nested giant) with the window's EXACT cameras: one multi-view inference per window gives metric, cross-view-consistent depth
    (DA3 aligns its predicted camera baseline to the supplied translations; pure rotation has no translation scale -> monocular fallback with
    the canvas / nominal alignment). Needs the Depth-Anything-3 source on sys.path (env DA3_SRC) and the HF weights depth-anything/DA3NESTED-GIANT-LARGE.
    """

    def __init__(self, device="cuda", model=None, process_res: int = 504):
        import os
        import sys

        if os.environ.get("DA3_SRC"):
            sys.path.insert(0, os.environ["DA3_SRC"])
        model = model or os.environ.get(
            "DA3_MODEL", "depth-anything/DA3NESTED-GIANT-LARGE"
        )  # env: a local snapshot or another DA3 checkpoint
        from depth_anything_3.api import DepthAnything3

        self.dev, self.res, self.repo = device, int(process_res), model
        self.model = DepthAnything3.from_pretrained(model).to(device).eval()

    @torch.no_grad()
    def _run(self, frames, w2c=None, Ks=None):
        """Follow DA3's normalization and align output camera scale, retaining raw sky predictions.

        The nested metric head rescales depth and predicted translations together.
        Its output needs camera-baseline alignment, not another input-normalization scale.
        """
        m = self.model
        posed = w2c is not None
        imgs_cpu, ex, ins = m._preprocess_inputs(frames, w2c, Ks, self.res, "upper_bound_resize")
        imgs, ex_t, in_t = m._prepare_model_inputs(imgs_cpu, ex, ins)
        raw = m._run_model_forward(
            imgs,
            m._normalize_extrinsics(ex_t.clone()) if posed else None,
            in_t if posed else None,
            [],
            False,
            False,
            "first" if posed else "saddle_balanced",
        )
        pred = m._convert_to_prediction(raw)
        if posed:
            pred.depth *= camera_baseline_scale(pred.extrinsics, w2c)
        sky = raw["sky"] if "sky" in raw else None
        if sky is not None:
            sky = torch.as_tensor(sky).float()
            while sky.ndim > 3:
                sky = sky.squeeze(1)
            pred.sky = (sky >= 0.3).cpu().numpy()  # DA3's own threshold (compute_sky_mask)
        else:
            pred.sky = None
        return pred

    @torch.no_grad()
    def metric_views(
        self, imgs01: torch.Tensor, cams, anchor_depth=None, anchor_mask=None, nominal: float = 3.0
    ) -> torch.Tensor:
        """imgs01 (N, 3, H, W) in [0, 1], cams: N dreamtour Cameras (OpenCV, cam-to-world) -> metric z depth (N, H, W).
        Poses are passed as world-to-camera 3x4 (+ intrinsics); if the translations are degenerate (pan) the monocular path is used and the result is
        aligned to the canvas depth of view 0 where covered (scale + shift in inverse depth), else to ``nominal`` metres median.
        """
        import numpy as np

        N, _, H, W = imgs01.shape
        frames = [(im.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8) for im in imgs01]
        w2c = np.stack([c.w2c.double().cpu().numpy() for c in cams]).astype(np.float32)
        Ks = np.stack([np.array([[c.fx, 0, c.cx], [0, c.fy, c.cy], [0, 0, 1]], np.float32) for c in cams])
        t = np.stack([c.c2w[:3, 3].cpu().numpy() for c in cams])
        moving = float(np.linalg.norm(t - t[:1], axis=1).max()) > 1e-3
        with torch.autocast(torch.device(self.dev).type, dtype=torch.bfloat16):
            try:
                if not moving:
                    raise RuntimeError("no translation: monocular")
                pred = self._run(frames, w2c, Ks)
                mono = False
            except Exception as e:
                if moving and not getattr(self, "_warned", False):
                    print(f"[DA3] multi-view path failed ({str(e)[:160]}); monocular fallback", flush=True)
                    self._warned = True
                pred = self._run(frames)
                mono = True
        self.last_mode = "mono" if mono else "multiview"
        d = torch.from_numpy(np.asarray(pred.depth, np.float32)).to(self.dev)
        d = F.interpolate(d[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0].clamp_min(1e-3)
        raw_conf = getattr(pred, "conf", None)
        if raw_conf is None:
            confidence = torch.full_like(d, 0.75)
        else:
            confidence = torch.as_tensor(np.asarray(raw_conf, np.float32), device=self.dev)
            confidence = F.interpolate(confidence[:, None], (H, W), mode="bilinear", align_corners=False)[
                :, 0
            ]
            finite = torch.isfinite(confidence)
            bounds = (
                confidence[finite].quantile(torch.tensor([0.1, 0.9], device=self.dev))
                if finite.any()
                else confidence.new_tensor([0, 1])
            )
            confidence = torch.where(
                finite, ((confidence - bounds[0]) / (bounds[1] - bounds[0]).clamp_min(1e-6)).clamp(0, 1), 0
            )
            if float(bounds[1] - bounds[0]) < 1e-6:
                confidence = finite.float() * 0.75
        sky = getattr(pred, "sky", None)
        self.last_sky = (
            torch.zeros_like(d, dtype=torch.bool)
            if sky is None
            else F.interpolate(
                torch.as_tensor(np.asarray(sky, np.float32), device=self.dev)[:, None], (H, W), mode="nearest"
            )[:, 0]
            > 0.5
        )
        self.last_confidence = confidence.masked_fill(self.last_sky | ~torch.isfinite(d), 0)
        anchor_mask = reliable_anchors(anchor_depth, anchor_mask)
        if anchor_mask is not None:
            anchor_mask &= (self.last_confidence[0] >= 0.5) & torch.isfinite(d[0]) & (d[0] > 0)
        ok = anchor_depth is not None and anchor_mask is not None and int(anchor_mask.sum()) >= 500
        usable = torch.isfinite(d[0]) & (d[0] > 0) & ~self.last_sky[0]
        median = float(d[0][usable].median()) if usable.any() else nominal
        if ok:
            ad = anchor_depth[anchor_mask].clamp_min(1e-3)
            q = ad.quantile(torch.tensor([0.1, 0.9], device=self.dev))
            wide = float(q[1] / q[0]) >= 1.5  # anchors span enough depth for a 2-parameter fit
        if (
            not mono
        ):  # posed multi-view depth is metric up to ONE global factor (canvas units per metre): robust median over anchors, remembered across calls (EMA), never a shift
            if ok:
                sc = float((ad / d[0][anchor_mask]).median())
                self.scale = sc if getattr(self, "scale", None) is None else 0.7 * self.scale + 0.3 * sc
            if getattr(self, "scale", None) is None:
                self.scale = 1.0  # posed depth already uses the supplied translations' units
            d = d * self.scale
        else:  # monocular: scale + shift in inverse depth against the canvas when the anchors are well conditioned, scale only when narrow, nominal median otherwise
            inv = 1.0 / d
            if ok and wide:
                x = inv[0][anchor_mask]
                y = 1.0 / ad
                A = torch.stack((x, torch.ones_like(x)), 1)
                sol = torch.linalg.lstsq(A, y[:, None]).solution[:, 0]
                a, b = max(float(sol[0]), 1e-6), float(sol[1])
            elif ok:
                a, b = float((1.0 / ad).median() / inv[0][anchor_mask].median()), 0.0
            else:
                a, b = median / nominal, 0.0
            d = 1.0 / (a * inv + b).clamp_min(1e-3)
        return d
