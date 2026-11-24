"""Pi3X (yyfz233/Pi3X) depth backend: a permutation-equivariant feed-forward model over an unordered set of views that returns per-view
local point maps (z = depth), continuous confidence and camera poses, and accepts exact intrinsics and camera-to-world poses as conditions.
Same contract as DA3Depth: metric_views(imgs01, cams, anchor_depth, anchor_mask, nominal) -> metric z depth (N, H, W) with last_confidence
(N, H, W) in [0, 1], last_sky (N, H, W) bool and last_mode. Pi3X has no sky head: the sky mask comes from SAM 3.1 (dreamtour.segment._sky_of)
when it can be loaded, else no pixel is sky. Needs the Pi3 repository on sys.path (env PI3_SRC) and the weights yyfz233/Pi3X
(CC BY-NC 4.0: research use)."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

from .depth import camera_baseline_scale, reliable_anchors
from .skymask import SkySegmenter


class Pi3XDepth:
    def __init__(self, device="cuda", repo: str = "yyfz233/Pi3X", process_res: int = 896, sky: bool = True):
        if os.environ.get("PI3_SRC"):
            sys.path.insert(0, os.environ["PI3_SRC"])
        from pi3.models.pi3x import Pi3X

        self.dev, self.res, self.repo = device, int(process_res), repo
        self.model = Pi3X.from_pretrained(repo).to(device).eval().requires_grad_(False)
        self.last_confidence = self.last_sky = self.last_mode = self.last_predictions = None
        self.sky = SkySegmenter(device, enabled=sky)

    def _predict_views(self, imgs01, cams, *, condition_poses=None):
        if (
            not isinstance(imgs01, torch.Tensor)
            or imgs01.ndim != 4
            or imgs01.shape[1] != 3
            or min(imgs01.shape) < 1
            or len(cams) != imgs01.shape[0]
            or any((c.H, c.W) != tuple(imgs01.shape[-2:]) for c in cams)
            or not torch.isfinite(imgs01).all()
        ):
            raise ValueError("Pi3X needs finite RGB views and matching camera dimensions")
        N, _, H, W = imgs01.shape
        imgs01 = imgs01.to(self.dev).float().clamp(0, 1)
        s = self.res / max(H, W)
        h, w = max(14, int(round(H * s / 14)) * 14), max(14, int(round(W * s / 14)) * 14)
        imgs = F.interpolate(imgs01, (h, w), mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)
        sx, sy = w / W, h / H
        K = torch.stack(
            [
                torch.tensor(
                    [[c.fx * sx, 0.0, c.cx * sx], [0.0, c.fy * sy, c.cy * sy], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                )
                for c in cams
            ]
        ).to(self.dev)
        c2w = torch.stack([c.c2w.float().cpu() for c in cams]).to(self.dev)
        t = c2w[:, :3, 3]
        moving = N > 1 and float((t - t[:1]).norm(dim=1).max()) > 1e-3
        posed = moving if condition_poses is None else condition_poses and N > 1
        conditions = dict(
            intrinsics=K[None],
            mask_add_depth=torch.zeros(1, N, dtype=torch.bool, device=imgs.device),
            mask_add_ray=torch.ones(1, N, dtype=torch.bool, device=imgs.device),
            mask_add_pose=torch.full((1, N), posed, dtype=torch.bool, device=imgs.device),
        )
        if posed:
            conditions["poses"] = c2w[None]
        dtype = (
            torch.bfloat16
            if imgs.is_cuda and torch.cuda.get_device_capability(imgs.device)[0] >= 8
            else torch.float16
        )
        with torch.amp.autocast("cuda", dtype=dtype, enabled=imgs.is_cuda):
            r = self.model(imgs[None], **conditions)
        if r["local_points"].shape != (1, N, h, w, 3):
            raise ValueError("Pi3X returned invalid local point dimensions")
        d = r["local_points"][0, ..., 2].float()
        d = F.interpolate(d[:, None], (H, W), mode="bilinear", align_corners=False)[:, 0]
        return r, d, imgs01, moving

    @torch.enable_grad()
    def depth_differentiable(
        self, imgs01: torch.Tensor, cams, *, condition_poses: bool = True
    ) -> torch.Tensor:
        """Raw Pi3X camera-z depth (N, H, W) with input gradients, without scale fitting or confidence weighting."""
        if type(condition_poses) is not bool:
            raise ValueError("Pi3X pose conditioning must be boolean")
        return self._predict_views(imgs01, cams, condition_poses=condition_poses)[1]

    @torch.no_grad()
    def metric_views(
        self, imgs01: torch.Tensor, cams, anchor_depth=None, anchor_mask=None, nominal: float = 3.0
    ) -> torch.Tensor:
        """imgs01 (N, 3, H, W) in [0, 1], cams: N dreamtour Cameras (OpenCV, camera-to-world) -> metric z depth (N, H, W)."""
        N, _, H, W = imgs01.shape
        r, d, imgs01, moving = self._predict_views(imgs01, cams)
        self.last_predictions = {k: v for k, v in r.items() if k in ("camera_poses", "metric")}
        d = d.clamp_min(1e-4)
        conf = torch.sigmoid(r["conf"][0, ..., 0].float())
        conf = F.interpolate(conf[:, None], (H, W), mode="bilinear", align_corners=False)[:, 0]
        anchors = reliable_anchors(anchor_depth, anchor_mask)
        anchored = anchor_depth is not None and anchors is not None and int(anchors.sum()) >= 200
        if anchored:
            # The anchor depth of view 0 fixes the scale (one scale for all views: they share the model's): the model's metric head
            # and its predicted baseline are only approximate, and a small baseline makes the predicted scale unreliable.
            # Scale only when posed, scale + shift in inverse depth when mono.
            x = 1.0 / d[0][anchors].clamp_min(1e-4)
            y = 1.0 / anchor_depth.to(d.device)[anchors].clamp_min(1e-3)
            if moving:
                a = float((x * y).sum() / (x * x).sum().clamp_min(1e-9))
                d = d / max(a, 1e-6)
            else:
                A = torch.stack((x, torch.ones_like(x)), 1)
                sol = torch.linalg.lstsq(A, y[:, None]).solution[:, 0]
                a, b = max(float(sol[0]), 1e-6), float(sol[1])
                d = 1.0 / (a / d + b).clamp_min(1e-3)
            self.last_mode = "multiview-anchored" if moving else "mono-anchored"
        elif moving:
            pred_c2w = r["camera_poses"][0].double().cpu()
            ours = torch.stack([c.w2c.double().cpu() for c in cams])
            d = d * camera_baseline_scale(torch.linalg.inv(pred_c2w), ours)
            self.last_mode = "multiview"
        else:
            d = d * (nominal / float(d[0].median()))
            self.last_mode = "mono"
        finite = torch.isfinite(conf)
        if finite.any():
            lo, hi = conf[finite].quantile(torch.tensor([0.1, 0.9], device=conf.device))
            conf = (
                torch.where(
                    finite, ((conf - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1), torch.zeros_like(conf)
                )
                if float(hi - lo) > 1e-6
                else finite.float() * 0.75
            )
        else:
            conf = torch.full_like(d, 0.75)
        sky = self.sky(imgs01)
        self.last_sky = sky
        self.last_confidence = conf.masked_fill(sky | ~torch.isfinite(d), 0)
        return d
