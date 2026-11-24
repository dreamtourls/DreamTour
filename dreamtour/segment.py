"""Open-vocabulary concept segmentation (SAM 3 / SAM 3.1) -> per-pixel surface class and instance id.
Classes: OTHER (unknown), SOLID (cars, houses, road, ...: a single opaque surface per instance), VEGETATION (foliage: many thin layers, lenient),
SKY. The canvas uses the class to decide how a point's surface patch may be extended (solid instances keep their full patch across internal depth
jumps, so a roof or hood sampled edge-on tiles the image when later seen from above) and which under-resolved surfaces shadow what is behind them.
Needs the SAM 3 source on sys.path (env SAM3_SRC) and the gated HF checkpoint facebook/sam3.1 (or facebook/sam3).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

OTHER, SOLID, VEGETATION, SKY = 0, 1, 2, 3
VOCAB = {
    SOLID: ("car", "truck", "bus", "house", "building", "road", "sidewalk", "fence", "wall"),
    VEGETATION: ("tree", "bush", "hedge", "grass"),
    SKY: ("sky",),
}


class Segmenter:
    def __init__(self, device="cuda", version: str = "sam3.1", threshold: float = 0.5):
        if os.environ.get("SAM3_SRC"):
            sys.path.insert(0, os.environ["SAM3_SRC"])
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf

        self.dev = device
        tf32 = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )  # the SAM 3 builder switches TF32 on globally; the canvas pipeline must keep its own numerics
        self.model = build_sam3_image_model(
            checkpoint_path=download_ckpt_from_hf(version=version), load_from_HF=False, device=device
        ).eval()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
        self.proc = Sam3Processor(self.model, device=device, confidence_threshold=threshold)
        self.calls = 0

    @torch.inference_mode()
    def __call__(self, image: torch.Tensor):
        """image (3, H, W) in [0, 1] -> (cls (H, W) uint8, inst (H, W) long; -1 = no instance). Smaller masks overwrite larger ones."""
        H, W = image.shape[-2:]
        pil = Image.fromarray(
            (image.permute(1, 2, 0).clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)
        )
        found, vehicles = [], []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.proc.set_image(pil)
            for cls, names in VOCAB.items():
                for name in names:
                    out = self.proc.set_text_prompt(prompt=name, state=state)
                    masks = out["masks"]
                    masks = masks[:, 0] if masks.ndim == 4 else masks
                    for k in range(masks.shape[0]):
                        mask = masks[k].bool()
                        found.append((cls, mask))
                        if name in ("car", "truck", "bus"):
                            vehicles.append(mask)
        self.calls += 1
        cls_map = torch.zeros(H, W, dtype=torch.uint8, device=self.dev)
        inst = torch.full((H, W), -1, dtype=torch.long, device=self.dev)
        found.sort(key=lambda t: -int(t[1].sum()))
        for i, (cls, m) in enumerate(found):
            m = m.to(self.dev)
            if m.shape != (H, W):
                m = F.interpolate(m[None, None].float(), (H, W), mode="nearest")[0, 0].bool()
            cls_map[m] = cls
            inst[m] = i
        resized = []
        for mask in vehicles:
            mask = mask.to(self.dev)
            if mask.shape != (H, W):
                mask = F.interpolate(mask[None, None].float(), (H, W), mode="nearest")[0, 0].bool()
            if not any(
                float((mask & other).sum() / (mask | other).sum().clamp_min(1)) > 0.8 for other in resized
            ):
                resized.append(mask)
        self.last_vehicle_masks = (
            torch.stack(resized) if resized else torch.zeros(0, H, W, device=self.dev, dtype=torch.bool)
        )
        return cls_map, inst


def snap_depth_to_instances(
    depth: torch.Tensor, inst: torch.Tensor, band: int = 3, window: int = 11
) -> torch.Tensor:
    """Depth models blend across object boundaries: a car's silhouette pixels get depths between the car and the wall behind it, a curtain of points
    that shows as a ghost copy of the car from any other viewpoint (and as streaks along fences and poles). Within ``band`` px of an instance boundary
    (-1 = no instance counts as one label) a pixel takes the median depth of the same-instance pixels in its window that are themselves off the band
    (interior), or of all same-instance pixels when the object is too thin to have an interior. Pixels without any same-instance neighbour keep their depth.
    """
    H, W = depth.shape
    boundary = torch.zeros_like(inst, dtype=torch.bool)
    dx = inst[:, 1:] != inst[:, :-1]
    dy = inst[1:] != inst[:-1]
    boundary[:, 1:] |= dx
    boundary[:, :-1] |= dx
    boundary[1:] |= dy
    boundary[:-1] |= dy
    near = F.max_pool2d(boundary.float()[None, None], 2 * band + 1, 1, band)[0, 0] > 0
    if not near.any():
        return depth
    pad = window // 2
    sel = near.flatten()
    d = torch.where(torch.isfinite(depth) & (depth > 0), depth, torch.full_like(depth, float("inf")))
    dn = F.unfold(F.pad(d[None, None], (pad,) * 4, value=float("inf")), window)[0].T[
        sel
    ]  # (n, k*k) neighbour depths
    ln = F.unfold(F.pad(inst[None, None].float(), (pad,) * 4, value=-2.0), window)[0].T[
        sel
    ]  # neighbour labels (-2 = outside the image)
    nn = (
        F.unfold(F.pad(near[None, None].float(), (pad,) * 4, value=1.0), window)[0].T[sel] > 0
    )  # neighbour on the band
    same = (ln == inst.flatten()[sel, None].float()) & torch.isfinite(dn)
    out = depth.clone().flatten()
    where = sel.nonzero()[:, 0]
    rows = torch.arange(len(where), device=depth.device)
    settled = torch.zeros(len(where), dtype=torch.bool, device=depth.device)
    for valid in (
        same & ~nn,
        same,
    ):  # interior first, any same-instance pixel as the fallback for pixels not settled yet
        valid = valid & ~settled[:, None]
        vals = torch.where(valid, dn, torch.full_like(dn, float("inf"))).sort(1).values
        cnt = valid.sum(1)
        ok = cnt > 0
        med = vals[rows, (cnt - 1).clamp_min(0) // 2]
        out[where[ok]] = med[ok]
        settled |= ok
    return out.view(H, W)


def _sky_of(segmenter, image: torch.Tensor):
    """image (3, H, W) in [0, 1] -> (H, W) bool: union of the SAM 3 masks for "sky" and "clouds". Sky is appearance without geometry: the depth model
    reports an arbitrary finite depth there, and lifting it would put shells of sky-coloured points around elevated viewpoints.
    """
    H, W = image.shape[-2:]
    pil = Image.fromarray((image.permute(1, 2, 0).clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8))
    sky = torch.zeros(H, W, dtype=torch.bool, device=segmenter.dev)
    proc = getattr(segmenter, "_sky_proc", None)
    if (
        proc is None
    ):  # a more permissive threshold than the instance vocabulary: a uniform overcast sky can score below 0.5 and would then be lifted as geometry
        from sam3.model.sam3_image_processor import Sam3Processor

        proc = segmenter._sky_proc = Sam3Processor(
            segmenter.model, device=segmenter.dev, confidence_threshold=0.3
        )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = proc.set_image(pil)
        for name in ("sky", "overcast sky", "clouds"):
            masks = proc.set_text_prompt(prompt=name, state=state)["masks"]
            masks = masks[:, 0] if masks.ndim == 4 else masks
            for k in range(masks.shape[0]):
                m = masks[k].bool().to(segmenter.dev)
                if m.shape != (H, W):
                    m = F.interpolate(m[None, None].float(), (H, W), mode="nearest")[0, 0].bool()
                sky |= m
    return sky


Segmenter.sky = _sky_of
Segmenter.sky_batch = lambda self, images: torch.stack([_sky_of(self, im) for im in images])


def snap_depth_to_image_edges(
    depth: torch.Tensor,
    image: torch.Tensor,
    ratio: float = 1.2,
    band: int = 16,
    window: int = 65,
    sigma_color: float = 0.08,
    iterations: int = 2,
    stride: int = 2,
    min_weight: float = 0.3,
) -> torch.Tensor:
    """Align the depth map's discontinuities with the image's colour edges. The depth estimate's edge between a near object and what is
    behind it can sit 10-15 px off the colour edge (dark background pixels next to a bright rim carry the object's depth), so the
    background pixels on the wrong side inherit the object's parallax and, warped, the silhouette becomes a sawtooth that an exact hold
    pastes into every frame. Within `band` px of a discontinuity (depth ratio > `ratio` within a local neighbourhood) every pixel takes
    the COLOUR-WEIGHTED MEDIAN depth of its `window` (joint bilateral median, Gaussian colour weight `sigma_color` in [0, 1] units, taps
    every `stride` px), `iterations` times, so the depth edge migrates to where the colour changes. The window must reach past the
    misplaced strip into the surface it belongs to, hence the wide default. Only neighbours of weight >= `min_weight` vote: thousands of
    faintly similar pixels would otherwise outvote a thin structure's own few (a mooring pole against water would take the water's depth
    with a soft weight). image (3, H, W) in [0, 1]; non-finite / non-positive depths are left alone."""
    H, W = depth.shape
    finite = torch.isfinite(depth) & (depth > 0)
    if not finite.any():
        return depth
    out = depth.clone()
    img = image.float()
    pad = window // 2
    for _ in range(iterations):
        d = torch.where(finite, out, torch.full_like(out, float("nan")))
        # a discontinuity may be a ramp a dozen pixels wide (the depth model blurs its edges): the ratio test runs over 15 x 15 neighbourhoods
        # so that a ramp wider than a small window still registers as one edge
        dmax = F.max_pool2d(torch.nan_to_num(d, nan=0.0)[None, None], 15, 1, 7)[0, 0]
        dmin = -F.max_pool2d(-torch.nan_to_num(d, nan=float("inf"))[None, None], 15, 1, 7)[0, 0]
        edge = finite & (dmax > ratio * dmin.clamp_min(1e-6))
        # a depth edge that coincides with a colour edge is right where it is (treating every edge would split a thin pole against the
        # water in two); only depth edges inside a uniform colour region are misplaced and define the band
        gx = (img[:, :, 1:] - img[:, :, :-1]).abs().sum(0)
        gy = (img[:, 1:, :] - img[:, :-1, :]).abs().sum(0)
        grad = torch.zeros_like(out)
        grad[:, :-1] = gx
        grad[:-1, :] = torch.maximum(grad[:-1, :], gy)
        colour_edge = F.max_pool2d((grad > 0.12).float()[None, None], 7, 1, 3)[0, 0] > 0
        near = (
            F.max_pool2d((edge & ~colour_edge).float()[None, None], 2 * band + 1, 1, band)[0, 0] > 0
        ) & finite
        sel = near.nonzero(as_tuple=True)
        n = sel[0].numel()
        if n == 0:
            break
        ys, xs = sel
        centre = img[:, ys, xs]  # (3, n)
        dpad = F.pad(d[None, None], (pad,) * 4, value=float("nan"))[0, 0]
        ipad = F.pad(img[None], (pad,) * 4, mode="replicate")[0]
        depths, weights = [], []
        for dy in range(0, window, stride):
            for dx in range(0, window, stride):
                dn = dpad[ys + dy, xs + dx]
                cn = ipad[:, ys + dy, xs + dx]
                w = torch.exp(-((cn - centre) ** 2).sum(0) / (2 * sigma_color**2))
                w = torch.where(torch.isfinite(dn) & (w >= min_weight), w, torch.zeros_like(w))
                depths.append(dn)
                weights.append(w)
        depths = torch.stack(depths, 1)
        weights = torch.stack(weights, 1)  # (n, k*k)
        order = torch.nan_to_num(depths, nan=float("inf")).argsort(1)
        ds = torch.gather(depths, 1, order)
        ws = torch.gather(weights, 1, order)
        cum = ws.cumsum(1)
        total = cum[:, -1:]
        ok = total[:, 0] > 1e-6
        idx = (cum >= 0.5 * total).float().argmax(1)
        med = ds[torch.arange(n, device=depth.device), idx]
        good = ok & torch.isfinite(med)
        out[ys[good], xs[good]] = med[good]
    return out
