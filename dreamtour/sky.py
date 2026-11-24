"""Sky mask for layout guides without a segmentation model: depth models place the sky at a finite depth close to the scene's own, so a
rise or truck would swing the sky like a wall and tear dark bands into the guide. Heuristic: bright, low-saturation or blue, smooth pixels
in the upper part of the frame, kept only as connected components that touch the top border; the guide then places them at infinity
(depth_far = 1000 x nominal). Optional: a DA3 sky head when available (MD3D_SKY=da3)."""

from __future__ import annotations

import os

import numpy as np


def sky_np(im: np.ndarray, max_row: float = 0.6) -> np.ndarray:
    """im (H, W, 3) uint8 RGB -> bool (H, W). Sky = (bright and unsaturated) or blueish, locally smooth, in the top 60 % of rows, and connected
    to the top border."""
    import cv2

    H, W = im.shape[:2]
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0].astype(np.float32), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    smooth = cv2.GaussianBlur(np.abs(gray - blur), (0, 0), 5) < 3.5
    warm = (h < 40) | (
        h > 160
    )  # OpenCV hue 0..180: reds/oranges/yellows (pale facades, snow in warm light are not sky unless nearly grey)
    grey = (v > 0.40) & ((s < 0.12) | ((s < 0.30) & ~warm))
    blue = (h > 85) & (h < 135) & (s > 0.12) & (v > 0.30)  # blue ~ 100-125
    cand = (grey | blue) & smooth
    cand[int(max_row * H) :] = False
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab = cv2.connectedComponents(cand, connectivity=8)
    keep = np.zeros((H, W), dtype=bool)
    for k in set(np.unique(lab[0:3, :])) - {0}:
        comp = lab == k
        if comp.sum() > 0.002 * H * W:
            keep |= comp
    # fill enclosed holes (grey clouds darker than the threshold inside the sky region)
    inv = (~keep).astype(np.uint8)
    n2, lab2 = cv2.connectedComponents(inv, connectivity=4)
    border = set(np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]])))
    for k in range(1, n2):
        if k not in border:
            keep |= lab2 == k
    return keep


def sky_mask_heuristic(img01, max_row: float = 0.6):
    """img01 torch (3, H, W) in [0, 1] -> torch bool (H, W)."""
    import torch

    im = (img01.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return torch.from_numpy(sky_np(im, max_row))


def add_clouds(sky: np.ndarray, max_row: float = 0.45) -> np.ndarray:
    """Dark clouds painted into the canvas margins are 'not sky' for every model (DA3 and the colour rule alike) and then hang at the scene
    depth. Add every non-sky component that touches the top border, lies entirely above max_row x H and borders the sky.
    """
    import cv2

    H, W = sky.shape
    inv = (~sky).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    out = sky.copy()
    ring = cv2.dilate(sky.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~sky
    for k in set(np.unique(lab[0:3, :])) - {0}:
        comp = lab == k
        rows = np.where(comp.any(1))[0]
        if rows.max() < max_row * H and (comp & ring).any():
            out |= comp
    return out


def sky_mask(img01, device: str = "cuda"):
    """Sky mask (H, W) bool: DA3's sky head when MD3D_SKY=da3 and Depth Anything 3 is importable, else the heuristic; dark clouds added."""
    import torch

    m = _sky_mask_model(img01, device)
    return torch.from_numpy(add_clouds(m.cpu().numpy().astype(bool)))


def _sky_mask_model(img01, device: str = "cuda"):
    if os.environ.get("MD3D_SKY", "heuristic") == "da3":
        try:
            from dreamtour.depth import DA3Depth

            da3 = DA3Depth(device)
            import torch.nn.functional as F

            frame = (img01.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            pred = da3._run([frame])
            sky = pred.sky
            if sky is not None:
                import torch

                m = torch.as_tensor(np.asarray(sky, np.float32))[0]
                return F.interpolate(m[None, None], img01.shape[-2:], mode="nearest")[0, 0] > 0.5
        except Exception as e:
            print(f"[sky] DA3 unavailable ({str(e)[:80]}); heuristic", flush=True)
    return sky_mask_heuristic(img01)
