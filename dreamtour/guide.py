"""Layout guides for training-free camera steering: every camera motion, with depth from a monocular / Pi3X / DA3 backend where needed.

A guide is the low-frequency layout the video prior should follow: a video of what a camera on the requested path would see, rendered by
reprojecting the photograph (or the wider FLUX canvas) with an estimated depth map. It is only used (i) low-passed, for layout steering, (ii) through
its median flow, for noise transport and rotary steering, so holes and stretching after the reprojection do not matter: the frozen video prior
invents the content. Pure rotations (pan, tilt, roll) are depth independent; translations use the depth up to a scale set by the scene's nominal
depth; orbits and bird's-eye views need the depth for their layout.

    cams = camera_path("orbit_left", 25., T=81, H=720, W=1280, nominal=4.)        # OpenCV cameras along the move
    depth = estimate_depth(image01, cams[0], backend="mono", nominal=4.)           # metric z (H, W) with median = nominal
    frames = render_guide(image01, depth, cams[0], cams)                           # (T, 3, H, W) in [0, 1]
    dx, dy, zoom = guide_motion(frames)                                            # per-frame camera motion read off the guide (px at 512 wide)
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from dreamtour.cameras import Camera, intrinsics, look_at

TRANSLATIONS = {
    "truck_right": (1.0, 0.0, 0.0),
    "truck_left": (-1.0, 0.0, 0.0),
    "rise": (0.0, -1.0, 0.0),
    "descend": (0.0, 1.0, 0.0),
    "forward": (0.0, 0.0, 1.0),
    "back": (0.0, 0.0, -1.0),
}
ROTATIONS = {
    "pan_left": ("yaw", -1.0),
    "pan_right": ("yaw", 1.0),
    "tilt_up": ("pitch", 1.0),
    "tilt_down": ("pitch", -1.0),
    "roll_left": ("roll", -1.0),
    "roll_right": ("roll", 1.0),
}
ORBITS = {"orbit_left": -1.0, "orbit_right": 1.0}
MOVES = tuple(TRANSLATIONS) + tuple(ROTATIONS) + tuple(ORBITS) + ("bev",)
UNITS = {m: "m" for m in TRANSLATIONS}
UNITS.update({m: "deg" for m in ROTATIONS})
UNITS.update({m: "deg" for m in ORBITS})
UNITS["bev"] = "m"


def _rot(axis: str, angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    if axis == "yaw":  # about the camera's y (down) axis: positive turns the view to the right
        return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)
    if (
        axis == "pitch"
    ):  # about x: with OpenCV y down, a positive angle turns the forward axis toward -y, i.e. tilts the view UP
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)  # roll about z


def rulebook_adjust(
    move: str, amount: float, refine: bool, fov_deg: float, W: int, ox: int, PW: int, CW: int, sx: float
):
    """Locked-recipe rulebook (re-locked 2026-11-08). Two rules: (1) the PLAIN guide everywhere - FLUX-Fill guide
    refinement is removed from the recipe (it imprinted its fills as artifacts); (2) a pan/orbit may not reproject
    beyond the canvas margin - the guide then samples the canvas rim and smears. Returns (amount, refine, notes).
    """
    import math

    notes = []
    if refine:
        refine = False
        notes.append("rulebook: plain guide (FLUX-Fill guide refinement removed from the recipe, 2026-11-08)")
    if move in ("pan_left", "pan_right", "orbit_left", "orbit_right") and CW > PW and ox > 0:
        fx = 0.5 * W / math.tan(math.radians(fov_deg) / 2)
        margin_chart = ox if move.endswith("_left") else CW - ox - PW
        cap = math.degrees(math.atan(max(margin_chart, 1) * sx / fx))
        if amount > cap + 0.5:
            notes.append(
                f"rulebook: {move} {amount:g} deg exceeds the canvas margin (cap {cap:.1f} deg); clamped"
            )
            amount = float(int(cap))
    return amount, refine, notes


def camera_path(
    move: str,
    amount: float,
    T: int,
    H: int,
    W: int,
    nominal: float,
    fov_deg: float = 50.0,
    pivot_scale: float = 1.0,
) -> list[Camera]:
    """T cameras from the photograph's pose (origin, looking down +z) to the end of the move, linearly spaced. Translations in metres along the
    camera axes; pan / tilt / roll in degrees about the camera centre; orbit_* in degrees around the point at ``nominal`` metres ahead (the camera
    keeps looking at it); bev rises by ``amount`` metres while keeping that point in view (a bird's-eye approach).
    """
    if move not in MOVES:
        raise ValueError(f"unknown move {move!r}; choose from {', '.join(MOVES)}")
    fx, fy, cx, cy = intrinsics(H, W, fov_deg)
    cams = []
    # orbit / bev centre: at the nominal (median) depth by default; pivot_scale < 1 orbits a nearer point, so everything beyond it shifts the same
    # way and the far shift approaches the full angle (an orbit about the median depth moves near and far content in opposite directions, which
    # cancel in the low pass and under-steer the prior)
    r = nominal * pivot_scale
    pivot = torch.tensor([0.0, 0.0, r], dtype=torch.float64)
    for t in range(T):
        f = t / max(T - 1, 1)
        if move in TRANSLATIONS:
            eye = torch.tensor(TRANSLATIONS[move], dtype=torch.float64) * amount * f
            c2w = torch.eye(4, dtype=torch.float64)
            c2w[:3, 3] = eye
        elif move in ROTATIONS:
            axis, sign = ROTATIONS[move]
            c2w = torch.eye(4, dtype=torch.float64)
            c2w[:3, :3] = _rot(axis, math.radians(amount) * sign * f)
        elif move in ORBITS:
            ang = math.radians(amount) * ORBITS[move] * f
            eye = pivot + torch.tensor(
                [math.sin(ang) * r, 0.0, -math.cos(ang) * r], dtype=torch.float64
            )  # circle of radius r around the pivot
            c2w = look_at(eye, pivot)
        else:  # bev
            eye = torch.tensor([0.0, -amount * f, 0.0], dtype=torch.float64)
            c2w = look_at(eye, pivot) if f > 0 else torch.eye(4, dtype=torch.float64)
        cams.append(Camera(c2w, fx, fy, cx, cy, H, W))
    return cams


@torch.no_grad()
def estimate_depth(
    image01: torch.Tensor, cam: Camera, backend: str = "pi3x", nominal: float = 4.0, device: str = "cuda"
) -> torch.Tensor:
    """Metric z depth (H, W) of the photograph: 'mono' = Depth Anything V2 small with its median set to ``nominal`` (always available),
    'pi3x' / 'da3' = the dreamtour backends (need their sources and weights), aligned to ``nominal`` the same way.
    """
    img = image01.to(device)
    if (
        backend == "plane"
    ):  # no depth model: a plane at the nominal depth; render_guide then warps the texture by the exact plane homography
        return torch.full(img.shape[-2:], float(nominal), device=device)
    d = _estimate_depth_model(img, cam, backend, nominal, device)
    # Sky at infinity: depth models put the sky at a finite depth close to the scene's own, so a rise, truck or dolly would swing the sky
    # like a wall and the hole fill would paint dark bands into the guide.
    from dreamtour.sky import sky_mask

    sky = sky_mask(img, device).to(d.device)
    if bool(
        sky.any()
    ):  # eroded by 4 px so no roofline pixel is sent to infinity with the sky (a duplicated roof edge otherwise)
        k = 9
        sky = (1 - F.max_pool2d((~sky).float()[None, None], k, stride=1, padding=k // 2)[0, 0]).bool()
        d = d.clone()
        d[sky] = 1000.0 * float(nominal)
    return d


def _estimate_depth_model(img, cam, backend, nominal, device):
    if backend == "mono":
        from dreamtour.depth import MonoDepth

        return MonoDepth(device).metric(img, None, None, nominal=nominal)
    if backend == "pi3x":
        from dreamtour.pi3depth import Pi3XDepth

        return Pi3XDepth(device).metric_views(img[None], [cam], nominal=nominal)[0]
    if backend == "da3":
        from dreamtour.depth import DA3Depth

        return DA3Depth(device).metric_views(img[None], [cam], nominal=nominal)[0]
    if backend in ("vggt", "vggto"):
        return _vggt_depth(img, nominal, device, omega=backend == "vggto")
    if backend == "unik3d":
        return _unik3d_depth(img, cam, nominal, device)
    raise ValueError(backend)


DEPTH_BACKENDS = ("plane", "mono", "pi3x", "da3", "vggt", "vggto", "unik3d")


def _median_to_nominal(d: torch.Tensor, nominal: float) -> torch.Tensor:
    d = d.float().clamp_min(1e-3)
    return d * (nominal / d.median())


@torch.no_grad()
def _vggt_depth(img: torch.Tensor, nominal: float, device: str, omega: bool) -> torch.Tensor:
    """VGGT-1B (facebook/VGGT-1B) or VGGT-Omega (facebook/VGGT-Omega, gated, 512 px) single-view depth, median set to ``nominal``
    (their scale is relative; only the depth *shape* is used)."""
    H, W = img.shape[-2:]
    if omega:
        import glob
        import os
        import tempfile

        from huggingface_hub import snapshot_download
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images

        snap = snapshot_download("facebook/VGGT-Omega")
        cks = sorted(
            glob.glob(os.path.join(snap, "**", "*.pt"), recursive=True)
            + glob.glob(os.path.join(snap, "**", "*.pth"), recursive=True)
        )
        ck = [c for c in cks if "512" in os.path.basename(c)] or cks
        if not ck:
            raise FileNotFoundError(f"no checkpoint in {snap}")
        model = VGGTOmega().to(device).eval()
        model.load_state_dict(torch.load(ck[0], map_location="cpu"))
        with tempfile.TemporaryDirectory() as td:
            from torchvision.utils import save_image

            save_image(img.clamp(0, 1), f"{td}/img.png")
            images = load_and_preprocess_images([f"{td}/img.png"], image_resolution=512).to(device)
    else:
        from vggt.models.vggt import VGGT

        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
        w = 518
        h = max(14, round(H * w / W / 14) * 14)
        images = F.interpolate(img[None].float(), (h, w), mode="bilinear", align_corners=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(images)
    d = pred["depth"]
    while d.dim() > 2:  # (B, S, h, w, 1) -> (h, w)
        d = d[0] if d.shape[0] == 1 or d.dim() > 3 else d[..., 0]
    d = F.interpolate(d.float()[None, None], (H, W), mode="bilinear", align_corners=False)[0, 0]
    del model
    torch.cuda.empty_cache()
    return _median_to_nominal(d, nominal)


@torch.no_grad()
def _unik3d_depth(img: torch.Tensor, cam: Camera, nominal: float, device: str) -> torch.Tensor:
    """UniK3D ViT-L (lpiccinelli/unik3d-vitl) metric depth with the guide's pinhole intrinsics when the camera class is available; the median
    is still set to ``nominal`` so every backend follows the same scale rule."""
    from unik3d.models import UniK3D

    model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl").to(device).eval()
    rgb = (img.clamp(0, 1) * 255).round().to(torch.uint8)
    camera = None
    try:
        from unik3d.utils.camera import Pinhole

        camera = Pinhole(
            params=torch.tensor(
                [float(cam.fx), float(cam.fy), float(cam.cx), float(cam.cy)], dtype=torch.float32
            )
        )
    except Exception:
        camera = None
    pred = model.infer(rgb, camera)
    d = pred["depth"][0, 0].float()
    del model
    torch.cuda.empty_cache()
    return _median_to_nominal(d, nominal)


@torch.no_grad()
def render_guide(
    image01: torch.Tensor,
    depth: torch.Tensor,
    cam0: Camera,
    cams: Sequence[Camera],
    footprint: int = 3,
    coarse: int = 16,
    keep: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reproject the textured depth map of ``cam0`` into every camera: z-buffered point splat with a ``footprint`` x ``footprint`` pixel footprint,
    holes filled coarsely (pull-push on a ``coarse``-times smaller grid). Returns (T, 3, H, W) in [0, 1] on the CPU. The result is a layout, not an
    image: the video prior will redraw it."""
    from dreamtour.fill import fill_holes

    dev = depth.device
    img = image01.to(dev).float()
    pts, pix = cam0.unproject(depth, keep)
    colors = img[:, pix[:, 0], pix[:, 1]].T  # (N, 3)
    T_out = []
    r = footprint // 2
    t0 = cam0.c2w[:3, 3].to(dev)
    planar = bool(
        torch.allclose(depth, depth.flatten()[0])
    )  # 'plane' backend: the whole guide is a homography of the texture, no renderer
    for cam in cams:
        if planar:
            T_out.append(_plane_resample(img, cam0, cam, float(depth.flatten()[0])).clamp(0, 1).cpu())
            continue
        if torch.allclose(
            cam.c2w[:3, 3].to(dev), t0, atol=1e-6
        ):  # pure rotation (pan, tilt, roll): exact inverse resampling, no holes, no depth
            T_out.append(_rotation_resample(img, cam0, cam).clamp(0, 1).cpu())
            continue
        u, v, z, front = cam.project(pts)
        H, W = cam.H, cam.W
        canvas = torch.zeros(3, H, W, device=dev)
        zbuf = torch.full((H, W), float("inf"), device=dev)
        order = torch.argsort(z, descending=True)  # far first, near overwrite
        u0, v0, zo, co = u[order], v[order], z[order], colors[order]
        keep = front[order]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                ui = (u0 + dx).round().long()
                vi = (v0 + dy).round().long()
                ok = keep & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
                idx = vi[ok] * W + ui[ok]
                canvas.view(3, -1)[:, idx] = co[ok].T
                zbuf.view(-1)[idx] = zo[ok]
        valid = torch.isfinite(zbuf)
        # disocclusion holes (the strip above a roof when the camera rises, beside a wall when it trucks) are filled from the FAR side:
        # known pixels weighted by their projected depth, so sky and background fill the gap instead of a smear of the near silhouette
        frame = (
            torch.where(
                valid[None],
                canvas,
                fill_holes(
                    canvas,
                    valid,
                    depth=torch.where(valid, zbuf, torch.zeros_like(zbuf)),
                    far_bias=2.0,
                    coarse=coarse,
                ),
            )
            if not valid.all()
            else canvas
        )
        T_out.append(frame.clamp(0, 1).cpu())
    return torch.stack(T_out)


@torch.no_grad()
def _plane_resample(img: torch.Tensor, cam0: Camera, cam: Camera, z0: float) -> torch.Tensor:
    """Frame of ``cam`` by inverse mapping through the plane z = z0 of ``cam0`` (the scene as a billboard at the nominal depth): every output ray is
    intersected with the plane and the texture is sampled there. Exact for rotations, a planar proxy for translations, orbits and bird's-eye moves;
    hole-free and depth-free (user 2026-10-28: no renderer in the guide path). Rays missing the plane or the texture are filled coarsely.
    """
    from dreamtour.fill import fill_holes

    dev = img.device
    H, W = cam.H, cam.W
    vs, us = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float64),
        torch.arange(W, device=dev, dtype=torch.float64),
        indexing="ij",
    )
    d = torch.stack([(us - cam.cx) / cam.fx, (vs - cam.cy) / cam.fy, torch.ones_like(us)], 0).reshape(
        3, -1
    )  # rays in cam
    R0t = cam0.c2w[:3, :3].T.to(dev).double()
    R = cam.c2w[:3, :3].to(dev).double()
    c = R0t @ (
        cam.c2w[:3, 3].to(dev).double() - cam0.c2w[:3, 3].to(dev).double()
    )  # cam centre in cam0 coordinates
    d0 = (R0t @ R) @ d  # ray directions in cam0 coordinates
    s_ = (z0 - c[2]) / d0[2]
    P = c[:, None] + s_ * d0  # intersection with the plane z = z0
    x = cam0.fx * P[0] / z0 + cam0.cx
    y = cam0.fy * P[1] / z0 + cam0.cy
    valid = (s_ > 1e-6) & (x >= 0) & (x <= cam0.W - 1) & (y >= 0) & (y <= cam0.H - 1)
    grid = torch.stack([x / (cam0.W - 1) * 2 - 1, y / (cam0.H - 1) * 2 - 1], -1).reshape(1, H, W, 2).float()
    frame = F.grid_sample(img[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
    valid = valid.reshape(H, W)
    return torch.where(valid[None], frame, fill_holes(frame, valid, coarse=16)) if not valid.all() else frame


@torch.no_grad()
def _rotation_resample(img: torch.Tensor, cam0: Camera, cam: Camera) -> torch.Tensor:
    """Frame of ``cam`` (same optical centre as ``cam0``) by inverse mapping: each output pixel's ray is rotated into ``cam0`` and the texture is sampled
    bilinearly, so a pan over the FLUX chart is exact and hole-free (the forward splat left a blurred third of the frame at 20 degrees, 2026-10-28).
    Rays leaving the texture are filled coarsely like splat holes."""
    from dreamtour.fill import fill_holes

    dev = img.device
    H, W = cam.H, cam.W
    vs, us = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    d = torch.stack([(us - cam.cx) / cam.fx, (vs - cam.cy) / cam.fy, torch.ones_like(us)], 0).reshape(
        3, -1
    )  # rays in cam
    R = (
        cam0.c2w[:3, :3].T.to(dev).double() @ cam.c2w[:3, :3].to(dev).double()
    ).float()  # cam -> world -> cam0
    d0 = R @ d
    z = d0[2]
    x = cam0.fx * d0[0] / z + cam0.cx
    y = cam0.fy * d0[1] / z + cam0.cy
    valid = (z > 1e-6) & (x >= 0) & (x <= cam0.W - 1) & (y >= 0) & (y <= cam0.H - 1)
    grid = torch.stack([x / (cam0.W - 1) * 2 - 1, y / (cam0.H - 1) * 2 - 1], -1).reshape(1, H, W, 2)
    frame = F.grid_sample(img[None], grid, mode="bilinear", padding_mode="border", align_corners=True)[0]
    valid = valid.reshape(H, W)
    return torch.where(valid[None], frame, fill_holes(frame, valid, coarse=16)) if not valid.all() else frame


def guide_motion(frames: torch.Tensor, width: int = 512):
    """Per-frame camera motion of a guide video read with optical flow at ``width`` px: (dx, dy) median flow (content motion; camera moves the
    other way) and zoom (mean divergence), each (T-1,). Used to turn an arbitrary guide into the pan offsets of noise transport and rotary steering.
    """
    import cv2
    import numpy as np

    gs = []
    for f in frames:
        im = (f.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        h = im.shape[0] * width // im.shape[1]
        gs.append(cv2.cvtColor(cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY))
    dx, dy, div = [], [], []
    for a, b in zip(gs[:-1], gs[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 3, 5, 1.2, 0)
        dx.append(float(np.median(fl[..., 0])))
        dy.append(float(np.median(fl[..., 1])))
        div.append(float(np.mean(np.gradient(fl[..., 0], axis=1) + np.gradient(fl[..., 1], axis=0))))
    return np.array(dx), np.array(dy), np.array(div)
