"""Pinhole cameras with exact poses (OpenCV convention: x right, y down, z forward; cam-to-world 4x4) and toy trajectories."""

from __future__ import annotations

import math

import torch


class Camera:
    def __init__(self, c2w: torch.Tensor, fx: float, fy: float, cx: float, cy: float, H: int, W: int):
        self.c2w, self.fx, self.fy, self.cx, self.cy, self.H, self.W = c2w.double(), fx, fy, cx, cy, H, W

    @property
    def w2c(self):
        return torch.linalg.inv(self.c2w)

    def resized(self, H: int, W: int):
        return Camera(
            self.c2w.clone(),
            self.fx * W / self.W,
            self.fy * H / self.H,
            self.cx * W / self.W,
            self.cy * H / self.H,
            H,
            W,
        )

    def sees_bounds(self, lo, hi, far=float("inf")):
        """Conservative box/frustum intersection, including footprints at the image border."""
        lo, hi = torch.as_tensor(lo, dtype=torch.float64), torch.as_tensor(hi, dtype=torch.float64)
        corners = torch.cartesian_prod(*[torch.stack((lo[i], hi[i])) for i in range(3)])
        transform = self.w2c.cpu()
        p = corners @ transform[:3, :3].T + transform[:3, 3]
        x, y, z = p.unbind(1)
        planes = (
            z,
            far - z,
            self.fx * x + self.cx * z,
            (self.W - self.cx) * z - self.fx * x,
            self.fy * y + self.cy * z,
            (self.H - self.cy) * z - self.fy * y,
        )
        return not any(bool((plane < 0).all()) for plane in planes)

    def project(self, xyz_world: torch.Tensor):
        """(N, 3) world -> pixel (u, v) float, depth z (camera), in-front mask."""
        w2c = self.w2c.to(xyz_world.device)
        p = (w2c[:3, :3] @ xyz_world.double().T).T + w2c[:3, 3]
        z = p[:, 2]
        u = self.fx * p[:, 0] / z.clamp_min(1e-6) + self.cx
        v = self.fy * p[:, 1] / z.clamp_min(1e-6) + self.cy
        return u.float(), v.float(), z.float(), z > 1e-4

    def unproject(self, depth: torch.Tensor, mask: torch.Tensor | None = None):
        """depth (H, W) metric z -> world points (M, 3) for mask pixels (all if None), plus their pixel indices."""
        H, W = depth.shape
        v, u = torch.meshgrid(
            torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij"
        )
        m = torch.ones_like(depth, dtype=torch.bool) if mask is None else mask
        z = depth[m].double()
        x = (u[m].double() + 0.5 - self.cx) / self.fx * z
        y = (v[m].double() + 0.5 - self.cy) / self.fy * z
        c2w = self.c2w.to(depth.device)
        pc = torch.stack((x, y, z), -1)
        pw = (c2w[:3, :3] @ pc.T).T + c2w[:3, 3]
        return pw, m.nonzero(as_tuple=False)

    def pixel_size_at(self, z: torch.Tensor):
        return (z / self.fx).float()  # world size of one pixel at depth z


def look_at(eye, target, up=(0.0, -1.0, 0.0)):
    """cam-to-world with OpenCV axes (y down): forward = target - eye."""
    eye, target, up = (torch.as_tensor(v, dtype=torch.float64) for v in (eye, target, up))
    f = target - eye
    f = f / f.norm()
    r = torch.cross(f, up, dim=0)
    r = r / r.norm()
    d = torch.cross(f, r, dim=0)  # d points down (y)
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, d, f, eye
    return c2w


def intrinsics(H: int, W: int, fov_deg: float):
    f = 0.5 * W / math.tan(math.radians(fov_deg) / 2)
    return f, f, W / 2, H / 2


def _ease(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)  # smooth-step in [0, 1]


def tour(n: int, step: float, yaw_deg: float):
    """A slow, smooth, natural camera path that revisits: walk forward, curve gently while walking, pan 180 degrees in place, walk back along the path
    looking the other way. Peak translation per view = ``step`` m and peak rotation per view = ``yaw_deg`` degrees (2.5 x that during the in-place pan),
    eased in and out of every segment. Returns (eye, yaw) per view; the return leg revisits the outbound scene from the opposite side.
    """
    pan_views = min(max(8, math.ceil(180.0 / (2.5 * yaw_deg))), n // 2)
    rest = n - pan_views
    segs = [("walk", round(rest * 0.36)), ("curve", round(rest * 0.28)), ("pan", pan_views)]
    segs.append(("walk", n - sum(m for _, m in segs)))
    out = []
    eye = [0.0, 0.0, 0.0]
    yaw = 0.0
    k_ease = 2.0 / math.pi  # eased profile: peak speed = pi/2 x mean
    for kind, m in segs:
        m = max(2, m)
        turn = (
            math.radians(yaw_deg) * m * k_ease * 0.6
            if kind == "curve"
            else (math.pi if kind == "pan" else 0.0)
        )
        dist = step * m * k_ease if kind != "pan" else 0.0
        for k in range(m):
            w = _ease((k + 1) / m) - _ease(k / m)
            yaw += turn * w
            eye[0] += math.sin(yaw) * dist * w
            eye[2] += math.cos(yaw) * dist * w
            out.append((tuple(eye), yaw))
    return out[:n]


def explore(n: int, step: float, yaw_deg: float, lane: float = 2.5):
    """Continuous coverage path for a navigable world: walk the centre lane forward, turn 180 degrees in place, walk back drifting onto the left lane
    (looking back), turn 180 degrees at the start, walk forward drifting onto the right lane. Per-view motion <= step m and <= 2.5 x yaw_deg degrees.
    """
    pan_views = min(max(8, math.ceil(180.0 / (2.5 * yaw_deg))), n // 6)
    walk = max(4, (n - 2 * pan_views) // 3)
    L = step * walk
    out = []

    def walk_seg(x0, x1, z0, z1, yaw):
        for k in range(walk):
            t = (k + 0.5) / walk
            out.append(
                ((x0 + (x1 - x0) * min(1.0, 3 * t), 0.0, z0 + (z1 - z0) * t), yaw)
            )  # lane change over the first third

    def pan_seg(x, z, yaw0):
        for k in range(pan_views):
            out.append(((x, 0.0, z), yaw0 + math.pi * (k + 1) / pan_views))

    walk_seg(0.0, 0.0, 0.0, L, 0.0)
    pan_seg(0.0, L, 0.0)
    walk_seg(0.0, -lane, L, 0.0, math.pi)
    pan_seg(-lane, 0.0, math.pi)
    walk_seg(-lane, lane, 0.0, L, 2 * math.pi)
    return out[:n]


def drone(n: int, target_z: float = 10.0, apex: float = 18.0, land_z: float = 20.0):
    """A drone rises from the start pose on the ground straight up to `apex` m while always looking at the same ground point (target_z m down the street),
    so the pitch sweeps smoothly from horizontal to straight-down and consecutive views overlap heavily. Eased so the ends are slow. OpenCV axes (y down).
    """
    out = []
    for i in range(n):
        t = _ease((i + 0.5) / n)
        eye = (0.0, -apex * t, target_z * t)
        out.append(
            (eye, (0.0, 0.0, target_z))
        )  # rises and drifts over the target so the pitch sweeps 0 -> 90 deg (straight-down BEV at the apex)
    return out


def trajectory(
    kind: str,
    n: int,
    H: int,
    W: int,
    fov_deg: float = 70.0,
    step: float = 0.25,
    yaw_deg: float = 2.0,
    apex: float = 18.0,
):
    """Toy trajectories: 'forward' (walk +z), 'pan' (yaw about the eye), 'arc' (walk forward while turning), 'tour' (slow smooth path with revisits, see ``tour``)."""
    fx, fy, cx, cy = intrinsics(H, W, fov_deg)
    cams = []
    tr = (
        tour(n, step, yaw_deg)
        if kind == "tour"
        else (explore(n, step, yaw_deg) if kind == "explore" else None)
    )
    if kind == "yaw3":  # three views from the same eye: yaw -10, 0, +10 degrees (a small horizontal fan)
        import math as _m

        cams3 = []
        for yaw in (_m.radians(-10), 0.0, _m.radians(10)):
            tgt = (_m.sin(yaw), 0.0, _m.cos(yaw))
            cams3.append(Camera(look_at((0.0, 0.0, 0.0), tgt), fx, fy, cx, cy, H, W))
        return cams3[:n]
    if (
        kind == "tilt"
    ):  # same eye, pitching down by yaw_deg per view (0, -yaw_deg, -2 yaw_deg, ...): a vertical fan, pure rotation
        import math as _m

        return [
            Camera(
                look_at(
                    (0.0, 0.0, 0.0),
                    (0.0, _m.sin(_m.radians(yaw_deg * i)), _m.cos(_m.radians(yaw_deg * i))),
                    up=(0.0, -1.0, 0.05),
                ),
                fx,
                fy,
                cx,
                cy,
                H,
                W,
            )
            for i in range(n)
        ]
    if (
        kind == "rise"
    ):  # straight vertical rise over the start by `step` m per view, always looking at the ground point 10 m ahead (view 0 = exact ground view)
        return [
            Camera(
                look_at((0.0, -step * i, 0.0), (0.0, 0.0, 10.0), up=(0.0, -1.0, 0.05)), fx, fy, cx, cy, H, W
            )
            for i in range(n)
        ]
    if kind == "roll":  # same eye and forward axis, rolling by yaw_deg per view about the optical axis
        import math as _m

        return [
            Camera(
                look_at(
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    up=(_m.sin(_m.radians(yaw_deg * i)), -_m.cos(_m.radians(yaw_deg * i)), 0.0),
                ),
                fx,
                fy,
                cx,
                cy,
                H,
                W,
            )
            for i in range(n)
        ]
    if kind in ("spiral", "gentle"):  # gentle eases bounded small motion in/out for video diagnostics
        import math as _m

        out6 = []
        for i in range(n):
            progress = _ease(i / max(1, n - 1)) * (n - 1) if kind == "gentle" else i
            ratios = (1.0, 0.15, 0.1) if kind == "gentle" else (1.0, 0.5, 0.5)
            yaw, pitch, roll = (_m.radians(yaw_deg * progress * k) for k in ratios)
            eye = (0.3 * step * progress, -0.5 * step * progress, step * progress)
            fwd = (_m.sin(yaw) * _m.cos(pitch), _m.sin(pitch), _m.cos(yaw) * _m.cos(pitch))
            tgt = tuple(e + f for e, f in zip(eye, fwd))
            out6.append(
                Camera(look_at(eye, tgt, up=(_m.sin(roll), -_m.cos(roll), 0.0)), fx, fy, cx, cy, H, W)
            )
        return out6
    if (
        kind == "bev3"
    ):  # minimal test: two ground views 1 m apart and one straight-down view from `apex` m above the target
        tgt = (0.0, 0.0, 10.0)
        poses = [((0.0, 0.0, 0.0), tgt), ((0.0, 0.0, 1.0), tgt), ((0.0, -apex, 10.0), tgt)]
        return [Camera(look_at(eye, t, up=(0.0, -1.0, 0.05)), fx, fy, cx, cy, H, W) for eye, t in poses[:n]]
    if (
        kind == "drone"
    ):  # look-at with a fixed up hint tilted slightly forward so the straight-down view at the apex is not degenerate
        return [
            Camera(look_at(eye, tgt, up=(0.0, -1.0, 0.05)), fx, fy, cx, cy, H, W)
            for eye, tgt in drone(n, apex=apex)
        ]
    if kind == "loop":
        distances = [step * min(i, n - 1 - i) for i in range(n)]
        return [Camera(look_at((0, 0, z), (0, 0, z + 1)), fx, fy, cx, cy, H, W) for z in distances]
    for i in range(n):
        if kind == "forward":
            eye = (0.0, 0.0, step * i)
            yaw = 0.0
        elif kind == "pan":
            eye = (0.0, 0.0, 0.0)
            yaw = math.radians(yaw_deg * i)
        elif kind in ("tour", "explore"):
            eye, yaw = tr[i] if i < len(tr) else tr[-1]
        else:
            eye = (step * i * 0.7, 0.0, step * i)
            yaw = math.radians(yaw_deg * i)
        tgt = (eye[0] + math.sin(yaw), eye[1], eye[2] + math.cos(yaw))
        cams.append(Camera(look_at(eye, tgt), fx, fy, cx, cy, H, W))
    return cams


def union_extent(cams: list):
    """Half-tangents (hx, hy) of the frustum that contains every view's frustum, seen from the middle view's pose (inf when a view points more than
    90 degrees away: no pinhole chart contains it)."""
    Rm = cams[len(cams) // 2].c2w[:3, :3]
    hx = hy = 0.0
    for c in cams:  # frustum corner directions of c expressed in the middle camera
        corners = torch.tensor(
            [
                [0.0, 0.0],
                [c.W, 0.0],
                [0.0, c.H],
                [c.W, c.H],
                [c.W / 2, 0.0],
                [c.W / 2, c.H],
                [0.0, c.H / 2],
                [c.W, c.H / 2],
            ],
            dtype=torch.float64,
        )
        d = torch.stack(
            (
                (corners[:, 0] - c.cx) / c.fx,
                (corners[:, 1] - c.cy) / c.fy,
                torch.ones(8, dtype=torch.float64),
            ),
            -1,
        )
        dm = (Rm.T @ (c.c2w[:3, :3] @ d.T)).T
        if bool((dm[:, 2] <= 1e-6).any()):
            return float("inf"), float("inf")
        dm = dm / dm[:, 2:3]
        hx = max(hx, float(dm[:, 0].abs().max()))
        hy = max(hy, float(dm[:, 1].abs().max()))
    return hx, hy


def split_window(cams: list, max_fov_deg: float = 120.0):
    """Consecutive sub-windows whose union chart stays a sane pinhole (field of view <= max_fov_deg per axis). A window of views that turn
    through a large angle (an in-place pan) would otherwise get a union camera with a near-zero focal length: a 170-degree chart the depth
    model cannot read, and spawned points thousands of kilometres away."""
    limit = math.tan(math.radians(max_fov_deg / 2))
    parts, start = [], 0
    while start < len(cams):
        end = start + 1
        while end < len(cams) and max(union_extent(cams[start : end + 1])) <= limit:
            end += 1
        parts.append(cams[start:end])
        start = end
    return parts


def union_camera(cams: list, max_w: int = 1024):
    """A virtual camera at the middle view's pose whose frustum contains every view's frustum (angular resolution of the views, width <= max_w)."""
    mid = cams[len(cams) // 2]
    hx, hy = union_extent(cams)
    if not math.isfinite(hx):
        raise ValueError("views span more than a pinhole chart can hold; split the window (split_window)")
    f = mid.fx
    W = int(min(max_w, 2 * math.ceil(hx * f / 16) * 16))
    f = (
        f * W / (2 * hx * f) if 2 * hx * f > max_w else f
    )  # keep the angular resolution unless the width cap applies
    H = int(2 * math.ceil(hy * f / 16) * 16)
    H = min(H, max_w)
    return Camera(mid.c2w.clone(), f, f, W / 2, H / 2, H, W)
