"""Camera pose files for the Wan2.2-Fun Control-Camera baseline (VideoX-Fun), in CameraCtrl's RealEstate10K text format:
one header line, then per frame ``i fx fy cx cy 0 0 w2c[3x4 row-major]`` with intrinsics normalised to the 1280x720 training frame
(VideoX-Fun rescales them to the sample size). The repository's own files (asset/Pan_Left.txt ...) are translations of 1.463 units
('Pan_*' are trucks, 'Zoom_*' 2.926 along z, CW/ACW 58.5 deg rolls) with zero rotation, so the unit is the dimensionless RealEstate10K
scale and --amount 1.463 reproduces the model card's own truck. Moves and axes replicate dreamtour.guide.camera_path (OpenCV camera axes:
x right, y down, z forward; translations along the camera axes, rotations in degrees about the camera centre, orbit / bev around the point
``nominal`` ahead), numpy only so it runs anywhere.

    python baselines/wanfun_poses.py --move truck_right --amount 1.463 --out poses/truck_right_1.463.txt
"""

import argparse
import math
from pathlib import Path

import numpy as np

FX, FY, CX, CY = (
    0.532139961,
    0.946026558,
    0.5,
    0.5,
)  # the repository's pose files (RealEstate10K intrinsics, normalised)
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


def _rot(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == "yaw":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "pitch":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def look_at(eye, target, up=(0.0, -1.0, 0.0)):
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = r, d, f, eye
    return c2w


def camera_path(move, amount, T, nominal=4.0):
    if move not in MOVES:
        raise ValueError(f"unknown move {move!r}; choose from {', '.join(MOVES)}")
    pivot = np.array([0.0, 0.0, nominal])
    out = []
    for t in range(T):
        f = t / max(T - 1, 1)
        c2w = np.eye(4)
        if move in TRANSLATIONS:
            c2w[:3, 3] = np.array(TRANSLATIONS[move]) * amount * f
        elif move in ROTATIONS:
            axis, sign = ROTATIONS[move]
            c2w[:3, :3] = _rot(axis, math.radians(amount) * sign * f)
        elif move in ORBITS:
            ang = math.radians(amount) * ORBITS[move] * f
            c2w = look_at(pivot + np.array([math.sin(ang) * nominal, 0.0, -math.cos(ang) * nominal]), pivot)
        else:
            c2w = look_at(np.array([0.0, -amount * f, 0.0]), pivot) if f > 0 else c2w
        out.append(c2w)
    return out


def write_poses(move, amount, out: Path, frames=81, nominal=4.0):
    lines = [f"{move} {amount} ({frames} frames, dreamtour camera_path, nominal {nominal})"]
    for i, c2w in enumerate(camera_path(move, amount, frames, nominal)):
        w2c = np.linalg.inv(c2w)[:3].reshape(-1)
        lines.append(
            " ".join(
                [str(i), f"{FX:.9f}", f"{FY:.9f}", f"{CX}", f"{CY}", "0", "0"] + [f"{v:.8f}" for v in w2c]
            )
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--move", required=True)
    p.add_argument("--amount", type=float, required=True)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--nominal", type=float, default=4.0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(write_poses(a.move, a.amount, a.out, a.frames, a.nominal))
