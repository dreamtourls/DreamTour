"""Render the layout guide of any camera move from a photograph (+ optional wider canvas) and write it as frames + mp4, with the motion read off it.

    python scripts/guide_video.py --image frame.png --move orbit_left --amount 25 --frames 81 --nominal 4 --depth mono --out runs/guides/orbit
    python scripts/guide_video.py --image photo.png --canvas canvas-000.png --offset 0 0 --move truck_right --amount 1.17 ...

Moves: truck_right/left, rise, descend, forward, back (m); pan_left/right, tilt_up/down, roll_left/right (deg); orbit_left/right (deg about the point
at the nominal depth); bev (m of rise while looking at that point). With --canvas, the canvas is the texture and --offset gives the photograph's
top-left corner inside it (the canvas shares the photograph's optical centre).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dreamtour.cameras import Camera  # noqa: E402
from dreamtour.guide import (  # noqa: E402
    MOVES,
    camera_path,
    estimate_depth,
    guide_motion,
    render_guide,
    rulebook_adjust,
)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--canvas", type=Path)
    p.add_argument("--offset", type=int, nargs=2, default=(0, 0))
    p.add_argument("--move", choices=MOVES, required=True)
    p.add_argument("--amount", type=float, required=True)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--nominal", type=float, default=4.0)
    p.add_argument("--fov", type=float, default=50.0)
    p.add_argument(
        "--depth", choices=("plane", "mono", "pi3x", "da3", "vggt", "vggto", "unik3d"), default="pi3x"
    )
    p.add_argument("--size", type=int, nargs=2, default=None, help="output W H (default: the image size)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--footprint",
        type=int,
        default=3,
        help="splat footprint in pixels for translations (5 fills the stretched ground of bird's-eye guides)",
    )
    p.add_argument(
        "--coords",
        action="store_true",
        help="also write coords.npz: per frame the chart coordinates (x, y in [0, 1]) each guide pixel was rendered from and a photo-visibility channel, for hybrid-source steering",
    )
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    photo = Image.open(a.image).convert("RGB")
    W, H = a.size or photo.size
    tex = Image.open(a.canvas).convert("RGB") if a.canvas else photo
    img = torch.from_numpy(np.asarray(tex, dtype=np.float32) / 255.0).permute(2, 0, 1)
    ox, oy = a.offset
    pw, ph = (
        (tex.width - 2 * ox, tex.height - 2 * oy)
        if (a.canvas and ox > 0 and oy > 0)
        else (photo.width, photo.height)
    )  # photo region inside a centred chart
    sx, sy = W / pw, H / ph  # output size vs the photo as it sits in the texture
    a.amount, _, notes = rulebook_adjust(a.move, a.amount, False, a.fov, W, ox, pw, tex.width, sx)
    for n in notes:
        print(n, flush=True)
    cams = camera_path(a.move, a.amount, a.frames, H, W, a.nominal, a.fov)
    cam_tex = Camera(
        cams[0].c2w.clone(),
        cams[0].fx / sx,
        cams[0].fy / sy,
        cams[0].cx / sx + ox,
        cams[0].cy / sy + oy,
        tex.height,
        tex.width,
    )  # texture shares the optical centre
    depth = estimate_depth(img, cam_tex, a.depth, a.nominal)
    frames = render_guide(img, depth, cam_tex, cams, footprint=a.footprint)
    for t, f in enumerate(frames):
        Image.fromarray((f.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(
            a.out / f"guide-{t:03d}.png"
        )
    if (
        a.coords
    ):  # coordinate guide: where every output pixel comes from in the texture (z = 1 inside the photograph, 0 in the margins)
        ys_c, xs_c = torch.meshgrid(
            torch.arange(tex.height, dtype=torch.float32),
            torch.arange(tex.width, dtype=torch.float32),
            indexing="ij",
        )
        inside = ((xs_c >= ox) & (xs_c < ox + pw) & (ys_c >= oy) & (ys_c < oy + ph)).float()
        coords = render_guide(
            torch.stack([xs_c / (tex.width - 1), ys_c / (tex.height - 1), inside], 0),
            depth,
            cam_tex,
            cams,
            footprint=a.footprint,
        )
        coords = torch.nn.functional.interpolate(
            coords, (H // 4, W // 4), mode="area"
        )  # quarter resolution (the steering reads it at the latent grid)
        np.savez_compressed(
            a.out / "coords.npz",
            coords=coords.numpy().astype(np.float16),
            photo_box=np.array([ox, oy, pw, ph, tex.width, tex.height]),
        )
    dx, dy, zoom = guide_motion(frames)
    json.dump(
        dict(
            move=a.move,
            amount=a.amount,
            unit=(
                "m"
                if a.move in ("truck_right", "truck_left", "rise", "descend", "forward", "back", "bev")
                else "deg"
            ),
            frames=a.frames,
            nominal=a.nominal,
            fov=a.fov,
            depth=a.depth,
            size=[W, H],
            dx=dx.tolist(),
            dy=dy.tolist(),
            zoom=zoom.tolist(),
            total=dict(dx=float(dx.sum()), dy=float(dy.sum()), zoom=float(zoom.sum())),
        ),
        open(a.out / "guide.json", "w"),
        indent=1,
    )
    try:
        import imageio_ffmpeg

        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ff,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                "16",
                "-i",
                str(a.out / "guide-%03d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "20",
                str(a.out / "guide.mp4"),
            ],
            check=True,
        )
    except Exception as e:
        print("mp4 skipped:", e)
    print(
        f"{a.move} {a.amount}: guide total dx {dx.sum():+.0f} dy {dy.sum():+.0f} zoom {zoom.sum():+.2f} (512-px units) -> {a.out}"
    )


if __name__ == "__main__":
    main()
