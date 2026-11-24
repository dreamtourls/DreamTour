"""Exposure lock (post-process): hold every frame's per-channel means at the reference's values (the input photograph when given, else the first
frame) with a per-channel gain, so a clip keeps the exposure of its conditioning frame (video priors drift by several grey levels over a clip,
with or without steering). Gain only: colour ratios and blacks are preserved; gains are clipped to [0.7, 1.4]. Writes <out> (default
video_lock.mp4 next to the input).
--guide: content-aware reference. A single photograph's means are wrong for a view whose brightness should change (a tilt up into the sky
should brighten, a pan into shade should darken), so the reference is the guide rendered for each frame (the canvas as seen by that camera),
means over its valid pixels on both sides, interpolated between sparse guides.

    python scripts/exposure_lock.py runs/w1/video_raw.mp4 [--out video_lock.mp4] [--ref 0] [--ref-image canvases/f19/photo.png] [--guide runs/w1/guide.mp4]
    python scripts/exposure_lock.py --batch runs/clips "p0*_*"            # every <dir>/<pattern>/video_raw.mp4 -> video_lock.mp4
"""

import argparse
import glob
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


def guide_stats(guide, size):
    """Per-channel means of the guide frames over their valid (non-hole) pixels, plus the masks, resized to the clip size. guide = guide.mp4 or a
    directory of guide-*.png (one guide per keyframe); the means are interpolated linearly over the clip, the mask taken from the nearest guide.
    """
    if os.path.isdir(guide):
        frames = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(guide, "guide-*.png")))]
    else:
        cap = cv2.VideoCapture(str(guide))
        frames = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(im)
    means, masks = [], []
    for g in frames:
        g = cv2.resize(g, size, interpolation=cv2.INTER_AREA)
        m = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY) > 8
        if m.mean() < 0.05:
            m = np.ones_like(m)
        means.append(g[m].reshape(-1, 3).mean(0).astype(np.float64))
        masks.append(m)
    return np.array(means), masks


def lock(src, dst, ref=0, lo=0.7, hi=1.4, ref_image=None, guide=None):
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    if not frames:
        return False
    if (
        ref_image is not None
    ):  # reference = the input photograph (resized to the frame), not the clip's own first frame
        rimg = cv2.resize(
            cv2.imread(str(ref_image)), (frames[0].shape[1], frames[0].shape[0]), interpolation=cv2.INTER_AREA
        )
        m0 = rimg.reshape(-1, 3).mean(0).astype(np.float64)
    else:
        m0 = frames[ref].reshape(-1, 3).mean(0).astype(np.float64)
    gm, gmask = (
        guide_stats(guide, (frames[0].shape[1], frames[0].shape[0])) if guide is not None else (None, None)
    )
    tmp = tempfile.mkdtemp()
    for i, im in enumerate(frames):
        if (
            gm is not None
        ):  # content-aware reference: the guide (canvas rendered for this view) says how bright THIS view should be; same pixels on both sides
            pos = i * (len(gm) - 1) / max(len(frames) - 1, 1)
            k0 = int(np.floor(pos))
            k1 = min(k0 + 1, len(gm) - 1)
            w = pos - k0
            m0 = (1 - w) * gm[k0] + w * gm[k1]
            msk = gmask[k0 if w < 0.5 else k1]
            m = im[msk].reshape(-1, 3).mean(0).astype(np.float64)
        else:
            m = im.reshape(-1, 3).mean(0).astype(np.float64)
        g = np.clip(m0 / np.maximum(m, 1e-3), lo, hi)
        cv2.imwrite(
            os.path.join(tmp, f"f-{i:04d}.png"), np.clip(im.astype(np.float64) * g, 0, 255).astype(np.uint8)
        )
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(fps),
            "-i",
            os.path.join(tmp, "f-%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            str(dst),
        ],
        check=True,
    )
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", nargs="?")
    p.add_argument("--out", default=None)
    p.add_argument("--ref", type=int, default=0)
    p.add_argument("--batch", nargs=2, metavar=("DIR", "PATTERN"))
    p.add_argument("--ref-image", default=None, help="reference photograph (single mode)")
    p.add_argument(
        "--ref-pattern",
        default=None,
        help="batch mode: reference photograph per clip, {pid} = clip name before the first underscore, e.g. canvases/{pid}/photo.png",
    )
    p.add_argument(
        "--guide",
        default=None,
        help="content-aware reference: guide.mp4 or a guide-*.png directory rendered for this clip's cameras; per-frame means over the guide's valid pixels",
    )
    p.add_argument(
        "--guide-pattern",
        default=None,
        help="batch mode: guide per clip, {cell} = clip directory name, e.g. guides/{cell}/guide.mp4",
    )
    a = p.parse_args()
    if a.batch:
        d, pat = a.batch
        n = 0
        for mp4 in sorted(glob.glob(os.path.join(d, pat, "video_raw.mp4"))):
            dst = Path(mp4).with_name("video_lock.mp4")
            cell = Path(mp4).parent.name
            pid = cell.split("_")[0]
            ref_img = a.ref_pattern.format(pid=pid) if a.ref_pattern else None
            gd = a.guide_pattern.format(cell=cell, pid=pid) if a.guide_pattern else None
            if gd is not None and not os.path.exists(gd):
                print("no guide for", cell)
                continue
            if lock(mp4, dst, a.ref, ref_image=ref_img, guide=gd):
                n += 1
        print(f"locked {n} clips under {d}/{pat}")
    else:
        dst = a.out or str(Path(a.src).with_name("video_lock.mp4"))
        print(dst if lock(a.src, dst, a.ref, ref_image=a.ref_image, guide=a.guide) else "failed")


if __name__ == "__main__":
    main()
