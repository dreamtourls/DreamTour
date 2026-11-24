"""Playback post-processing for chained tours (visualisation only, the method and recipe are untouched): per-frame affine tone lock to the
photograph (mean and std per channel, so contrast and saturation cannot step at a window boundary), removal of the duplicated hand-off frame
at every window boundary, and optional retiming to uniform camera speed (the cumulative median optical-flow displacement is resampled at
constant increments; in-between frames are flow-warped blends of their two neighbours, so a window the prior moved through slowly is played
faster and vice versa; total length is kept).

    python scripts/chain_post.py in.mp4 out.mp4 --photo photo.png [--window 81] [--retime] [--fps 16]
"""

import argparse
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np


def read(p):
    cap = cv2.VideoCapture(p)
    fr = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        fr.append(im)
    return fr


def write(frames, p, fps):
    h, w = frames[0].shape[:2]
    ff = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            "-movflags",
            "+faststart",
            p,
        ],
        stdin=subprocess.PIPE,
    )
    for f in frames:
        ff.stdin.write(np.ascontiguousarray(f).tobytes())
    ff.stdin.close()
    ff.wait()


def flow(a, b, scale=0.5):
    """Farneback flow a -> b at reduced resolution, upsampled to full size (px)."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    h, w = ga.shape
    sa = cv2.resize(ga, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    sb = cv2.resize(gb, sa.shape[::-1], interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(sa, sb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) / scale


def warp(im, f, alpha):
    """im warped along alpha * f (backward mapping: output pixel p takes im at p - alpha f, the content moved by alpha f)."""
    h, w = im.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(
        im, gx - alpha * f[..., 0], gy - alpha * f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--photo", required=True, help="the input photograph: tone reference")
    p.add_argument(
        "--window",
        type=int,
        default=81,
        help="frames per window in the input; the first frame of every window after the first is the duplicated hand-off and is dropped (0 = keep all)",
    )
    p.add_argument(
        "--clip", type=float, nargs=2, default=(0.7, 1.4), help="bounds on the per-channel std gain"
    )
    p.add_argument(
        "--retime",
        action="store_true",
        help="resample to uniform camera speed (median flow displacement per output frame constant)",
    )
    p.add_argument("--fps", type=float, default=16.0)
    a = p.parse_args()
    fr = read(a.src)
    if a.window > 0:
        fr = [f for t, f in enumerate(fr) if t == 0 or t % a.window != 0]
    ref = cv2.imread(a.photo).astype(np.float32)
    mu_r, sd_r = ref.mean((0, 1)), ref.std((0, 1))
    out = []
    for (
        f
    ) in fr:  # affine tone lock: every frame gets the photograph's per-channel mean and std (gain clipped)
        x = f.astype(np.float32)
        mu, sd = x.mean((0, 1)), x.std((0, 1))
        g = np.clip(sd_r / (sd + 1e-6), a.clip[0], a.clip[1])
        out.append(np.clip((x - mu) * g + mu_r, 0, 255).astype(np.uint8))
    info = f"{len(out)} frames, affine lock"
    if a.retime:
        small = [
            cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (480, 270), interpolation=cv2.INTER_AREA)
            for f in out
        ]
        sp = [0.0] + [
            float(
                np.median(
                    np.hypot(
                        *np.moveaxis(
                            cv2.calcOpticalFlowFarneback(
                                small[t - 1], small[t], None, 0.5, 3, 15, 3, 5, 1.2, 0
                            ),
                            -1,
                            0,
                        )
                    )
                )
            )
            for t in range(1, len(out))
        ]
        sp = np.maximum(np.array(sp), 0.02)
        D = np.concatenate(
            [[0.0], np.cumsum(sp[1:])]
        )  # cumulative displacement; the floor keeps static stretches sampled at all
        n = len(out)
        targets = np.linspace(0, D[-1], n)
        src_idx = np.interp(targets, D, np.arange(n))
        res = []
        cache = {}
        for s in src_idx:
            i = int(np.floor(s))
            al = float(s - i)
            if al < 1e-3 or i + 1 >= n:
                res.append(out[min(i, n - 1)])
                continue
            if i not in cache:
                cache = {i: (flow(out[i], out[i + 1]), flow(out[i + 1], out[i]))}
            f01, f10 = cache[i]
            res.append(cv2.addWeighted(warp(out[i], f01, al), 1 - al, warp(out[i + 1], f10, 1 - al), al, 0))
        out = res
        info += f", retimed to uniform speed ({np.percentile(sp[1:], 10):.2f}..{np.percentile(sp[1:], 90):.2f} px/frame before)"
    write(out, a.dst, a.fps)
    print(f"{a.dst}: {info}")


if __name__ == "__main__":
    main()
