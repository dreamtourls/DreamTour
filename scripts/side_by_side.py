"""Side-by-side video (left | right) with labels, same height: e.g. a prior's own camera conditioning vs the same prior with layout steering.
Frames are composed with OpenCV/PIL (the bundled ffmpeg has no drawtext) and piped to ffmpeg as raw video; both clips are scaled to --height,
the shorter clip sets the length.

    python scripts/side_by_side.py --left runs/text/video_raw.mp4 --right runs/layout/video_raw.mp4 --labels "text only" "+ layout steering (ours)" --out runs/side_by_side.mp4
"""

import argparse
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FD = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf") + "/"


def frames(p, height):
    cap = cv2.VideoCapture(p)
    fps = cap.get(5) or 16
    out = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        h, w = im.shape[:2]
        nw = int(round(w * height / h)) // 2 * 2
        out.append(cv2.resize(im, (nw, height), interpolation=cv2.INTER_AREA))
    return out, fps


def label(im, text, font, acc):
    pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil, "RGBA")
    x0, y0 = 24, 20
    bb = d.textbbox((x0, y0), text, font=font)
    d.rectangle([bb[0] - 12, bb[1] - 8, bb[2] + 12, bb[3] + 8], fill=(0, 0, 0, 120))
    d.text((x0, y0), text, font=font, fill=acc)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--labels", nargs=2, default=None)
    p.add_argument("--height", type=int, default=704)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=int, default=16)
    a = p.parse_args()
    L, fps = frames(a.left, a.height)
    R, _ = frames(a.right, a.height)
    n = min(len(L), len(R))
    font = ImageFont.truetype(FD + "STIXGeneralBol.ttf", 34)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    W = L[0].shape[1] + R[0].shape[1] + 8
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
            f"{W}x{a.height}",
            "-r",
            str(a.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            str(a.out),
        ],
        stdin=subprocess.PIPE,
    )
    for i in range(n):
        l, r = L[i], R[i]
        if a.labels:
            l = label(l, a.labels[0], font, (255, 255, 255, 255))
            r = label(r, a.labels[1], font, (150, 200, 255, 255))
        ff.stdin.write(np.concatenate([l, np.full((a.height, 8, 3), 255, np.uint8), r], 1).tobytes())
    ff.stdin.close()
    ff.wait()
    print(a.out, f"{n} frames at {a.fps} fps")


if __name__ == "__main__":
    main()
