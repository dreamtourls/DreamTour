"""Concatenate the windows of a chained tour into one video: <src>/w<k>/video_raw.mp4 for k < --windows, optionally exposure-locked to a
photograph first (the post-process for chained windows; the locked copy is cached as w<k>/video_lock.mp4), re-encoded with libx264.

    python scripts/assemble_chain_video.py --src runs --windows 12 --photo canvases/f19/photo.png --out runs/tour_60s.mp4
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exposure_lock import lock


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--windows", type=int, default=12)
    p.add_argument("--photo", default=None, help="exposure-lock every window to this photograph")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fps", type=int, default=16)
    a = p.parse_args()
    parts = []
    for k in range(a.windows):
        mp4 = a.src / f"w{k}" / "video_raw.mp4"
        if not mp4.exists():
            print(f"missing {mp4}; stopping at {k} windows")
            break
        if a.photo:
            lk = a.src / f"w{k}" / "video_lock.mp4"
            if not lk.exists():
                lock(str(mp4), str(lk), ref_image=a.photo)
            parts.append(lk)
        else:
            parts.append(mp4)
    if not parts:
        raise SystemExit("no windows")
    lst = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    lst.write("".join(f"file '{p.resolve()}'\n" for p in parts))
    lst.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            lst.name,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            "-r",
            str(a.fps),
            str(a.out),
        ],
        check=True,
    )
    os.unlink(lst.name)
    print(a.out, f"{len(parts)} windows, {len(parts) * 5} s")


if __name__ == "__main__":
    main()
