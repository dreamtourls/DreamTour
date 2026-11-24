"""Hand-off frame between chained windows: the previous window's last frame becomes the next window's start frame.

Default (native): the frame is used at the generation size as it is, no crop and no resampling; its colour statistics are anchored 50 % to
the photograph (per-channel mean/std), which is what keeps a long chain from drifting in tone. --legacy reproduces a resampling hand-off
(crop to 16:9, LANCZOS to 1024x576, then the generator's bilinear upsample back to the generation size). --canvas-out writes a 1024x576
copy to outpaint the next window's canvas from (the canvas geometry is 1024x576 + 512/288 margins).

    python scripts/handoff.py runs/w3/frames_raw/vace-080.png canvases/f19/photo.png runs/w4/start.png --canvas-out runs/w4/start_canvas.png [--legacy]
"""

import argparse

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prev", help="last frame of the previous window")
    p.add_argument("ref", help="the input photograph")
    p.add_argument("out")
    p.add_argument(
        "--canvas-out", default=None, help="also write a 1024x576 copy for the next window's canvas"
    )
    p.add_argument(
        "--legacy",
        action="store_true",
        help="crop to 16:9 and resample to 1024x576 instead of keeping the native frame",
    )
    p.add_argument(
        "--no-anchor",
        action="store_true",
        help="no colour anchoring (plain hand-off for baseline chains, e.g. CamTrol)",
    )
    a = p.parse_args()
    im = Image.open(a.prev).convert("RGB")
    ref = Image.open(a.ref).convert("RGB")
    if a.legacy:
        W, H = im.size
        w = int(round(H * 16 / 9))
        x0 = (W - w) // 2
        im = im.crop((x0, 0, x0 + w, H)).resize((1024, 576), Image.LANCZOS)
    x = np.asarray(im, np.float32)
    r = np.asarray(ref.resize(im.size, Image.LANCZOS), np.float32)
    if not a.no_anchor:
        m = (x - x.mean((0, 1))) / (x.std((0, 1)) + 1e-6) * r.std((0, 1)) + r.mean((0, 1))
        x = 0.5 * x + 0.5 * m
    out = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
    out.save(a.out)
    if a.canvas_out:
        (out if out.size == (1024, 576) else out.resize((1024, 576), Image.LANCZOS)).save(a.canvas_out)
    print(
        f"{im.size[0]}x{im.size[1]} -> {a.out}"
        + (" (colour anchored 50 % to the photograph)" if not a.no_anchor else "")
    )


if __name__ == "__main__":
    main()
