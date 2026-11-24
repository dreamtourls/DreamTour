"""Motion normalised to the requested camera trajectory. The clip's rendered guide (<guides>/<cell>/guide.mp4) is a static scene seen along
the requested camera path, so its dense optical flow is the flow the request should produce. Clips are <videos>/<cell>/<clip-name> with one
cell directory per (scene, move), named p<scene>_<move>; the cell list is taken from --cells or the last --videos directory. Per frame pair
(Farneback flow on both, guide-valid pixels only):
  compliance  = sum(f_clip . f_guide) / sum(|f_guide|^2)   projection of the clip's flow on the requested flow (1 = moves as requested,
                                                           0 = static, > 1 overshoot, < 0 opposite)
  unrequested = mean |f_clip - compliance * f_guide|        motion the requested camera does not explain (hallucinated object motion,
                                                           deviation of the camera), in guide pixels per frame pair
Prints per-method means over the clips; --csv writes one row per clip.

    python scripts/motion_vs_request.py --videos runs/ours runs/text_only runs/camtrol --guides runs/guides --limit 48 [--csv motion.csv]
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np


def frames(p, size=None):
    cap = cv2.VideoCapture(p)
    out = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if size is not None and (im.shape[1], im.shape[0]) != size:
            im = cv2.resize(im, size, interpolation=cv2.INTER_AREA)
        out.append(im)
    return out


def flow(a, b):
    return cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )


def measure(clip_mp4, guide_mp4, step=2, width=416):
    g = frames(guide_mp4)
    H, W = g[0].shape[:2]
    s = width / W
    size = (width, int(round(H * s)))
    g = [cv2.resize(x, size, interpolation=cv2.INTER_AREA) for x in g]
    c = frames(clip_mp4, size)
    n = min(len(g), len(c))
    comp, unreq, gmag = [], [], []
    for i in range(0, n - step, step):
        fg = flow(g[i], g[i + step])
        fc = flow(c[i], c[i + step])
        valid = (cv2.cvtColor(g[i], cv2.COLOR_BGR2GRAY) > 8) & (
            cv2.cvtColor(g[i + step], cv2.COLOR_BGR2GRAY) > 8
        )
        valid = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        if valid.mean() < 0.2:
            continue
        fg_v, fc_v = fg[valid], fc[valid]
        den = (fg_v**2).sum()
        k = float((fc_v * fg_v).sum() / den) if den > 1e-6 else float("nan")
        comp.append(k)
        unreq.append(float(np.linalg.norm(fc_v - k * fg_v, axis=1).mean()))
        gmag.append(float(np.linalg.norm(fg_v, axis=1).mean()))
    return (np.nanmean(comp), np.mean(unreq), np.mean(gmag)) if comp else (np.nan, np.nan, np.nan)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument(
        "--guides", default="runs/guides", help="<guides>/<cell>/guide.mp4: the rendered guide of each cell"
    )
    ap.add_argument("--clip-name", default="video_raw.mp4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--cells", default=None, help="take the cell list from this directory (cells shared by every method)"
    )
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    cells = sorted(
        os.path.basename(d)
        for d in glob.glob(os.path.join(a.cells or a.videos[-1], "p*_*"))
        if os.path.exists(os.path.join(a.cells or a.videos[-1], os.path.basename(d), a.clip_name))
    )
    if a.limit:
        cells = cells[: a.limit]
    rows = []
    for v in a.videos:
        vals = []
        for cell in cells:
            mp4 = os.path.join(v, cell, a.clip_name)
            gd = os.path.join(a.guides, cell, "guide.mp4")
            if not (os.path.exists(mp4) and os.path.exists(gd)):
                continue
            k, u, gm = measure(mp4, gd)
            vals.append((k, u, gm))
            rows.append([os.path.basename(v), cell, f"{k:.3f}", f"{u:.3f}", f"{gm:.3f}"])
        if vals:
            vv = np.array(vals)
            print(
                f"{os.path.basename(v):12s} n={len(vv):3d}  compliance (motion / requested) {np.nanmean(vv[:, 0]):.2f}  (median {np.nanmedian(vv[:, 0]):.2f})   unrequested motion {np.nanmean(vv[:, 1]):.2f} px   requested flow {np.nanmean(vv[:, 2]):.2f} px"
            )
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "cell", "compliance", "unrequested_px", "requested_px"])
            w.writerows(rows)


if __name__ == "__main__":
    main()
