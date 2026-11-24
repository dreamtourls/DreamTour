"""Layout steering of a frozen Wan video prior along a requested camera move, from one photograph.

    # a steered clip: the photograph is the exact first frame, the guide is rendered inside (dreamtour.guide) from the canvas + estimated depth
    python scripts/video_between.py --frames 81 --backend i2v --size 1280 --guide none --no-paste-anchors \
      --image canvases/f19/photo.png --frame-prompt "<caption>" --move orbit_left --amount 15 \
      --guide3d pi3x --guide3d-canvas canvases/f19/canvas.png --guide3d-offset 512 288 --layout pan --layout-steps 12 --layout-sigma 3 --out runs/f19_orbit_left
    # a chained window: the previous window's hand-off frame starts the clip
    python scripts/video_between.py ... --first-frame runs/w1/start.png --guide3d-canvas runs/w1/start_canvas.png --out runs/w1

Layout steering (--layout pan): during the first K denoising steps the Gaussian low-pass of the guide's latent, noised to the step's sigma,
replaces the low-pass of the running latent with a weight that decays to zero (or stays constant, --layout-schedule); the prior keeps every
high frequency. --layout camtrol runs the CamTrol baseline (single restart from the guide at step K) and --layout-video takes an external
layout video. Backends: Wan 2.2 I2V / TI2V (--backend i2v) or Wan 2.1 VACE (--backend vace), which also supports the older render-guided mode
(--run: a directory with frames/frame-NNN.png, frames/keyframe-NNN.png and report.json, where keyframes are kept exact and the frames between
them are regenerated).

Outputs: frames_raw/vace-NNN.png and video_raw.mp4 (the prior's frames), frames/ and video.mp4 (after the optional tone match / anchor paste),
guide/guide-NN.png (the rendered guide), masks/, compare.mp4 (guide | output), slices.png (y-t slices) and metrics.json (temporal residuals,
first-frame reproduction error, mean flow between output and guide).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))  # the dreamtour package for --guide3d
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def ffmpeg_exe():
    found = shutil.which("ffmpeg")
    if found:
        return found
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def load(paths, size):
    return [Image.open(p).convert("RGB").resize(size, Image.LANCZOS) for p in paths]


def gray(im: Image.Image, width=512):
    im = im.convert("L")
    return np.asarray(
        im.resize((width, int(round(im.height * width / im.width))), Image.BILINEAR), dtype=np.float32
    )


def flow_residual(frames):
    """mean |warp(f[t+1]) - f[t]| (Farneback) and mean |f[t+1]-f[t]| per step, on 512-px grays."""
    import cv2

    plain, res = [], []
    for a, b in zip(frames[:-1], frames[1:]):
        plain.append(float(np.abs(b - a).mean()))
        flow = cv2.calcOpticalFlowFarneback(
            a.astype(np.uint8), b.astype(np.uint8), None, 0.5, 4, 21, 3, 5, 1.2, 0
        )
        h, w = a.shape
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        warped = cv2.remap(
            b, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        res.append(float(np.abs(warped - a).mean()))
    return np.array(plain), np.array(res)


def flow_between(a, b):
    import cv2

    flow = cv2.calcOpticalFlowFarneback(
        a.astype(np.uint8), b.astype(np.uint8), None, 0.5, 4, 21, 3, 5, 1.2, 0
    )
    return float(np.hypot(flow[..., 0], flow[..., 1]).mean())


def pan_video(run: Path, report: dict, idx, total: int, size):
    """2D camera control from the canvas: a window the size of the photograph slides across the canvas by the move's planar shift (the canvas margin = the shift at the scene's nominal depth), linearly over the clip; frame t of
    `total` is the crop at fraction t/total. No lift: parallax and disocclusions are the video prior's job. Returns PIL frames at `size`.
    """
    canvas = Image.open(run / "frames" / "canvas-000.png").convert("RGB")
    photo = Image.open(sorted((run / "frames").glob("keyframe-*.png"))[0])
    W, H = photo.size
    CW, CH = canvas.size
    move = report.get("move", "truck_right")
    mx, my = CW - W, CH - H
    frames = []
    for t in idx:
        f = t / max(total, 1)
        if move == "truck_right":
            box = (int(round(f * mx)), 0, int(round(f * mx)) + W, H)
        elif move == "truck_left":
            box = (int(round((1 - f) * mx)), 0, int(round((1 - f) * mx)) + W, H)
        elif move == "rise":
            box = (0, int(round((1 - f) * my)), W, int(round((1 - f) * my)) + H)
        elif move == "forward":  # the 2x chart: zoom from the full chart window to its centre
            z = 1 - 0.5 * f
            cx, cy = CW / 2, CH / 2
            box = (int(cx - z * W / 2), int(cy - z * H / 2), int(cx + z * W / 2), int(cy + z * H / 2))
        else:
            raise SystemExit(f"--guide pan: move {move} not supported")
        frames.append(canvas.crop(box).resize(size, Image.LANCZOS))
    return frames


def tone_match(v: Image.Image, r: Image.Image, how: str) -> Image.Image:
    """colour-match the VACE frame `v` to the render `r` (same pose): per-channel affine on the means/stds, or the render's low-pass."""
    from PIL import ImageFilter

    a = np.asarray(v, dtype=np.float32)
    b = np.asarray(r, dtype=np.float32)
    if how == "affine":
        out = (a - a.mean((0, 1))) / (a.std((0, 1)) + 1e-3) * b.std((0, 1)) + b.mean((0, 1))
    else:
        la = np.asarray(v.filter(ImageFilter.GaussianBlur(24)), dtype=np.float32)
        lb = np.asarray(r.filter(ImageFilter.GaussianBlur(24)), dtype=np.float32)
        out = a - la + lb
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def unstable_mask(frames_gray, thresh=12.0, dilate=15):
    """white where the render pops: flow-compensated residual to BOTH neighbours above `thresh` gray levels, dilated."""
    import cv2

    n = len(frames_gray)
    masks = [np.zeros_like(frames_gray[0], dtype=np.uint8) for _ in range(n)]
    for t in range(n):
        acc = []
        for u in (t - 1, t + 1):
            if 0 <= u < n:
                a, b = frames_gray[t], frames_gray[u]
                flow = cv2.calcOpticalFlowFarneback(
                    a.astype(np.uint8), b.astype(np.uint8), None, 0.5, 4, 21, 3, 5, 1.2, 0
                )
                h, w = a.shape
                gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
                warped = cv2.remap(
                    b, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                )
                acc.append(np.abs(warped - a))
        bad = np.min(acc, 0) > thresh if acc else np.zeros_like(frames_gray[0], dtype=bool)
        m = cv2.dilate(bad.astype(np.uint8) * 255, np.ones((dilate, dilate), np.uint8))
        masks[t] = m
    return masks


def parse_args():
    """Defines the command line and returns the parsed arguments (I2V card defaults for --backend i2v)."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--run",
        type=Path,
        help="chunked_refine output (render guidance); omit with --scene for pure Wan from the photograph (--guide none)",
    )
    p.add_argument(
        "--scene", help="scenes/scenes.json entry: photograph + prompt, no render (implies --guide none)"
    )
    p.add_argument(
        "--image",
        type=Path,
        help="with --scene: use this photograph instead of the scene's (e.g. a mirrored copy)",
    )
    p.add_argument(
        "--frame-prompt", help="without scenes.json: --image is the starting frame and this is its caption"
    )
    p.add_argument("--move", default="truck_right")
    p.add_argument(
        "--amount", type=float, default=None, help="metres (yaw: degrees), for the camera sentence"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--start", type=int, default=0, help="first playback frame of the window (should be an anchor index)"
    )
    p.add_argument("--frames", type=int, default=81, help="window length, 4n+1")
    p.add_argument(
        "--mode",
        choices=("anchors", "inpaint", "holes"),
        default="anchors",
        help="anchors: regenerate every non-anchor frame (render = guidance); inpaint: regenerate only temporally unstable render; "
        "holes: regenerate where the render is not reliable keyframe texture (frames/known-NNN.png from chunked_refine) plus unstable",
    )
    p.add_argument("--model", default="Wan-AI/Wan2.1-VACE-14B-diffusers")
    p.add_argument(
        "--backend",
        choices=("vace", "i2v"),
        default="vace",
        help="i2v = the official Wan2.1-I2V-14B image-to-video pipeline (first frame as image condition, no guidance video or reference); "
        "layout steering, rotary shift and noise transport apply unchanged. Default model for i2v: Wan-AI/Wan2.2-TI2V-5B-Diffusers (card: guidance 5, 50 steps, default scheduler)",
    )
    p.add_argument("--size", type=int, default=832, help="generation side (square scenes); multiple of 16")
    p.add_argument(
        "--size-mult",
        type=int,
        default=16,
        help="round the generated height to this multiple (32 for Wan 2.2 TI2V-5B: 16x VAE and 2x patch; keeps the guide and the generation the same size)",
    )
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--flow-shift", type=float, default=5.0, help="5.0 for 720p-class, 3.0 for 480p")
    p.add_argument("--conditioning-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=float, default=16.0)
    p.add_argument("--prompt", help="override; default = the caption + camera sentence")
    p.add_argument("--thresh", type=float, default=12.0)
    p.add_argument("--dilate", type=int, default=15)
    p.add_argument("--offload", action="store_true", help="enable_model_cpu_offload (shares a GPU)")
    p.add_argument(
        "--guide",
        choices=("render", "depth", "pan", "none"),
        default="render",
        help="render = the rendered RGB playback is VACE's video (it copies textures); depth = Depth-Anything inverse depth of the rendered "
        "playback as a control video (camera motion and geometry only, appearance from --reference and the prompt); none = text only",
    )
    p.add_argument(
        "--keep-anchors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="with --guide depth: also keep the anchor frames as exact RGB frames (mask 0)",
    )
    p.add_argument(
        "--first-frame",
        type=Path,
        help="with --guide none: this image is the exact first frame instead of the photograph (chaining windows)",
    )
    p.add_argument(
        "--canvas-tone",
        choices=("none", "first", "all"),
        default="none",
        help="chained windows drift in saturation and contrast: anchor the colour statistics to the canvas instead of to the previous window. first: the chained first frame is affine-matched (per-channel mean/std) to the "
        "canvas crop at the window's start pose; all: every output frame is matched to its canvas crop as well. Needs --layout pan.",
    )
    p.add_argument(
        "--guide3d",
        default="none",
        help="render the layout guide for ANY camera move from the photograph (+ canvas) with an estimated depth (dreamtour.guide): "
        "'mono' (Depth Anything V2 small), 'pi3x' or 'da3'. Replaces the canvas pan as the layout / noise / rotary guide; --move takes the dreamtour.guide move names",
    )
    p.add_argument(
        "--guide3d-canvas",
        type=Path,
        help="texture for the 3D guide: the canvas (default: the run's canvas-000.png if present, else the photograph)",
    )
    p.add_argument(
        "--guide3d-offset",
        type=int,
        nargs=2,
        default=None,
        help="top-left corner of the photograph inside --guide3d-canvas (default: from the move)",
    )
    p.add_argument(
        "--nominal",
        type=float,
        default=None,
        help="scene depth in metres for the 3D guide (default: the run report's nominal, else 4)",
    )
    p.add_argument(
        "--fov", type=float, default=50.0, help="horizontal field of view of the 3D guide cameras (deg)"
    )
    p.add_argument(
        "--guide3d-pivot",
        type=float,
        default=1.0,
        help="orbit / bev centre as a fraction of the nominal depth (1 = the median depth; 0.5 orbits a nearer point, so the far content moves)",
    )
    p.add_argument(
        "--guide3d-preroll",
        type=int,
        default=0,
        help="windows of the same move already travelled from the canvas pose: the guide covers window preroll+1 of a linear path (chaining on one canvas)",
    )
    p.add_argument(
        "--layout",
        choices=("none", "pan", "render", "camtrol"),
        default="none",
        help="early-step low-frequency layout guidance: during the first --layout-steps "
        "denoising steps the low-pass of the canvas-pan (or render) latent, noised to the step's sigma, replaces the low-pass of "
        "Wan's latent; Wan keeps every high frequency (parallax, disocclusions, texture). Use with --guide none",
    )
    p.add_argument("--layout-steps", type=int, default=6)
    p.add_argument(
        "--layout-sigma",
        type=float,
        default=2.0,
        help="Gaussian blur (latent px; 16 px each) of the low-pass",
    )
    p.add_argument(
        "--layout-weight",
        type=float,
        default=1.0,
        help="blend weight at step 0, decaying linearly to 0 at --layout-steps",
    )
    p.add_argument(
        "--camtrol-mode",
        choices=("restart", "hold"),
        default="restart",
        help="--layout camtrol: restart = single SDEdit restart at step K (faithful to Hou et al.), hold = replace the latent at every step <= K (over-constrained variant)",
    )
    p.add_argument(
        "--layout-video",
        type=Path,
        default=None,
        help="external layout video (mp4 or directory of frames) used as the layout frames instead of the guide3d render, e.g. a CamTrol point-cloud + inpainting guide from baselines/camtrol_guide.py",
    )
    p.add_argument(
        "--layout-schedule",
        choices=("ramp", "const"),
        default="ramp",
        help="ramp (default): w_i = weight (1 - (i+1)/K); const: w_i = weight for i < K, 0 after (stronger, for small requested motions)",
    )
    p.add_argument(
        "--layout-source",
        choices=("guide", "self", "hybrid"),
        default="guide",
        help="guide: the low-pass target is the rendered guide's latent (appearance from the canvas). self: motion-only steering: "
        "the guide only provides the per-frame warp from the first frame; the target is the prior's own first-frame latent transported along that warp, "
        "masked to the region sourced from the photograph, so no foreign appearance (texture, tone) can enter. hybrid: the prior's own transported content where the "
        "photograph is visible and the canvas only in the newly revealed region, so the canvas contributes appearance exactly where the prior has none",
    )
    p.add_argument(
        "--layout-conf",
        choices=("none", "stretch"),
        default="stretch",
        help="weight the correction by the plain guide's reliability: 'stretch' attenuates it where the splat magnified "
        "the texture or filled a hole (source density of the coordinate maps); 'none' = uniform",
    )
    p.add_argument(
        "--layout-tone",
        choices=("mean", "keep", "guide"),
        default="mean",
        help="mean (default): remove the per-frame per-channel spatial mean of the correction, so it carries the layout and not the guide's exposure or colour cast; "
        "keep: also standardise the std (weakens the steering and can flatten contrast); guide: the raw correction, including the guide's tone",
    )
    p.add_argument(
        "--rope-shift",
        choices=("none", "pan"),
        default="none",
        help="positional steering: shift every latent frame's rotary W (and H) positions by the canvas-pan offset in tokens, so Wan's "
        "attention relates frame t's tokens to frame 0's at the camera-shifted location",
    )
    p.add_argument(
        "--rope-scale",
        type=float,
        default=1.0,
        help="gain on the rotary shift (1 = the full pan offset; on its own the shift tends to overshoot the requested travel)",
    )
    p.add_argument(
        "--noise-transport",
        choices=("none", "pan", "flow"),
        default="none",
        help="initial noise = a window sliding over ONE large noise field by the canvas-pan offset per latent frame (2D noise transport): "
        "the video prior follows the correlation structure of its noise, so content slides with the camera without prescribing pixels",
    )
    p.add_argument(
        "--reference",
        choices=("none", "canvas", "photo"),
        default="none",
        help="VACE reference image: the canvas (frames/canvas-000.png, letterboxed) or the photograph, so the regenerated "
        "frames know the scene beyond what the camera has revealed",
    )
    p.add_argument(
        "--tone",
        choices=("none", "affine", "lowpass"),
        default="none",
        help="colour-match every VACE frame to the render at the same pose: per-channel mean/std (affine) or swap in the render's "
        "low frequencies (lowpass, sigma 24 px); the video prior shifts saturation and tone",
    )
    p.add_argument(
        "--paste-anchors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="write the keyframes themselves at the anchor indices (the video VAE packs 4 frames per latent, so VACE cannot hold them exactly)",
    )
    p.add_argument(
        "--anchor-blend",
        type=int,
        default=4,
        help="crossfade length (frames) on each side of a pasted anchor; 0 = hard paste",
    )
    a = p.parse_args()
    a.flow_shift_set = any(x.startswith("--flow-shift") for x in sys.argv)
    if a.backend == "i2v" and not any(x.startswith("--guidance") for x in sys.argv):
        a.guidance = 3.5  # Wan 2.2 I2V card
    if a.backend == "i2v" and not any(x.startswith("--steps") for x in sys.argv):
        a.steps = 40
    return a


def load_inputs(a, S, log):
    """Loads the scene or run: report, renders, keyframes, window indices, anchors, size, guidance frames."""
    if a.scene or a.frame_prompt:  # pure Wan from a photograph / generated frame: no run, no render
        if a.scene:
            scenes = json.loads((ROOT / "scenes/scenes.json").read_text())
            entry = scenes["scenes"][a.scene]
            photo = a.image if a.image else ROOT / entry["image"]
            caption, nominal = entry["prompt"] + scenes.get("_suffix", ""), entry["nominal"]
        else:
            photo, caption, nominal = a.image, a.frame_prompt, 10.0
        report = dict(
            move=a.move,
            amount=a.amount,
            keyframes=1,
            render_frames=a.frames - 1,
            locked=dict(prompt=caption, nominal=nominal),
        )
        keys, renders = [photo], [photo] * a.frames
        a.guide, a.start = "none", 0
    else:
        report = json.loads((a.run / "report.json").read_text())
        renders = sorted((a.run / "frames").glob("frame-*.png"))
        keys = sorted((a.run / "frames").glob("keyframe-*.png"))
    K, R = (
        len(keys) - 1,
        len(renders) - 1,
    )  # the report's render_frames is 0 when the driver chose 80 per interval itself
    assert K >= 0 and R >= 1 and (report.get("render_frames") in (0, None, R)), (
        len(renders),
        report.get("render_frames"),
        len(keys),
        report.get("keyframes"),
    )
    idx = list(range(a.start, a.start + a.frames))
    assert idx[-1] <= R, "window exceeds the clip"
    anchors = {int(round(i * R / max(K, 1))): i for i in range(K + 1)}  # playback index -> keyframe
    anchors_in = {t: anchors[t] for t in idx if t in anchors}
    log(f"window {idx[0]}..{idx[-1]} of {R}; keyframes {K}; anchors in window: {anchors_in}")
    W0, H0 = Image.open(renders[0]).size
    size = (a.size, int(round(H0 * a.size / W0 / a.size_mult)) * a.size_mult)
    video = load([renders[t] for t in idx], size)
    for t, k in anchors_in.items():
        video[idx.index(t)] = load([keys[k]], size)[0]
    S.report, S.keys, S.renders, S.K, S.R = report, keys, renders, K, R
    S.idx, S.anchors_in, S.size, S.video = idx, anchors_in, size, video


def build_guide_video(a, S, log):
    """Builds the VACE video and masks for --guide (render/depth/pan/none) and --mode; saves the masks."""
    report, keys, R, idx = S.report, S.keys, S.R, S.idx
    anchors_in, size, video = S.anchors_in, S.size, S.video
    if a.guide == "pan":
        video = pan_video(a.run, report, idx, R, size)
        anchors_in = {idx[0]: 0} if a.start == 0 else {}
        masks = [Image.new("L", size, 0 if t in anchors_in else 255) for t in idx]
        log(
            f"guide pan: {len(idx)} canvas crops sliding by the margin; first frame kept exact: {bool(anchors_in)}"
        )
    canvas_crops = None
    rgb_video = list(video)  # the guidance frames (metrics compare against them)
    if a.guide == "depth":
        # VACE control mode: a grayscale inverse-depth video of the rendered playback. Depth Anything V2 (the annotator family VACE was
        # trained with) on every render frame; one normalisation for the whole window so the gray level means the same depth in every frame.
        import torch as _t
        from transformers import pipeline as hf_pipeline

        da = hf_pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if _t.cuda.is_available() else -1,
        )
        inv = [
            (
                np.asarray(da(v)["predicted_depth"].squeeze().float().cpu())
                if hasattr(da(v)["predicted_depth"], "cpu")
                else np.asarray(da(v)["predicted_depth"])
            )
            for v in video
        ]
        inv = [np.asarray(Image.fromarray(d.astype(np.float32)).resize(size, Image.BILINEAR)) for d in inv]
        lo, hi = np.percentile(np.stack(inv), 1), np.percentile(np.stack(inv), 99)
        depth_video = [
            Image.fromarray(np.clip((d - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)).convert(
                "RGB"
            )
            for d in inv
        ]
        del da
        if a.keep_anchors:
            video = [video[i] if idx[i] in anchors_in else depth_video[i] for i in range(len(idx))]
            masks = [Image.new("L", size, 0 if t in anchors_in else 255) for t in idx]
        else:
            video, masks = depth_video, [Image.new("L", size, 255) for _ in idx]
            anchors_in = {}
        a.tone = "none"
        log(
            f"guide depth: Depth-Anything inverse depth of {len(idx)} render frames as the control video"
            + (", anchors kept" if a.keep_anchors else "")
        )
    elif a.guide == "pan":
        pass
    elif a.guide == "none":
        a.tone = "none"  # there is no render to match against (matching to the gray placeholders turned the clip gray)
        anchors_in = {idx[0]: 0}
        first = load([a.first_frame or keys[0]], size)[0]
        canvas_crops = (
            pan_video(a.run, report, idx, R, size)
            if (a.canvas_tone != "none" and a.run is not None and a.layout == "pan")
            else None
        )
        if canvas_crops is not None and a.first_frame is not None:
            first = tone_match(first, canvas_crops[0], "affine")
            log(f"canvas tone: chained first frame matched to the canvas crop at {idx[0]}/{R}")
        video = [first] + [Image.new("RGB", size, (128, 128, 128)) for _ in idx[1:]]
        masks = [Image.new("L", size, 0)] + [Image.new("L", size, 255) for _ in idx[1:]]
        log(
            "guide none: photograph as the exact first frame, the rest generated from the prompt"
            + (" and the reference" if a.reference != "none" else "")
        )
    elif a.mode == "anchors":
        masks = [Image.new("L", size, 0 if t in anchors_in else 255) for t in idx]
    elif a.mode == "holes":
        import cv2

        g = [gray(v, a.size) for v in video]
        ms = unstable_mask(g, a.thresh, a.dilate)
        masks = []
        for t, m in zip(idx, ms):
            if t in anchors_in:
                masks.append(Image.new("L", size, 0))
                continue
            kp = a.run / "frames" / f"known-{t:03d}.png"
            if not kp.exists():
                raise SystemExit(f"{kp} missing: the run predates known-mask saving; use --mode inpaint")
            unknown = 255 - np.asarray(Image.open(kp).convert("L").resize(m.shape[::-1], Image.NEAREST))
            unknown = cv2.dilate(unknown, np.ones((a.dilate, a.dilate), np.uint8))
            masks.append(Image.fromarray(np.maximum(unknown, m)).resize(size, Image.NEAREST))
        frac = float(np.mean([np.asarray(m).mean() / 255 for m in masks]))
        log(f"holes mask: {100 * frac:.1f} % of the window regenerated")
    else:
        g = [gray(v, a.size) for v in video]
        ms = unstable_mask(g, a.thresh, a.dilate)
        masks = [
            Image.fromarray(m).resize(size, Image.NEAREST) if t not in anchors_in else Image.new("L", size, 0)
            for t, m in zip(idx, ms)
        ]
        frac = float(np.mean([np.asarray(m).mean() / 255 for m in masks]))
        log(f"inpaint mask: {100 * frac:.1f} % of the window regenerated")
    (a.out / "masks").mkdir(exist_ok=True)
    for t, m in zip(idx, masks):
        m.save(a.out / "masks" / f"mask-{t:03d}.png")
    S.video, S.masks, S.anchors_in = video, masks, anchors_in
    S.canvas_crops, S.rgb_video = canvas_crops, rgb_video


def load_reference(a, S, log):
    """Builds the letterboxed VACE reference image (--reference canvas / photo), or None."""
    size = S.size
    reference = None
    if a.reference != "none":
        src = a.run / "frames" / ("canvas-000.png" if a.reference == "canvas" else "keyframe-000.png")
        if not src.exists():
            raise SystemExit(f"{src} missing: the run has no saved canvas")
        ref = Image.open(src).convert("RGB")
        s_ = min(size[0] / ref.width, size[1] / ref.height)
        ref = ref.resize((max(16, int(ref.width * s_)), max(16, int(ref.height * s_))), Image.LANCZOS)
        reference = Image.new("RGB", size, (128, 128, 128))
        reference.paste(ref, ((size[0] - ref.width) // 2, (size[1] - ref.height) // 2))
        reference.save(a.out / "reference.png")
        log(f"reference: {a.reference} {src.name} -> letterboxed {size}")
    S.reference = reference


def build_prompt(a, S, log):
    """Composes the prompt: the caption plus the camera sentence for the move (or the --prompt override)."""
    report = S.report
    locked = report.get("locked", {})
    move = report.get("move", "truck_right")
    amt = report.get("amount")
    dist = (
        f" by about {amt:g} degrees"
        if (amt and ("yaw" in move or move.startswith(("pan_", "tilt_", "roll_", "orbit_"))))
        else (f" by about {amt:g} metres" if amt else "")
    )
    camera = {
        "none": "",
        "static": "The camera is completely static, a fixed tripod shot",
        "truck_right": f"The camera glides steadily to the right{dist} in a sideways tracking shot, revealing more of the scene on the right",
        "truck_left": f"The camera glides steadily to the left{dist} in a sideways tracking shot, revealing more of the scene on the left",
        "forward": f"The camera moves steadily straight forward{dist} into the scene, a dolly-in",
        "rise": f"The camera rises steadily straight up{dist}, a crane shot",
        "descend": f"The camera lowers steadily straight down{dist}, a reverse crane shot",
        "back": f"The camera moves steadily straight backward{dist}, a dolly-out",
        "pan_left": f"The camera pans steadily to the left{dist}, turning in place",
        "pan_right": f"The camera pans steadily to the right{dist}, turning in place",
        "tilt_up": f"The camera tilts steadily upward{dist}, turning in place",
        "tilt_down": f"The camera tilts steadily downward{dist}, turning in place",
        "roll_left": f"The camera rolls steadily counter-clockwise{dist}",
        "roll_right": f"The camera rolls steadily clockwise{dist}",
        "orbit_left": f"The camera orbits steadily to the left around the centre of the scene{dist}, keeping it in view",
        "orbit_right": f"The camera orbits steadily to the right around the centre of the scene{dist}, keeping it in view",
        "bev": f"The camera rises steadily{dist} while tilting down to keep the scene in view, climbing toward a bird's-eye view",
        "yaw_right": f"The camera pans steadily to the right{dist}",
        "yaw_left": f"The camera pans steadily to the left{dist}",
    }.get(move, "The camera moves steadily through the static scene")
    prompt = a.prompt or (
        locked.get("prompt", "")
        + (f" {camera}; nothing in the scene moves." if camera else " Nothing in the scene moves.")
    )
    log(f"prompt: {prompt[:160]}...")
    S.prompt = prompt


def load_pipeline(a, S, log):
    """Loads the diffusers pipeline (VACE or I2V): float32 VAE, scheduler flow shift, offload rule for the large checkpoints."""
    reference = S.reference
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline, WanVACEPipeline

    t0 = time.time()
    if a.backend == "i2v":
        if "VACE" in a.model:
            a.model = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        reference = None  # the I2V pipeline has no reference-image input; the canvas steers through layout / positions / noise only
    vae = AutoencoderKLWan.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.float32)
    pipe = (WanImageToVideoPipeline if a.backend == "i2v" else WanVACEPipeline).from_pretrained(
        a.model, vae=vae, torch_dtype=torch.bfloat16
    )
    if a.backend != "i2v" or a.flow_shift_set:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=a.flow_shift)
    if a.backend == "i2v" and not any(s in a.model for s in ("5B", "1.3B")) and not a.offload:
        a.offload = True  # the large two-expert checkpoints + umt5 do not fit one 80 GB GPU at 720p next to the activations (the default TI2V-5B fits)
    S.t0, S.pipe, S.reference = t0, pipe, reference


def render_guide3d(a, S, log):
    """Renders the 3D guide (dreamtour.guide): frames, camera displacement, magnification, coordinate maps."""
    report, keys, size, idx, R = S.report, S.keys, S.size, S.idx, S.R
    g3d = None
    if a.guide3d != "none":
        # general layout guide (dreamtour.guide): the canvas (or photograph) re-projected along the requested camera path with an estimated depth; its
        # frames replace the canvas pan for layout steering, and its median flow gives the per-frame camera displacement for the position-level variants
        import numpy as _np

        from dreamtour.cameras import Camera as _Cam
        from dreamtour.guide import MOVES as _G_MOVES
        from dreamtour.guide import (
            camera_path,
            estimate_depth,
            guide_motion,
            render_guide,
            rulebook_adjust,
        )

        move_g = report.get("move", "truck_right")
        amount_g = float(report.get("amount") or a.amount or 1.0)
        if move_g not in _G_MOVES:
            raise SystemExit(
                f"--guide3d: move {move_g} is not in dreamtour.guide.MOVES ({', '.join(_G_MOVES)})"
            )
        nominal_g = a.nominal or float(
            (report.get("locked") or {}).get("nominal") or report.get("nominal") or 4.0
        )
        photo_g = Image.open(keys[0]).convert("RGB")
        PW_g, PH_g = photo_g.size
        canvas_path = a.guide3d_canvas or ((a.run / "frames" / "canvas-000.png") if a.run else None)
        tex = (
            Image.open(canvas_path).convert("RGB")
            if (canvas_path is not None and canvas_path.exists())
            else photo_g
        )
        CW_g, CH_g = tex.size
        if a.guide3d_offset is not None:
            ox_g, oy_g = a.guide3d_offset
            if (
                a.guide3d_canvas is not None and ox_g > 0 and oy_g > 0
            ):  # centred canvas: the photo region is the canvas minus its margins, not the --image file size (a larger input frame would zoom the texture and run pans out of the canvas)
                PW_g, PH_g = CW_g - 2 * ox_g, CH_g - 2 * oy_g
        else:
            ox_g = (
                0
                if move_g == "truck_right"
                else (CW_g - PW_g if move_g == "truck_left" else (CW_g - PW_g) // 2)
            )
            oy_g = (
                CH_g - PH_g
                if move_g == "rise"
                else (CH_g - PH_g) // 2 if move_g not in ("truck_right", "truck_left") else 0
            )
        amount_g, _, _rb_notes = rulebook_adjust(
            move_g, amount_g, False, a.fov, size[0], ox_g, PW_g, CW_g, size[0] / PW_g
        )
        for _n in _rb_notes:
            log(_n)
        P_ = int(a.guide3d_preroll)
        cams_all = camera_path(
            move_g,
            amount_g * (P_ + 1),
            (P_ + 1) * R + 1,
            size[1],
            size[0],
            nominal_g,
            a.fov,
            pivot_scale=a.guide3d_pivot,
        )[
            P_ * R :
        ]  # the window's R+1 cameras, P_ windows along the path
        sx_g, sy_g = size[0] / PW_g, size[1] / PH_g
        c0 = cams_all[0]
        cam_tex = _Cam(
            c0.c2w.clone(), c0.fx / sx_g, c0.fy / sy_g, c0.cx / sx_g + ox_g, c0.cy / sy_g + oy_g, CH_g, CW_g
        )
        tex01 = torch.from_numpy(_np.asarray(tex, dtype=_np.float32) / 255.0).permute(2, 0, 1)
        depth_g = estimate_depth(tex01, cam_tex, a.guide3d, nominal_g)
        frames_g = render_guide(tex01, depth_g, cam_tex, [cams_all[t] for t in idx])  # (T, 3, H, W)
        # coordinate guide: where every output pixel comes from in the chart (channels: x, y normalised; z = 1 inside the photograph, 0 in the margins)
        ys_c, xs_c = torch.meshgrid(
            torch.arange(CH_g, dtype=torch.float32), torch.arange(CW_g, dtype=torch.float32), indexing="ij"
        )
        inside = ((xs_c >= ox_g) & (xs_c < ox_g + PW_g) & (ys_c >= oy_g) & (ys_c < oy_g + PH_g)).float()
        coords_g = render_guide(
            torch.stack([xs_c / (CW_g - 1), ys_c / (CH_g - 1), inside], 0),
            depth_g,
            cam_tex,
            [cams_all[t] for t in idx],
        )  # (T, 3, H, W)
        dx_g, dy_g, div_g = guide_motion(frames_g)
        cum_dx = _np.concatenate([[0.0], _np.cumsum(dx_g)]) * size[0] / 512.0
        cum_dy = _np.concatenate([[0.0], _np.cumsum(dy_g)]) * size[0] / 512.0
        mag_g = _np.exp(
            _np.concatenate([[0.0], _np.cumsum(div_g)]) / 2.0
        )  # magnification of the view: summed divergence of a uniform zoom = 2 ln M
        g3d = dict(
            frames=[Image.fromarray((f.permute(1, 2, 0).numpy() * 255).astype(_np.uint8)) for f in frames_g],
            pan_px=[
                (-float(cum_dx[t]), -float(cum_dy[t])) for t in range(len(idx))
            ],  # camera displacement in generation px (content moves the other way)
            mag=[float(m) for m in mag_g],
            coords=coords_g,
            photo_box=(ox_g, oy_g, PW_g, PH_g, CW_g, CH_g),
        )
        (a.out / "guide").mkdir(exist_ok=True)
        for t_, fr in enumerate(g3d["frames"][:: max(1, len(idx) // 8)]):
            fr.save(a.out / "guide" / f"guide-{t_:02d}.png")
        log(
            f"3D guide ({a.guide3d} depth, nominal {nominal_g:g} m, fov {a.fov:g}): {move_g} {amount_g:g}; camera displacement over the window "
            f"{g3d['pan_px'][-1][0]:+.0f} x {g3d['pan_px'][-1][1]:+.0f} px, magnification {g3d['mag'][-1]:.2f}; texture {CW_g}x{CH_g} offset ({ox_g}, {oy_g})"
        )
        if a.layout == "none":
            a.layout = "pan"  # the guide is only useful if something reads it
        torch.cuda.empty_cache()
    S.g3d = g3d


def encode_layout_frames(a, S, log):
    """Encodes the layout frames (3D guide, canvas pan, render or --layout-video) into Wan latents."""
    g3d, report, idx, R = S.g3d, S.report, S.idx, S.R
    size, renders, pipe, keys = S.size, S.renders, S.pipe, S.keys
    z_layout = layout_mask = None  # only set when layout steering is on
    if a.layout != "none":
        # the layout video (canvas pan or render) in Wan's latent space, normalised like the pipeline's own latents; encoded with only the
        # VAE on the GPU (the transformer is moved afterwards: encoding 81 frames at 832 px next to it ran out of memory)
        if (
            a.layout_video is not None
        ):  # external layout video: resized to the generation size, frames matched to the window (repeat the last if short)
            if a.layout_video.is_dir():
                lf = [Image.open(f).convert("RGB") for f in sorted(a.layout_video.glob("*.png"))]
            else:
                import cv2 as _cv2

                cap = _cv2.VideoCapture(str(a.layout_video))
                lf = []
                while True:
                    ok_, fr_ = cap.read()
                    if not ok_:
                        break
                    lf.append(Image.fromarray(_cv2.cvtColor(fr_, _cv2.COLOR_BGR2RGB)))
            lf = [f.resize(size, Image.LANCZOS) for f in lf]
            lf = (lf + [lf[-1]] * len(idx))[: len(idx)]
            layout_frames = lf
            log(f"layout video: {a.layout_video} ({len(lf)} frames)")
        else:
            layout_frames = (
                g3d["frames"]
                if g3d
                else (
                    pan_video(a.run, report, idx, R, size)
                    if a.layout == "pan"
                    else load([renders[t] for t in idx], size)
                )
            )
        pipe.vae.to("cuda")
        lv = pipe.video_processor.preprocess_video(layout_frames, height=size[1], width=size[0]).to(
            "cuda", torch.float32
        )
        with torch.no_grad():
            z = pipe.vae.encode(lv).latent_dist.mode()
        mean = torch.tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(z)
        std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1).to(z)
        z_layout = ((z - mean) * std).to(torch.bfloat16)
        layout_mask = None
        if a.layout_source in ("self", "hybrid") and g3d is not None:
            # appearance = the prior's own first frame (the conditioned photograph) in latent space, moved along the guide's warp; margins masked out
            lv0 = pipe.video_processor.preprocess_video(
                [load([a.first_frame or keys[0]], size)[0]], height=size[1], width=size[0]
            ).to("cuda", torch.float32)
            with torch.no_grad():
                z0 = pipe.vae.encode(lv0).latent_dist.mode()
            z0 = ((z0 - mean) * std)[:, :, 0].float()  # (1, C, H', W')
            Tl, Hl, Wl = z_layout.shape[2], z_layout.shape[3], z_layout.shape[4]
            T_g = len(g3d["coords"])
            ox_, oy_, PW_, PH_, CW_, CH_ = g3d["photo_box"]
            tgt = []
            msk = []
            for tau in range(Tl):
                t_g = min(round(tau * (T_g - 1) / max(Tl - 1, 1)), T_g - 1)
                cm = torch.nn.functional.interpolate(g3d["coords"][t_g][None].float(), (Hl, Wl), mode="area")[
                    0
                ].to(
                    "cuda"
                )  # (3, H', W')
                gx = (cm[0] * (CW_ - 1) - ox_) / max(PW_ - 1, 1) * 2 - 1
                gy = (cm[1] * (CH_ - 1) - oy_) / max(PH_ - 1, 1) * 2 - 1
                valid = ((gx.abs() <= 1) & (gy.abs() <= 1) & (cm[2] > 0.5)).float()
                grid = torch.stack([gx, gy], -1)[None]
                tgt.append(
                    torch.nn.functional.grid_sample(
                        z0, grid, mode="bilinear", padding_mode="border", align_corners=True
                    )[0]
                )
                msk.append(valid)
            z_self = torch.stack(tgt, 1)[None]  # (1, C, T', H', W') the prior's own content, transported
            m_known = torch.stack(msk, 0)[None, None].to("cuda", torch.float32)  # (1, 1, T', H', W')
            if (
                a.layout_source == "hybrid"
            ):  # canvas appearance only where the photograph is not visible; no mask on the correction afterwards
                z_layout = (m_known * z_self + (1 - m_known) * z_layout.float()).to(torch.bfloat16)
                layout_mask = None
            else:
                z_layout = z_self.to(torch.bfloat16)
                layout_mask = m_known
            log(
                f"layout source: {a.layout_source} (first-frame latent transported along the guide warp); known-region fraction per latent frame "
                f"{[round(float(m.mean()), 2) for m in msk[::max(1, Tl // 6)]]}"
            )
        if a.layout_conf == "stretch" and g3d is not None and layout_mask is None:
            # weight the correction by the guide's source density: where the splat magnifies the texture (a nearby
            # surface under a rising camera) or pull-push filled a hole, neighbouring output pixels sample nearly the
            # same chart coordinates, and the correction is attenuated there instead of imprinting stretched low bands
            Tl, Hl, Wl = z_layout.shape[2], z_layout.shape[3], z_layout.shape[4]
            T_g = len(g3d["coords"])
            ox_, oy_, PW_, PH_, CW_, CH_ = g3d["photo_box"]
            exp_x, exp_y = PW_ / size[0], PH_ / size[1]  # chart px per output px for unmagnified content
            ms = []
            for tau in range(Tl):
                t_g = min(round(tau * (T_g - 1) / max(Tl - 1, 1)), T_g - 1)
                cm = g3d["coords"][t_g].float()
                gx = torch.gradient(cm[0] * (CW_ - 1), dim=1)[0].abs()
                gy = torch.gradient(cm[1] * (CH_ - 1), dim=0)[0].abs()
                conf = (gx / max(exp_x, 1e-6)).clamp(max=1.0) * (gy / max(exp_y, 1e-6)).clamp(max=1.0)
                ms.append(torch.nn.functional.interpolate(conf[None, None], (Hl, Wl), mode="area")[0, 0])
            layout_mask = torch.stack(ms, 0)[None, None].to("cuda", torch.float32).clamp(0, 1)
            log(
                f"stretch confidence: correction weighted by the guide's source density; mean per latent frame "
                f"{[round(float(m.mean()), 2) for m in ms[::max(1, Tl // 6)]]}"
            )
        del lv, z
        torch.cuda.empty_cache()
    S.z_layout, S.layout_mask = z_layout, layout_mask


def move_pipeline_to_gpu(a, S, log):
    """Moves the pipeline to the GPU (or enables CPU offload) after the layout encoding; logs the time."""
    pipe, t0 = S.pipe, S.t0
    if a.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    log(f"model loaded in {time.time() - t0:.0f} s")


def prepare_layout_steering(a, S, log):
    """Builds the step-end callback: low-pass replacement from the noised layout latent (or CamTrol)."""
    z_layout, layout_mask = S.z_layout, S.layout_mask
    callback = None
    if a.layout != "none":
        eps = torch.randn(
            z_layout.shape,
            generator=torch.Generator("cuda").manual_seed(a.seed + 1),
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = max(1, int(round(3 * a.layout_sigma)))
        xs = torch.arange(-k, k + 1, device="cuda", dtype=torch.float32)
        g1 = torch.exp(-(xs**2) / (2 * a.layout_sigma**2))
        g1 = g1 / g1.sum()

        def lowpass(x):  # separable Gaussian over the spatial latent dims, per frame and channel
            B, C, T, H, W = x.shape
            y = x.float().permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            y = torch.nn.functional.conv2d(
                torch.nn.functional.pad(y, (k, k, 0, 0), mode="reflect"),
                g1.view(1, 1, 1, -1).expand(C, 1, 1, -1),
                groups=C,
            )
            y = torch.nn.functional.conv2d(
                torch.nn.functional.pad(y, (0, 0, k, k), mode="reflect"),
                g1.view(1, 1, -1, 1).expand(C, 1, -1, 1),
                groups=C,
            )
            return y.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).to(x.dtype)

        def callback(pipe_, i, t, kw):
            lat = kw["latents"]
            if (
                a.layout == "camtrol"
            ):  # CamTrol baseline: restart = the sampler is restarted once from the layout video noised to the level of step K (CamTrol's noise layout prior, t0 = K); hold = the latent is replaced at every step <= K (over-constrained variant)
                if (i + 1 == a.layout_steps) if a.camtrol_mode == "restart" else (i + 1 <= a.layout_steps):
                    sigmas = getattr(pipe_.scheduler, "sigmas", None)
                    sig = (
                        float(sigmas[i + 1])
                        if sigmas is not None and len(sigmas) > i + 1
                        else float(t) / 1000.0
                    )
                    off = lat.shape[2] - z_layout.shape[2]
                    lat = torch.cat(
                        [lat[:, :, :off], ((1 - sig) * z_layout.float() + sig * eps.float()).to(lat.dtype)], 2
                    )
                return {"latents": lat}
            if i + 1 < a.layout_steps:  # latents now sit at sigma of step i+1
                sigmas = getattr(pipe_.scheduler, "sigmas", None)
                sig = (
                    float(sigmas[i + 1]) if sigmas is not None and len(sigmas) > i + 1 else float(t) / 1000.0
                )
                z_t = (1 - sig) * z_layout + sig * eps
                w = (
                    a.layout_weight * (1 - (i + 1) / a.layout_steps)
                    if a.layout_schedule == "ramp"
                    else a.layout_weight
                )  # ramp: linear decay to 0 at K; const: full weight for K steps then off
                off = (
                    lat.shape[2] - z_layout.shape[2]
                )  # VACE prepends one latent frame per reference image: leave it alone
                part = lat[:, :, off:]
                lp_g = lowpass(z_t.to(lat.dtype)).float()
                lp_l = lowpass(part).float()
                if a.layout_tone == "keep":  # contrast standardisation
                    mg, sg = lp_g.mean(dim=(-2, -1), keepdim=True), lp_g.std(dim=(-2, -1), keepdim=True)
                    ml, sl = lp_l.mean(dim=(-2, -1), keepdim=True), lp_l.std(dim=(-2, -1), keepdim=True)
                    lp_g = (lp_g - mg) / (sg + 1e-6) * sl + ml
                corr = lp_g - lp_l
                if (
                    layout_mask is not None
                ):  # motion-only steering: correct only where the prior's own content is known (soft mask at the low-pass scale)
                    m_ = lowpass(layout_mask.to(lat.dtype)).float().clamp(0, 1)
                    if a.layout_tone == "mean":
                        corr = corr - (corr * m_).sum(dim=(-2, -1), keepdim=True) / (
                            m_.sum(dim=(-2, -1), keepdim=True) + 1e-6
                        )
                    corr = corr * m_
                elif (
                    a.layout_tone == "mean"
                ):  # tone-neutral: the correction carries the layout, not the guide's exposure or colour cast
                    corr = corr - corr.mean(dim=(-2, -1), keepdim=True)
                corr = corr.to(lat.dtype)
                lat = torch.cat([lat[:, :, :off], part + w * corr], 2)
            return {"latents": lat}

        log(
            f"layout guidance from {a.layout}: {a.layout_steps} steps, sigma {a.layout_sigma} latent px, weight {a.layout_weight}, tone {a.layout_tone}"
        )
    S.callback = callback


def check_noise_transport_move(a, S, log):
    """Turns noise transport off for a forward move without a 3D guide (the canvas pan only knows shifts)."""
    report, g3d = S.report, S.g3d
    if report.get("move") == "forward" and a.noise_transport != "none" and g3d is None:
        log(
            "forward move: noise transport only knows pans; disabled for this run (layout steering and the rotary zoom stay)"
        )
        a.noise_transport = "none"


def install_rope_shift(a, S, log):
    """Patches the rotary embedding so every latent frame's positions follow the camera (--rope-shift pan)."""
    pipe, reference, keys, size = S.pipe, S.reference, S.keys, S.size
    report, g3d, idx, R = S.report, S.g3d, S.idx, S.R
    if a.rope_shift == "pan":
        import math as _math

        rope = pipe.transformer.rope
        original_rope = rope.forward
        rope2 = getattr(
            getattr(pipe, "transformer_2", None), "rope", None
        )  # Wan 2.2: the low-noise expert has its own rope module
        original_rope2 = rope2.forward if rope2 is not None else None
        canvas_p = (a.run / "frames" / "canvas-000.png") if a.run else None
        if canvas_p is not None and canvas_p.exists():
            canvas_ = Image.open(canvas_p)
            photo_ = Image.open(keys[0])
            mx_tok = (
                (canvas_.width - photo_.width) * size[0] / photo_.width / 16.0
            )  # pan margin in tokens (patch 2 on the 8x latent = 16 px)
            my_tok = (canvas_.height - photo_.height) * size[1] / photo_.height / 16.0
        else:
            mx_tok = my_tok = 0.0
        move_r = report.get("move", "truck_right")
        n_ref = 1 if reference is not None else 0

        def offsets(ppf):
            """per latent frame: rotary position offset (tokens) along W and H and a zoom factor about the image centre. A token showing
            canvas content q must carry position q: pans add the pan offset, a forward move (pan_video's crop fraction z = 1 - f/2)
            contracts positions toward the centre by z so that frame t's tokens meet frame 0's at the same scene point.
            """
            out = []
            for j in range(ppf):
                k = j - n_ref
                if k < 0:
                    out.append((0.0, 0.0, 1.0))
                    continue
                if g3d is not None:
                    t_ = min(len(idx) - 1, 4 * k)
                    px, py = g3d["pan_px"][t_]
                    z = 1.0 / max(g3d["mag"][t_], 1e-3)
                    out.append(
                        (a.rope_scale * px / 16.0, a.rope_scale * py / 16.0, 1.0 - a.rope_scale * (1.0 - z))
                    )
                    continue
                f = (idx[0] + min(len(idx) - 1, 4 * k)) / max(R, 1)
                ox = (
                    f * mx_tok
                    if move_r == "truck_right"
                    else ((1 - f) * mx_tok if move_r == "truck_left" else 0.0)
                )
                oy = (1 - f) * my_tok if move_r == "rise" else 0.0
                z = 1.0 - 0.5 * f if move_r == "forward" else 1.0
                out.append((a.rope_scale * ox, a.rope_scale * oy, 1.0 - a.rope_scale * (1.0 - z)))
            return out

        def steered(hidden_states):
            nonlocal original_rope
            cos, sin = original_rope(hidden_states)
            B, C, F_, H_, W_ = hidden_states.shape
            p_t, p_h, p_w = rope.patch_size
            ppf, pph, ppw = F_ // p_t, H_ // p_h, W_ // p_w
            dev_ = cos.device
            offs = torch.tensor(offsets(ppf), dtype=torch.float64, device=dev_)  # (ppf, 3): dx, dy, zoom
            for axis, dim, n_pos, off in (
                ("w", rope.w_dim, ppw, offs[:, 0]),
                ("h", rope.h_dim, pph, offs[:, 1]),
            ):
                zoom = offs[:, 2]
                if float(off.abs().max()) == 0.0 and float((zoom - 1).abs().max()) == 0.0:
                    continue
                inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float64, device=dev_) / dim))
                base = torch.arange(n_pos, dtype=torch.float64, device=dev_)[None, :]
                centre = (n_pos - 1) / 2.0
                pos = (
                    centre + (base - centre) * zoom[:, None] + off[:, None]
                )  # (ppf, n): contracted about the centre, then shifted
                ang = pos[..., None] * inv  # (ppf, n, dim/2)
                c = torch.cos(ang).repeat_interleave(2, -1)
                s_ = torch.sin(ang).repeat_interleave(2, -1)
                if axis == "w":
                    c = c[:, None, :, :].expand(ppf, pph, ppw, dim)
                    s_ = s_[:, None, :, :].expand(ppf, pph, ppw, dim)
                    sl = slice(rope.t_dim + rope.h_dim, rope.t_dim + rope.h_dim + dim)
                else:
                    c = c[:, :, None, :].expand(ppf, pph, ppw, dim)
                    s_ = s_[:, :, None, :].expand(ppf, pph, ppw, dim)
                    sl = slice(rope.t_dim, rope.t_dim + dim)
                cos = cos.clone()
                sin = sin.clone()
                cos[..., sl] = c.reshape(1, ppf * pph * ppw, 1, dim).to(cos.dtype)
                sin[..., sl] = s_.reshape(1, ppf * pph * ppw, 1, dim).to(sin.dtype)
            return cos, sin

        rope.forward = steered
        if rope2 is not None:

            def steered2(hidden_states, _orig=original_rope2):
                nonlocal original_rope
                saved = original_rope
                original_rope = _orig
                try:
                    return steered(hidden_states)
                finally:
                    original_rope = saved

            rope2.forward = steered2
        log(
            f"rope shift: pan margin {mx_tok:.1f} x {my_tok:.1f} tokens across the clip"
            + (" (both experts)" if rope2 is not None else "")
        )


def make_init_latents(a, S, log):
    """Builds the initial noise: a window over one noise field (pan) or gathered by guide coords (flow)."""
    pipe, idx, size, g3d = S.pipe, S.idx, S.size, S.g3d
    keys, report, R, reference = S.keys, S.report, S.R, S.reference
    init_latents = None
    C_lat = int(getattr(pipe.vae.config, "z_dim", 16))
    S_lat = int(
        getattr(pipe, "vae_scale_factor_spatial", 8) or 8
    )  # Wan 2.1 family: 16 ch, 8x; Wan 2.2 TI2V-5B: 48 ch, 16x
    if a.noise_transport == "pan":
        # latent grid of the clip: (1 + (T-1)/4) frames, H/8 x W/8 (Wan VAE: 4x temporal, 8x spatial), 16 channels; the canvas pan shifts by
        # (frame t / total) * margin px -> in latent px /8; one noise field covers the whole canvas width, each latent frame is a crop
        T_lat = 1 + (len(idx) - 1) // 4
        Hl, Wl = size[1] // S_lat, size[0] // S_lat
        gen = torch.Generator("cuda").manual_seed(a.seed)
        if (
            g3d is not None
        ):  # per latent frame: the guide's camera displacement in latent cells (content shifts the other way)
            offs = [
                (
                    -int(round(g3d["pan_px"][min(len(idx) - 1, 4 * j)][0] / S_lat)),
                    -int(round(g3d["pan_px"][min(len(idx) - 1, 4 * j)][1] / S_lat)),
                )
                for j in range(T_lat)
            ]
            minx, miny = min(o[0] for o in offs), min(o[1] for o in offs)
            maxx, maxy = max(o[0] for o in offs), max(o[1] for o in offs)
            field = torch.randn(
                1,
                C_lat,
                T_lat,
                Hl + (maxy - miny) + 2,
                Wl + (maxx - minx) + 2,
                generator=gen,
                device="cuda",
                dtype=torch.float32,
            )
            frames_lat = [
                field[:, :, j, oy - miny : oy - miny + Hl, ox - minx : ox - minx + Wl]
                for j, (ox, oy) in enumerate(offs)
            ]
            mx = my = float("nan")
        else:
            canvas = Image.open(a.run / "frames" / "canvas-000.png")
            photo0 = Image.open(keys[0])
            CW, CH = canvas.size
            PW, PH = photo0.size
            mx, my = (CW - PW) * size[0] / PW, (CH - PH) * size[1] / PH  # margins in generation px
            field = torch.randn(
                1,
                C_lat,
                T_lat,
                Hl + int(round(my / S_lat)) + 2,
                Wl + int(round(mx / S_lat)) + 2,
                generator=gen,
                device="cuda",
                dtype=torch.float32,
            )
            move_ = report.get("move", "truck_right")
            frames_lat = []
            for j in range(T_lat):
                f = (idx[0] + min(len(idx) - 1, 4 * j)) / max(R, 1)
                ox = (
                    int(round(f * mx / S_lat))
                    if move_ == "truck_right"
                    else (int(round((1 - f) * mx / S_lat)) if move_ == "truck_left" else 0)
                )
                oy = int(round((1 - f) * my / S_lat)) if move_ == "rise" else 0
                frames_lat.append(field[:, :, j, oy : oy + Hl, ox : ox + Wl])
        init_latents = torch.stack(frames_lat, 2)
    elif a.noise_transport == "flow":
        # Go-with-the-Flow-style warped noise WITHOUT its fine-tuning: one Gaussian field lives on the chart at latent density; latent frame j takes,
        # at every cell, the field value at the chart coordinate the guide's coordinate map sends that cell to (nearest cell: every sample stays a
        # unit Gaussian, correlated along the camera motion). Needs --guide3d (the coordinate maps).
        if g3d is None:
            raise SystemExit("--noise-transport flow needs --guide3d (the guide's coordinate maps)")
        T_lat = 1 + (len(idx) - 1) // 4
        Hl, Wl = size[1] // S_lat, size[0] // S_lat
        gen = torch.Generator("cuda").manual_seed(a.seed)
        ox_, oy_, PW_, PH_, CW_, CH_ = g3d["photo_box"]
        Hc, Wc = max(Hl, int(round(CH_ / PH_ * Hl))), max(Wl, int(round(CW_ / PW_ * Wl)))
        field = torch.randn(1, C_lat, Hc, Wc, generator=gen, device="cuda", dtype=torch.float32)
        frames_lat = []
        for j in range(T_lat):
            cm = torch.nn.functional.interpolate(
                g3d["coords"][min(len(idx) - 1, 4 * j)][None].float(), (Hl, Wl), mode="area"
            )[0].to(
                "cuda"
            )  # (3, Hl, Wl) chart x, y in [0, 1]
            cx = (cm[0] * (Wc - 1)).round().long().clamp(0, Wc - 1)
            cy = (cm[1] * (Hc - 1)).round().long().clamp(0, Hc - 1)
            frames_lat.append(field[0][:, cy, cx][None])  # (1, 16, Hl, Wl)
        init_latents = torch.stack(frames_lat, 2)
        mx = my = float("nan")
        log(
            f"noise transport (flow): {T_lat} latent frames gathered from one {Wc}x{Hc} noise field through the guide's coordinate maps"
        )
    if a.noise_transport != "none":
        if reference is not None:  # VACE prepends one latent frame per reference image
            init_latents = torch.cat(
                [
                    torch.randn(1, C_lat, 1, Hl, Wl, generator=gen, device="cuda", dtype=torch.float32),
                    init_latents,
                ],
                2,
            )
        init_latents = init_latents.to(torch.bfloat16)
        log(
            f"noise transport: {T_lat} latent frames sliding over a {field.shape[-1]}x{field.shape[-2]} noise field (margin {mx:.0f}x{my:.0f} px); init latents {tuple(init_latents.shape)} mean {init_latents.float().mean():.3f} std {init_latents.float().std():.3f} nan {int(torch.isnan(init_latents).sum())}"
        )
    S.init_latents = init_latents


def generate(a, S, log):
    """Runs the pipeline (I2V from the first frame, or VACE with video/masks/reference); keeps raw frames."""
    video, canvas_crops, keys, size, pipe, prompt = S.video, S.canvas_crops, S.keys, S.size, S.pipe, S.prompt
    idx, init_latents, callback, masks, reference = S.idx, S.init_latents, S.callback, S.masks, S.reference
    t0 = time.time()
    if a.backend == "i2v":
        first = (
            video[0]
            if a.guide != "none" or a.first_frame is None or canvas_crops is not None
            else load([a.first_frame], size)[0]
        )  # canvas-tone: the matched frame
        if a.guide == "none" and a.first_frame is None:
            first = load([keys[0]], size)[0]
        out = pipe(
            image=first,
            prompt=prompt,
            negative_prompt=NEGATIVE,
            height=size[1],
            width=size[0],
            num_frames=len(idx),
            num_inference_steps=a.steps,
            guidance_scale=a.guidance,
            latents=init_latents,
            generator=torch.Generator("cuda").manual_seed(a.seed),
            output_type="pil",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        ).frames[0]
    else:
        out = pipe(
            video=video,
            mask=masks,
            reference_images=None if reference is None else [reference],
            latents=init_latents,
            prompt=prompt,
            negative_prompt=NEGATIVE,
            height=size[1],
            width=size[0],
            num_frames=len(idx),
            num_inference_steps=a.steps,
            guidance_scale=a.guidance,
            conditioning_scale=a.conditioning_scale,
            generator=torch.Generator("cuda").manual_seed(a.seed),
            output_type="pil",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        ).frames[0]
    log(f"VACE {len(idx)} frames at {size} in {time.time() - t0:.0f} s")
    raw_out = list(out)
    S.out, S.raw_out = out, raw_out


def postprocess_output(a, S, log):
    """Applies the optional tone match, canvas tone and anchor paste (crossfade) to the generated frames."""
    out, video, canvas_crops, anchors_in, idx = S.out, S.video, S.canvas_crops, S.anchors_in, S.idx
    if a.tone != "none":
        out = [tone_match(v, r, a.tone) for v, r in zip(out, video)]
    if a.canvas_tone == "all" and canvas_crops is not None:
        out = [tone_match(v, r, "affine") for v, r in zip(out, canvas_crops)]
        log("canvas tone: every output frame matched to its canvas crop")
    if a.paste_anchors:
        matched = list(out)
        for t in anchors_in:
            i = idx.index(t)
            out[i] = video[i]
            for d in range(
                1, a.anchor_blend + 1
            ):  # linear crossfade from the pasted keyframe into the VACE frames on both sides
                w = 1 - d / (a.anchor_blend + 1)
                for j in (i - d, i + d):
                    if 0 <= j < len(out) and idx[j] not in anchors_in:
                        ref = tone_match(video[i], matched[j], "affine") if a.tone != "none" else video[i]
                        out[j] = Image.fromarray(
                            np.clip(
                                (1 - w) * np.asarray(matched[j], np.float32)
                                + w * np.asarray(video[i], np.float32),
                                0,
                                255,
                            ).astype(np.uint8)
                        )
    S.out = out


def write_frames(a, S):
    """Writes frames_raw/ (the prior's frames) and frames/ (after post-processing)."""
    idx, raw_out, out = S.idx, S.raw_out, S.out
    (a.out / "frames_raw").mkdir(exist_ok=True)
    for t, im in zip(idx, raw_out):
        im.save(a.out / "frames_raw" / f"vace-{t:03d}.png")
    (a.out / "frames").mkdir(exist_ok=True)
    for t, im in zip(idx, out):
        im.save(a.out / "frames" / f"vace-{t:03d}.png")


def write_metrics(a, S, log):
    """Computes temporal residuals, anchor reproduction error, output-vs-guide flow; writes metrics.json."""
    rgb_video, out, raw_out, idx = S.rgb_video, S.out, S.raw_out, S.idx
    anchors_in, video, size = S.anchors_in, S.video, S.size
    # metrics
    gr = [gray(v) for v in rgb_video]
    gv = [gray(v) for v in out]
    pr, fr = flow_residual(gr)
    pv, fv = flow_residual(gv)
    graw = [gray(v) for v in raw_out]
    anchor_err = {str(t): float(np.abs(gr[idx.index(t)] - graw[idx.index(t)]).mean()) for t in anchors_in}
    anchor_err_matched = {
        str(t): float(
            np.abs(
                gr[idx.index(t)]
                - gray(
                    tone_match(raw_out[idx.index(t)], video[idx.index(t)], a.tone)
                    if a.tone != "none"
                    else raw_out[idx.index(t)]
                )
            ).mean()
        )
        for t in anchors_in
    }
    dev = [flow_between(gr[i], gv[i]) for i in range(len(idx))]
    m = dict(
        run=str(a.run),
        backend=a.backend,
        mode=a.mode,
        guide=a.guide,
        layout=a.layout,
        layout_steps=a.layout_steps,
        layout_tone=a.layout_tone,
        layout_source=a.layout_source,
        layout_conf=a.layout_conf,
        canvas_tone=a.canvas_tone,
        noise_transport=a.noise_transport,
        rope_shift=a.rope_shift,
        reference=a.reference,
        model=a.model,
        window=[idx[0], idx[-1]],
        size=size,
        steps=a.steps,
        guidance=a.guidance,
        anchors=list(anchors_in),
        render=dict(
            t_plain=float(pr.mean()), t_flow=float(fr.mean()), t_flow_p90=float(np.percentile(fr, 90))
        ),
        vace=dict(t_plain=float(pv.mean()), t_flow=float(fv.mean()), t_flow_p90=float(np.percentile(fv, 90))),
        anchor_err_gray=anchor_err,
        anchor_err_matched=anchor_err_matched,
        tone=a.tone,
        camera_dev_px_mean=float(np.mean(dev)),
        camera_dev_px_max=float(np.max(dev)),
        per_frame=dict(render_flow=fr.tolist(), vace_flow=fv.tolist(), dev=dev),
    )
    (a.out / "metrics.json").write_text(json.dumps(m, indent=1))
    log(json.dumps({k: v for k, v in m.items() if k != "per_frame"}))


def write_compare(a, S, log):
    """Writes the side-by-side compare.mp4, video.mp4, video_raw.mp4 and the y-t slices.png."""
    rgb_video, out, size, idx = S.rgb_video, S.out, S.size, S.idx
    # side-by-side video and y-t slices
    (a.out / "compare").mkdir(exist_ok=True)
    for i, (r, v) in enumerate(zip(rgb_video, out)):
        canvas = Image.new("RGB", (2 * size[0] + 8, size[1]), (255, 255, 255))
        canvas.paste(r, (0, 0))
        canvas.paste(v, (size[0] + 8, 0))
        canvas.save(a.out / "compare" / f"cmp-{i:03d}.png")
    subprocess.run(
        [
            ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(a.fps),
            "-i",
            str(a.out / "compare" / "cmp-%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(a.out / "compare.mp4"),
        ],
        check=True,
    )
    subprocess.run(
        [
            ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(a.fps),
            "-start_number",
            str(idx[0]),
            "-i",
            str(a.out / "frames" / "vace-%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(a.out / "video.mp4"),
        ],
        check=True,
    )
    subprocess.run(
        [
            ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(a.fps),
            "-start_number",
            str(idx[0]),
            "-i",
            str(a.out / "frames_raw" / "vace-%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(a.out / "video_raw.mp4"),
        ],
        check=True,
    )
    rows = (0.35, 0.55, 0.8)
    panels = []
    for frames in (rgb_video, out):
        arr = np.stack([np.asarray(f.resize((512, int(round(size[1] * 512 / size[0]))))) for f in frames])
        for r in rows:
            panels.append(arr[:, int(r * arr.shape[1])])
            panels.append(np.full((4, 512, 3), 255, np.uint8))
    Image.fromarray(np.concatenate(panels, 0)).save(a.out / "slices.png")
    log("done")


def main():
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    log_f = open(a.out / "log.txt", "a")

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    # state shared by the stages: each reads what it needs from S and writes back what it produces
    S = types.SimpleNamespace()
    load_inputs(a, S, log)
    build_guide_video(a, S, log)
    load_reference(a, S, log)
    build_prompt(a, S, log)
    load_pipeline(a, S, log)
    render_guide3d(a, S, log)
    encode_layout_frames(a, S, log)
    move_pipeline_to_gpu(a, S, log)
    prepare_layout_steering(a, S, log)
    check_noise_transport_move(a, S, log)
    install_rope_shift(a, S, log)
    make_init_latents(a, S, log)
    generate(a, S, log)
    postprocess_output(a, S, log)
    write_frames(a, S)
    write_metrics(a, S, log)
    write_compare(a, S, log)


if __name__ == "__main__":
    main()
