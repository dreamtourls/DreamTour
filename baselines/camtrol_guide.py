"""Faithful CamTrol layout video (Hou et al. 2024, Algorithm 1): ZoeDepth on the input image -> point cloud -> for every camera of the path:
z-buffered render of the accumulated cloud, Stable Diffusion inpainting of the holes under the scene prompt (25 steps), ZoeDepth on the
inpainted frame with an l1 scale fit to the rendered depth on the known pixels, the new pixels lifted into the cloud. The result is the
"rearranged image" sequence V0 that CamTrol noises to t0 (scripts/video_between.py --layout camtrol --camtrol-mode restart --layout-video).

Conventions: cameras = dreamtour.guide.camera_path (the same requested motions as the layout guides, fov 50); ZoeDepth is relative to its own scale,
so the first depth map is rescaled to median = --nominal (4 m) exactly as the layout guides are, otherwise a 0.576 m truck would mean a
different image shift. Models: Intel/zoedepth-nyu-kitti (transformers) and a Stable Diffusion inpainting pipeline (diffusers): CamTrol's
runwayml/stable-diffusion-inpainting, falling back to stable-diffusion-v1-5/stable-diffusion-inpainting and
diffusers/stable-diffusion-xl-1.0-inpainting-0.1, or the repository given by --inpainter.

    python baselines/camtrol_guide.py --image examples/f19.png --prompt "..." --move truck_right --amount 0.576 --frames 49 --out runs/f19_camtrol_truck_right
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dreamtour.guide import camera_path


def zoe_depth(model, proc, img: Image.Image, device):
    with torch.no_grad():
        inputs = proc(images=img, return_tensors="pt").to(device)
        out = model(**inputs)
        d = proc.post_process_depth_estimation(
            out, source_sizes=[(img.height, img.width)], target_sizes=[(img.height, img.width)]
        )[0]["predicted_depth"]
    return d.float().to(device)


def render_cloud(pts, cols, cam, footprint=1):
    """z-buffered splat of the world points into cam: (3,H,W) colours in [0,1], (H,W) depth (inf where empty)."""
    u, v, z, front = cam.project(pts)
    H, W = cam.H, cam.W
    dev = pts.device
    canvas = torch.zeros(3, H, W, device=dev)
    zbuf = torch.full((H, W), float("inf"), device=dev)
    order = torch.argsort(z, descending=True)
    u0, v0, zo, co, keep = u[order], v[order], z[order], cols[order], front[order]
    r = footprint // 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ui = (u0 + dx).round().long()
            vi = (v0 + dy).round().long()
            ok = keep & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            idx = vi[ok] * W + ui[ok]
            canvas.view(3, -1)[:, idx] = co[ok].T
            zbuf.view(-1)[idx] = zo[ok]
    return canvas, zbuf


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--move", required=True)
    p.add_argument("--amount", type=float, required=True)
    p.add_argument("--frames", type=int, default=49)
    p.add_argument("--nominal", type=float, default=4.0)
    p.add_argument("--fov", type=float, default=50.0)
    p.add_argument("--size", type=int, nargs=2, default=None, help="render size W H (default: the image's)")
    p.add_argument(
        "--inpainter",
        default=None,
        help="HF repo of a Stable Diffusion inpainting pipeline (default: runwayml/stable-diffusion-inpainting, then the stable-diffusion-v1-5 mirror, then SDXL inpainting)",
    )
    p.add_argument("--inpaint-steps", type=int, default=25)
    p.add_argument(
        "--min-hole", type=float, default=0.002, help="skip inpainting when the hole fraction is below this"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    dev = "cuda"
    img = Image.open(a.image).convert("RGB")
    if a.size:
        img = img.resize(tuple(a.size), Image.LANCZOS)
    W, H = img.size
    W8, H8 = W - W % 8, H - H % 8
    if (W8, H8) != (W, H):
        img = img.resize((W8, H8), Image.LANCZOS)
        W, H = W8, H8
    from diffusers import StableDiffusionInpaintPipeline
    from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation

    proc = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
    zoe = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").to(dev).eval()
    inp = None  # CamTrol uses Runway's SD 1.5 inpainting; the v1-5 mirror and SDXL inpainting are fallbacks
    for repo in ([a.inpainter] if a.inpainter else []) + [
        "runwayml/stable-diffusion-inpainting",
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    ]:
        try:
            inp = StableDiffusionInpaintPipeline.from_pretrained(
                repo, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
            ).to(dev)
            print("inpainter:", repo)
            break  # no safety checker: it blacks out frames of harmless scenes
        except Exception as e:
            print("inpainter", repo, "unavailable:", str(e)[:100])
    if inp is None:
        raise SystemExit("no inpainting model available")
    inp.set_progress_bar_config(disable=True)
    cams = camera_path(a.move, a.amount, a.frames, H, W, a.nominal, a.fov)
    cam0 = cams[0]
    d0 = zoe_depth(zoe, proc, img, dev)
    d0 = d0 * (a.nominal / d0.median())  # ZoeDepth is relative: the same scale convention as dreamtour.guide
    img01 = torch.from_numpy(np.asarray(img, np.float32) / 255.0).permute(2, 0, 1).to(dev)
    pts, pix = cam0.unproject(d0)
    cols = img01[:, pix[:, 0], pix[:, 1]].T
    a.out.mkdir(parents=True, exist_ok=True)
    frames = []
    gen = torch.Generator(dev).manual_seed(a.seed)
    holes = []
    for i, cam in enumerate(cams):
        canvas, zbuf = render_cloud(pts, cols, cam)
        valid = torch.isfinite(zbuf)
        hole = (~valid).float().mean().item()
        holes.append(hole)
        frame = canvas
        if hole > a.min_hole:
            mask = Image.fromarray(((~valid).cpu().numpy() * 255).astype(np.uint8))
            init = Image.fromarray((canvas.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            res = inp(
                prompt=a.prompt,
                image=init,
                mask_image=mask,
                num_inference_steps=a.inpaint_steps,
                generator=gen,
                height=H,
                width=W,
            ).images[0]
            frame = (
                torch.from_numpy(np.asarray(res.resize((W, H)), np.float32) / 255.0).permute(2, 0, 1).to(dev)
            )
            frame = torch.where(
                valid[None], canvas, frame
            )  # keep the rendered pixels, take the inpainting only in the holes
            # lift the new pixels: ZoeDepth of the completed frame, l1 scale fit on the known pixels (weighted median of the ratio), then unproject the holes
            di = zoe_depth(
                zoe, proc, Image.fromarray((frame.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)), dev
            )
            ratio = zbuf[valid] / di[valid].clamp(min=1e-3)
            s = ratio.median()
            dnew = torch.where(valid, zbuf, di * s)
            pn, pixn = cam.unproject(dnew)
            sel = (~valid)[pixn[:, 0], pixn[:, 1]]
            pts = torch.cat([pts, pn[sel]])
            cols = torch.cat([cols, frame[:, pixn[sel, 0], pixn[sel, 1]].T])
        frames.append((frame.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    for i, f in enumerate(frames):
        Image.fromarray(f).save(a.out / f"frame-{i:03d}.png")
    import subprocess

    import imageio_ffmpeg

    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            "16",
            "-i",
            str(a.out / "frame-%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "12",
            str(a.out / "guide.mp4"),
        ],
        check=True,
    )
    json.dump(
        dict(
            image=str(a.image),
            move=a.move,
            amount=a.amount,
            frames=a.frames,
            nominal=a.nominal,
            fov=a.fov,
            size=[W, H],
            depth="zoedepth",
            inpaint=getattr(inp, "name_or_path", "sd-inpainting"),
            inpaint_steps=a.inpaint_steps,
            hole_fraction=holes,
            points=int(pts.shape[0]),
        ),
        open(a.out / "guide.json", "w"),
        indent=1,
    )
    print(
        a.out / "guide.mp4",
        f"{len(frames)} frames, mean hole fraction {np.mean(holes):.3f}, {pts.shape[0]} points",
    )


if __name__ == "__main__":
    main()
