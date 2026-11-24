"""CogVideoX 1.5 5B I2V with layout steering: the first K denoising steps replace the Gaussian low-pass of the running latent by the low-pass
of the guide video's latent noised to the next step's level (tone-neutral: the per-frame per-channel spatial mean of the correction is removed),
weight decaying linearly to zero. Output layout as video_between.py (frames_raw/vace-NNN.png, video_raw.mp4, metrics.json). Model-card settings:
1360x768, 81 frames, 50 steps, guidance 6 with dynamic cfg. The guide (mp4 or frame directory) is resampled to --frames and resized to --size.

    python scripts/cog_layout.py --image photo.png --prompt "..." --guide runs/guides/bev/guide.mp4 --out runs/cog/bev_layout
    python scripts/cog_layout.py --image photo.png --prompt "..." --out runs/cog/bev_text          # no guide = text only
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def log(m):
    print(f"[cog_layout {time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_frames(path: Path, n: int, size) -> list[Image.Image]:
    if path.is_dir():
        fr = [Image.open(q).convert("RGB") for q in sorted(glob.glob(str(path / "*.png")))]
    else:
        import cv2

        cap = cv2.VideoCapture(str(path))
        fr = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            fr.append(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    idx = [round(k * (len(fr) - 1) / max(n - 1, 1)) for k in range(n)]
    return [fr[k].resize(size, Image.LANCZOS) for k in idx]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--guide", type=Path, default=None, help="layout guide video (mp4 or frame dir); omitted = text only"
    )
    p.add_argument("--layout-steps", type=int, default=15)
    p.add_argument("--layout-sigma", type=float, default=3.0, help="latent cells (8 px each at 1360x768)")
    p.add_argument("--layout-weight", type=float, default=1.0)
    p.add_argument("--frames", type=int, default=81)
    p.add_argument("--size", type=int, nargs=2, default=(1360, 768))
    p.add_argument(
        "--layout-ramp-end",
        type=float,
        default=1.0,
        help="fraction of the latent frames over which the ramp rises to 1 (1.0 = linear over the whole clip; 0.5 = full correction from mid-clip)",
    )
    p.add_argument(
        "--layout-ramp",
        action="store_true",
        help="scale the correction linearly over the latent frames (0 at the conditioned first frame, 1 at the last) so the content does not jump after frame 0",
    )
    p.add_argument(
        "--layout-source",
        choices=("guide", "hybrid"),
        default="guide",
        help="hybrid: where the photograph is visible, the layout latent is the prior's OWN first-frame latent transported along the guide's warp (needs --coords); the guide's canvas content only in revealed regions",
    )
    p.add_argument(
        "--coords",
        type=Path,
        default=None,
        help="coords.npz written by guide_video.py --coords (same move, size and frame count as --guide)",
    )
    p.add_argument(
        "--layout-tone",
        choices=("keep", "mean", "raw"),
        default="mean",
        help="mean (default): mean-neutral correction; keep: also match the std (weaker steering and can flatten contrast into bleached texture); raw: plain low-pass difference",
    )
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="THUDM/CogVideoX1.5-5B-I2V")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    W, H = a.size
    from diffusers import CogVideoXImageToVideoPipeline

    tok = None
    try:
        from transformers import T5Tokenizer

        tok = T5Tokenizer.from_pretrained(a.model, subfolder="tokenizer")
    except Exception as e:
        log(f"slow tokenizer unavailable ({str(e)[:60]})")
    t0 = time.time()
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        a.model, torch_dtype=torch.bfloat16, **({"tokenizer": tok} if tok is not None else {})
    )
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    log(f"model loaded in {time.time() - t0:.0f} s")
    image = Image.open(a.image).convert("RGB").resize((W, H), Image.LANCZOS)
    cb = None
    if a.guide is not None:
        frames = read_frames(a.guide, a.frames, (W, H))
        vid = torch.stack(
            [
                torch.from_numpy(np.asarray(f, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)
                for f in frames
            ],
            1,
        )[
            None
        ]  # 1,3,T,H,W
        with torch.no_grad():
            zg = (
                pipe.vae.encode(vid.to("cuda", torch.bfloat16)).latent_dist.sample()
                * pipe.vae.config.scaling_factor
            )  # 1,C,T',H',W'
        zg = zg.permute(0, 2, 1, 3, 4).contiguous()  # CogVideoX latents are (B, T, C, H, W)
        log(
            f"guide latent {tuple(zg.shape)}; K={a.layout_steps}, sigma={a.layout_sigma} cells, w0={a.layout_weight}"
        )
        if a.layout_source == "hybrid":
            if a.coords is None or not a.coords.exists():
                raise SystemExit("--layout-source hybrid needs --coords coords.npz (guide_video.py --coords)")
            import torch.nn.functional as Fn

            cz = np.load(a.coords)
            coords = torch.from_numpy(cz["coords"].astype(np.float32))
            ox_, oy_, PW_, PH_, CW_, CH_ = [int(v) for v in cz["photo_box"]]
            v0 = torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[
                None, :, None
            ]  # 1,3,1,H,W: the conditioned frame alone
            with torch.no_grad():
                z0 = (
                    pipe.vae.encode(v0.to("cuda", torch.bfloat16)).latent_dist.mode()
                    * pipe.vae.config.scaling_factor
                )[
                    :, :, 0
                ].float()  # 1,C,H',W'
            Tl, Hl, Wl = zg.shape[1], zg.shape[3], zg.shape[4]
            T_g = coords.shape[0]
            tgt = []
            msk = []
            for tau in range(Tl):
                t_g = min(round(tau * (T_g - 1) / max(Tl - 1, 1)), T_g - 1)
                cm = Fn.interpolate(coords[t_g][None], (Hl, Wl), mode="area")[0].to("cuda")
                gx = (cm[0] * (CW_ - 1) - ox_) / max(PW_ - 1, 1) * 2 - 1
                gy = (cm[1] * (CH_ - 1) - oy_) / max(PH_ - 1, 1) * 2 - 1
                valid = ((gx.abs() <= 1) & (gy.abs() <= 1) & (cm[2] > 0.5)).float()
                tgt.append(
                    Fn.grid_sample(
                        z0,
                        torch.stack([gx, gy], -1)[None],
                        mode="bilinear",
                        padding_mode="border",
                        align_corners=True,
                    )[0]
                )
                msk.append(valid)
            z_self = torch.stack(tgt, 0)[None]
            m_known = torch.stack(msk, 0)[None, :, None]  # (1, T', C, H', W'), (1, T', 1, H', W')
            zg = (m_known * z_self + (1 - m_known) * zg.float()).to(zg.dtype)
            log(
                f"layout source: hybrid (first-frame latent transported along the guide warp); known fraction per latent frame {[round(float(m.mean()), 2) for m in msk[::max(1, Tl // 6)]]}"
            )
        k = max(1, int(round(3 * a.layout_sigma)))
        xs = torch.arange(-k, k + 1, device="cuda", dtype=torch.float32)
        g1 = torch.exp(-(xs**2) / (2 * a.layout_sigma**2))
        g1 = g1 / g1.sum()
        import torch.nn.functional as F

        def lowpass(x):  # (B, T, C, H, W), separable Gaussian over H, W per frame and channel
            B, T, C, Hh, Ww = x.shape
            y = x.float().reshape(B * T * C, 1, Hh, Ww)
            y = F.conv2d(F.pad(y, (k, k, 0, 0), mode="reflect"), g1.view(1, 1, 1, -1))
            y = F.conv2d(F.pad(y, (0, 0, k, k), mode="reflect"), g1.view(1, 1, -1, 1))
            return y.reshape(B, T, C, Hh, Ww).to(x.dtype)

        gen = torch.Generator("cuda").manual_seed(a.seed + 1)

        state = {"zg": zg, "logged": False}

        def cb(pipe_, i, t, kw):
            lat = kw["latents"]
            zg_ = state["zg"]
            if (
                lat.shape[1] != zg_.shape[1] and lat.shape[2:] == zg_.shape[2:]
            ):  # CogVideoX 1.5 pads the latent frames to a multiple of its temporal patch (21 -> 22)
                T = lat.shape[1]
                zg_ = (
                    torch.cat([zg_, zg_[:, -1:].expand(-1, T - zg_.shape[1], -1, -1, -1)], 1)
                    if T > zg_.shape[1]
                    else zg_[:, :T]
                )
                state["zg"] = zg_
            if not state["logged"]:
                log(
                    f"callback: latents {tuple(lat.shape)} guide {tuple(zg_.shape)} -> {'steering' if lat.shape == zg_.shape else 'SHAPE MISMATCH, no steering'}"
                )
                state["logged"] = True
            if i + 1 < a.layout_steps and lat.shape == zg_.shape:
                ts = pipe_.scheduler.timesteps
                t_next = ts[i + 1]
                eps = torch.randn(zg_.shape, generator=gen, device="cuda", dtype=zg_.dtype)
                zt = pipe_.scheduler.add_noise(zg_, eps, t_next[None])
                w = a.layout_weight * (1 - (i + 1) / a.layout_steps)
                lp_g = lowpass(zt.to(lat.dtype)).float()
                lp_l = lowpass(lat).float()
                if a.layout_tone == "keep":
                    mg, sg = lp_g.mean(dim=(-2, -1), keepdim=True), lp_g.std(dim=(-2, -1), keepdim=True)
                    ml, sl = lp_l.mean(dim=(-2, -1), keepdim=True), lp_l.std(dim=(-2, -1), keepdim=True)
                    corr = ((lp_g - mg) / (sg + 1e-6) * sl + ml - lp_l).to(lat.dtype)
                elif a.layout_tone == "mean":
                    c = lp_g - lp_l
                    corr = (c - c.mean(dim=(-2, -1), keepdim=True)).to(lat.dtype)
                else:
                    corr = (lp_g - lp_l).to(lat.dtype)
                if a.layout_ramp:
                    T_ = corr.shape[1]
                    ramp = (
                        torch.linspace(0.0, 1.0, T_, device=corr.device, dtype=corr.dtype)
                        / max(a.layout_ramp_end, 1e-3)
                    ).clamp(max=1.0)
                    corr = corr * ramp.view(1, T_, 1, 1, 1)
                lat = lat + w * corr
            return {"latents": lat}

    t0 = time.time()
    out = pipe(
        image=image,
        prompt=a.prompt,
        height=H,
        width=W,
        num_frames=a.frames,
        num_inference_steps=a.steps,
        guidance_scale=6.0,
        use_dynamic_cfg=True,
        generator=torch.Generator("cuda").manual_seed(a.seed),
        output_type="pil",
        **({"callback_on_step_end": cb, "callback_on_step_end_tensor_inputs": ["latents"]} if cb else {}),
    ).frames[0]
    log(f"{len(out)} frames at {out[0].size} in {time.time() - t0:.0f} s")
    (a.out / "frames_raw").mkdir(exist_ok=True)
    for i, f in enumerate(out):
        f.save(a.out / "frames_raw" / f"vace-{i:03d}.png")
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
            str(a.out / "frames_raw" / "vace-%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            str(a.out / "video_raw.mp4"),
        ],
        check=True,
    )
    (a.out / "metrics.json").write_text(
        json.dumps(
            dict(
                model=a.model,
                guide=str(a.guide),
                layout_steps=a.layout_steps if a.guide else 0,
                layout_sigma=a.layout_sigma,
                layout_weight=a.layout_weight,
                layout_tone=a.layout_tone,
                layout_ramp=a.layout_ramp,
                layout_source=a.layout_source,
                frames=len(out),
                size=list(out[0].size),
                seed=a.seed,
                prompt=a.prompt,
            ),
            indent=1,
        )
    )
    log("done")


if __name__ == "__main__":
    main()
