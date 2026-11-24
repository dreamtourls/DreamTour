"""Canvas by outpainting: the photograph sits at (512, 288) inside a canvas 1024 px wider and 576 px taller and an inpainting model paints
the margins. The recipe is FLUX.1 Fill dev over the whole canvas in one pass (--model black-forest-labs/FLUX.1-Fill-dev --whole); an SDXL
inpainting checkpoint painting 1024-px tiles is the fallback for users without access to FLUX.1 Fill. The margins are used as painted: no
tone match is needed. Output: <out>/canvas.png, photo.png (the resized photograph) and canvas.json (canvas size, photo offset and size, fov,
prompt), the contract read by guide_video.py and video_between.py.

    python scripts/make_canvas_outpaint.py --image photo.png --prompt "<caption>" --model black-forest-labs/FLUX.1-Fill-dev --whole --out canvases/photo
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

NEG = "cartoon, illustration, painting, render, 3d, anime, toy, plastic, blurry, low quality, text, watermark, frame, border"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--model",
        default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        help="Hugging Face repo or local single-file SDXL inpainting checkpoint; black-forest-labs/FLUX.1-Fill-dev for the recipe",
    )
    p.add_argument(
        "--margins",
        type=int,
        nargs=2,
        default=(288, 512),
        help="vertical and horizontal margin around the photograph (px)",
    )
    p.add_argument(
        "--max-side", type=int, default=1024, help="the photograph is downscaled to this longest side"
    )
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=6.0, help="defaults: 6 for SDXL, 30 for FLUX.1 Fill")
    p.add_argument("--strength", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--fov",
        type=float,
        default=50.0,
        help="horizontal field of view of the photograph (deg), written to canvas.json",
    )
    p.add_argument(
        "--whole",
        action="store_true",
        help="outpaint the whole canvas in one pass instead of 1024-px tiles (FLUX.1 Fill handles 2048x1152)",
    )
    p.add_argument(
        "--suffix",
        default=", photograph, DSLR, natural light, sharp detail, true-to-life colours",
        help="appended to the prompt",
    )
    a = p.parse_args()
    if (a.out / "canvas.json").exists():
        print("exists", a.out)
        return
    t0 = time.time()
    fill = "FLUX.1-Fill" in a.model
    if (
        fill
    ):  # FLUX.1 Fill dev (gated on Hugging Face: accept the licence and log in first): mask-conditioned outpainting
        from diffusers import FluxFillPipeline

        pipe = FluxFillPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16).to("cuda")
        if a.guidance == 6.0:
            a.guidance = 30.0
        if a.steps == 30:
            a.steps = 50
    else:
        from diffusers import StableDiffusionXLInpaintPipeline

        if a.model.endswith(".safetensors"):
            pipe = StableDiffusionXLInpaintPipeline.from_single_file(a.model, torch_dtype=torch.float16)
        else:
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                a.model, torch_dtype=torch.float16, variant="fp16" if "diffusers/" in a.model else None
            )
        pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print(f"inpainter {a.model} loaded in {time.time() - t0:.0f} s", flush=True)
    img = Image.open(a.image).convert("RGB")
    s = min(1.0, a.max_side / max(img.size))
    if s < 1:
        img = img.resize((max(16, round(img.width * s)), max(16, round(img.height * s))), Image.LANCZOS)
    W, H = img.width - img.width % 16, img.height - img.height % 16
    img = img.resize((W, H), Image.LANCZOS) if (W, H) != img.size else img
    mv, mh = a.margins
    CW, CH = W + 2 * mh, H + 2 * mv
    canvas = Image.new("RGB", (CW, CH), (128, 128, 128))
    canvas.paste(img, (mh, mv))
    known = np.zeros((CH, CW), bool)
    known[mv : mv + H, mh : mh + W] = True
    text = a.prompt.strip() + a.suffix
    gen = torch.Generator("cuda").manual_seed(a.seed)

    def outpaint(box):
        """Inpaint the unknown pixels inside box = (x0, y0, x1, y1), paste the result with a feathered mask and mark the pixels known."""
        x0, y0, x1, y1 = box
        tile = canvas.crop(box)
        m = ~known[y0:y1, x0:x1]
        if not m.any():
            return
        mask = Image.fromarray((m * 255).astype(np.uint8))
        tw, th = tile.size
        rw, rh = tw - tw % 8, th - th % 8
        if fill:
            res = (
                pipe(
                    prompt=text,
                    image=tile.resize((rw, rh)),
                    mask_image=mask.resize((rw, rh)),
                    num_inference_steps=a.steps,
                    guidance_scale=a.guidance,
                    generator=gen,
                    height=rh,
                    width=rw,
                    max_sequence_length=512,
                )
                .images[0]
                .resize((tw, th), Image.LANCZOS)
            )
        else:
            res = (
                pipe(
                    prompt=text,
                    negative_prompt=NEG,
                    image=tile.resize((rw, rh)),
                    mask_image=mask.resize((rw, rh)),
                    num_inference_steps=a.steps,
                    guidance_scale=a.guidance,
                    strength=a.strength,
                    generator=gen,
                    height=rh,
                    width=rw,
                )
                .images[0]
                .resize((tw, th), Image.LANCZOS)
            )
        feather = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))
        feather = Image.fromarray(
            np.minimum(np.asarray(feather), (m * 255).astype(np.uint8))
        )  # feather inside the hole only
        canvas.paste(Image.composite(res, tile, feather), (x0, y0))
        known[y0:y1, x0:x1] = True

    if (
        a.whole
    ):  # one pass over the whole canvas: the model sees the entire photograph while painting every margin
        outpaint((0, 0, CW, CH))
    T = 1024  # tiles (anything --whole left unpainted, or the whole margin for the SDXL fallback): first the side bands at the photo's rows, then the top and bottom bands
    outpaint((0, mv, min(T, CW), mv + H))
    outpaint((max(0, CW - T), mv, CW, mv + H))
    for x0 in range(0, CW, T - 128):
        x1 = min(x0 + T, CW)
        x0 = x1 - T
        outpaint((x0, 0, x1, min(T, mv + H // 2)))
        outpaint((x0, max(0, CH - T), x1, CH))
        if x1 == CW:
            break
    a.out.mkdir(parents=True, exist_ok=True)
    canvas.save(a.out / "canvas.png")
    img.save(a.out / "photo.png")
    json.dump(
        dict(
            image=str(a.image),
            canvas=[CW, CH],
            offset=[mh, mv],
            size=[W, H],
            fov=a.fov,
            steps=a.steps,
            texture=None,
            seed=a.seed,
            prompt=text,
            prior=f"outpaint:{a.model}",
            guidance=a.guidance,
        ),
        open(a.out / "canvas.json", "w"),
        indent=1,
    )
    print(
        f"canvas {CW}x{CH}, photo {W}x{H} at ({mh}, {mv}) -> {a.out} ({time.time() - t0:.0f} s)", flush=True
    )


if __name__ == "__main__":
    main()
