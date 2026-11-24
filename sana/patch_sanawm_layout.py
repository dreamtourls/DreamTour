"""Add layout steering to SANA-WM (NVlabs/Sana): a per-step hook in LTXFlowEuler.sample and CLI options in inference_sana_wm.py.
Patches a checkout of the Sana repository in place (idempotent):
  python sana/patch_sanawm_layout.py path/to/Sana
New options of inference_sana_wm.py: --layout_guide <mp4 or dir of frames>  --layout_steps 18  --layout_sigma 1.5  --layout_weight 1.0
The guide (same frame count as the generation, resized to 1280x704) is encoded by the pipeline's VAE; during the first K steps the low-pass of the
running latent is replaced by the low-pass of the guide latent noised to the next step's level, with a weight decaying linearly to zero; the
guide's low-pass is first standardised to the prior's per-frame per-channel mean and std."""

import sys
from pathlib import Path

root = Path(sys.argv[1])
# 1. solver hook
f = root / "diffusion/scheduler/flow_euler_sampler.py"
s = f.read_text()
old = """            tokens_to_denoise_mask = t / 1000 - 1e-6 < (1.0 - condition_mask)
            latents = torch.where(tokens_to_denoise_mask, denoised_latents, latents)
"""
new = """            tokens_to_denoise_mask = t / 1000 - 1e-6 < (1.0 - condition_mask)
            latents = torch.where(tokens_to_denoise_mask, denoised_latents, latents)
            hook = getattr(self, "layout_hook", None)   # DreamTour layout steering
            if hook is not None:
                nxt = float(timesteps[i + 1]) / 1000.0 if i + 1 < len(timesteps) else 0.0
                latents = torch.where(tokens_to_denoise_mask, hook(i, nxt, latents), latents)
"""
if "layout_hook" not in s:
    assert s.count(old) == 1, s.count(old)
    f.write_text(s.replace(old, new))
    print("solver hook added")
else:
    print("solver already patched")
# 2. pipeline: options, guide encoding, hook construction
g = root / "inference_video_scripts/wm/inference_sana_wm.py"
p = g.read_text()
if "layout_guide" not in p:
    # a) dispatch: attach the hook
    old = """        if algo == "flow_euler_ltx":
            return LTXFlowEuler(self.model, **base).sample(z, steps=steps, generator=generator)"""
    new = """        if algo == "flow_euler_ltx":
            solver = LTXFlowEuler(self.model, **base)
            solver.layout_hook = getattr(self, "layout_hook", None)
            return solver.sample(z, steps=steps, generator=generator)"""
    assert p.count(old) == 1
    p = p.replace(old, new)
    # b) build the hook right after the first-frame latent is encoded (VAE on the GPU at that point)
    old = """        first_latent = vae_encode(
            self.config.vae.vae_type,
            self.vae,
            img.to(self.device, dtype=self.vae_dtype),
            device=self.device,
        ).to(self.weight_dtype)
"""
    new = (
        old
        + """        self.layout_hook = None
        lg = getattr(self, "layout_guide", None)
        if lg:
            self.layout_hook = self._build_layout_hook(lg, params, first_latent)
"""
    )
    assert p.count(old) >= 1, p.count(old)
    p = p.replace(old, new, 1)  # first occurrence = generate(); generate_streaming() keeps its own
    # c) the builder method (inserted before _resolve_flow_shift)
    old = "    def _resolve_flow_shift(self, override: float | None) -> float:"
    new = '''    def _build_layout_hook(self, guide_path, params, first_latent):
        """DreamTour layout steering for SANA-WM: encode the guide video with the pipeline VAE and return a per-step hook."""
        import glob as _glob
        import numpy as _np
        import torch.nn.functional as _F
        from PIL import Image as _Image
        K = int(getattr(self, "layout_steps", 18)); sig = float(getattr(self, "layout_sigma", 1.5)); w0 = float(getattr(self, "layout_weight", 1.0))
        gp = str(guide_path); frames = []
        if gp.endswith(".mp4"):
            import cv2 as _cv2
            cap = _cv2.VideoCapture(gp)
            while True:
                ok, im = cap.read()
                if not ok: break
                frames.append(_Image.fromarray(_cv2.cvtColor(im, _cv2.COLOR_BGR2RGB)))
        else:
            frames = [_Image.open(q).convert("RGB") for q in sorted(_glob.glob(gp + "/*.png"))]
        n = int(params.num_frames)
        if len(frames) < n:
            raise SystemExit(f"layout guide has {len(frames)} frames, generation needs {n}")
        idx = [round(k * (len(frames) - 1) / max(n - 1, 1)) for k in range(n)]
        frames = [frames[k].resize((TARGET_WIDTH, TARGET_HEIGHT), _Image.LANCZOS) for k in idx]
        vid = torch.stack([T.ToTensor()(fr) * 2.0 - 1.0 for fr in frames], dim=1).unsqueeze(0)   # 1,3,F,H,W
        with torch.no_grad():
            try:
                zg = vae_encode(self.config.vae.vae_type, self.vae, vid.to(self.device, dtype=self.vae_dtype), device=self.device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); chunks = []; step = 8 * 8 + 1
                for s0 in range(0, n, step - 1):
                    part = vid[:, :, s0:s0 + step]
                    if part.shape[2] < 2: break
                    zc = vae_encode(self.config.vae.vae_type, self.vae, part.to(self.device, dtype=self.vae_dtype), device=self.device)
                    chunks.append(zc if not chunks else zc[:, :, 1:])
                zg = torch.cat(chunks, 2)
        zg = zg.to(self.weight_dtype)
        if zg.shape[2] != first_latent.shape[2] and zg.shape[2] > 1:
            pass
        self.logger.info("[layout steering] guide latent %s, K=%d, sigma=%.2f cells, w0=%.2f", tuple(zg.shape), K, sig, w0)
        r = max(1, int(3 * sig)); xs = torch.arange(-r, r + 1, device=self.device, dtype=torch.float32); ker = torch.exp(-xs ** 2 / (2 * sig ** 2)); ker = ker / ker.sum()
        def lowpass(x):
            B, C, Fh, H, W = x.shape; y = x.float().reshape(B * C * Fh, 1, H, W)
            y = _F.conv2d(_F.pad(y, (r, r, 0, 0), mode="replicate"), ker.view(1, 1, 1, -1)); y = _F.conv2d(_F.pad(y, (0, 0, r, r), mode="replicate"), ker.view(1, 1, -1, 1))
            return y.reshape(B, C, Fh, H, W).to(x.dtype)
        gen = torch.Generator(device=self.device).manual_seed(int(params.seed) + 7)
        def hook(i, sigma_next, latents):
            if i + 1 >= K or latents.shape != zg.shape:
                return latents
            eps = torch.randn(zg.shape, generator=gen, device=self.device, dtype=zg.dtype)
            zt = (1.0 - sigma_next) * zg + sigma_next * eps
            w = w0 * (1.0 - (i + 1) / K)
            lp_g = lowpass(zt).float(); lp_l = lowpass(latents).float()   # keep the prior's low-pass mean and contrast per frame and channel
            mg, sg = lp_g.mean(dim=(-2, -1), keepdim=True), lp_g.std(dim=(-2, -1), keepdim=True); ml, sl = lp_l.mean(dim=(-2, -1), keepdim=True), lp_l.std(dim=(-2, -1), keepdim=True)
            corr = ((lp_g - mg) / (sg + 1e-6) * sl + ml - lp_l).to(latents.dtype)
            return latents + w * corr
        return hook

    def _resolve_flow_shift(self, override: float | None) -> float:'''
    assert p.count(old) == 1
    p = p.replace(old, new, 1)
    # d) CLI args + wiring onto the pipeline object (find the parser and the pipeline construction)
    import re

    m = re.search(r'    p\.add_argument\("--seed"', p)
    assert m, "parser --seed not found"
    ins = """    p.add_argument("--layout_guide", type=str, default=None, help="DreamTour layout steering: mp4 or directory of frames showing what the camera should see")
    p.add_argument("--layout_steps", type=int, default=18, help="layout steering: number of early steps (of --step) that receive the low-pass guide")
    p.add_argument("--layout_sigma", type=float, default=1.5, help="layout steering: Gaussian low-pass sigma in latent cells (32 px each)")
    p.add_argument("--layout_weight", type=float, default=1.0, help="layout steering: weight at step 0, decaying linearly to 0 at --layout_steps")
"""
    p = p[: m.start()] + ins + p[m.start() :]
    old_w = """        offload_refiner=args.offload_refiner,
        logger=logger,
    )
"""
    new_w = (
        old_w
        + """    pipeline.layout_guide = args.layout_guide; pipeline.layout_steps = args.layout_steps; pipeline.layout_sigma = args.layout_sigma; pipeline.layout_weight = args.layout_weight
"""
    )
    assert p.count(old_w) == 1, p.count(old_w)
    p = p.replace(old_w, new_w, 1)
    g.write_text(p)
    print("pipeline patched (options, encoder, hook, wiring)")
else:
    print("pipeline already patched")
