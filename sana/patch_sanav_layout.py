"""Layout steering for SANA-Video 2.0 (NVlabs/Sana, inference_video_scripts/inference_sana_video.py): same hook as patch_sanawm_layout.py.
Patches a checkout of the Sana repository in place (idempotent):
  python sana/patch_sanav_layout.py path/to/Sana
Adds the config fields layout_guide (mp4 or frame dir), layout_steps (40), layout_sigma (1.5), layout_weight (1.0), layout_tone (raw),
layout_ramp (false) and layout_ramp_end (0.5); encodes the guide with the pipeline VAE after the first-frame latent; during the first K steps the
low-pass of the running latent is replaced by that of the guide latent noised to the next step's level, weight decaying linearly to zero.
Tone: raw = plain difference (SANA's DC-AE latent stores the layout in its channel means, so mean removal cancels the steering);
mean = per-frame per-channel mean removed; keep = mean and std standardised to the prior's. Ramp: the correction is scaled 0 -> 1 over the latent
frames (full strength from --layout_ramp_end of the clip) so the content does not jump after the conditioned first frame.
"""

import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
# 1. solver hooks (LTXFlowEuler: inside its masked update; FlowEuler: after the scheduler step)
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
    assert s.count(old) == 1, ("LTX block", s.count(old))
    s = s.replace(old, new)
    old2 = """            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
"""
    new2 = (
        old2
        + """            hook = getattr(self, "layout_hook", None)   # DreamTour layout steering
            if hook is not None:
                nxt = float(timesteps[i + 1]) / 1000.0 if i + 1 < len(timesteps) else 0.0
                latents = hook(i, nxt, latents)
"""
    )
    n2 = s.count(old2)
    print("plain FlowEuler step blocks:", n2)
    if n2:
        s = s.replace(old2, new2, 1)
    f.write_text(s)
    print("solver hooks added")
else:
    print("solver already patched")
# 2. inference script
g = root / "inference_video_scripts/inference_sana_video.py"
p = g.read_text()
if "layout_guide" not in p:
    # a) dataclass fields
    old = "    motion_score: int = 10\n"
    assert p.count(old) == 1
    p = p.replace(
        old,
        old
        + "    layout_guide: Optional[str] = None   # DreamTour layout steering: mp4 or directory of frames the camera should see\n"
        "    layout_steps: int = 40\n    layout_sigma: float = 1.5   # latent cells (32 px each)\n    layout_weight: float = 1.0\n"
        "    layout_tone: str = 'raw'   # raw | mean | keep\n    layout_ramp: bool = False\n    layout_ramp_end: float = 0.5\n",
    )
    # b) hook builder (module level, before def visualize)
    old = "def visualize(config, args, model, items, bs, sample_steps, cfg_scale):"
    builder = '''def build_layout_hook(config, args, vae, vae_dtype, device, out_wh, num_frames, seed):
    """DreamTour layout steering: encode the guide video with the pipeline VAE, return a per-step hook(i, sigma_next, latents)."""
    import glob as _glob
    import torch.nn.functional as _F
    from PIL import Image as _Image
    import torchvision.transforms as _T
    K, sig, w0 = int(args.layout_steps), float(args.layout_sigma), float(args.layout_weight)
    tone = str(getattr(args, "layout_tone", "raw")); ramp_on = bool(getattr(args, "layout_ramp", False)); ramp_end = float(getattr(args, "layout_ramp_end", 0.5))
    gp = str(args.layout_guide); frames = []
    if gp.endswith(".mp4"):
        import cv2 as _cv2
        cap = _cv2.VideoCapture(gp)
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(_Image.fromarray(_cv2.cvtColor(im, _cv2.COLOR_BGR2RGB)))
    else:
        frames = [_Image.open(q).convert("RGB") for q in sorted(_glob.glob(gp + "/*.png"))]
    n = int(num_frames); idx = [round(k * (len(frames) - 1) / max(n - 1, 1)) for k in range(n)]
    frames = [frames[k].resize(out_wh, _Image.LANCZOS) for k in idx]
    vid = torch.stack([_T.ToTensor()(fr) * 2.0 - 1.0 for fr in frames], dim=1).unsqueeze(0)   # 1,3,F,H,W
    with torch.no_grad():
        try:
            zg = vae_encode(config.vae.vae_type, vae, vid.to(device, dtype=vae_dtype), device=device)
        except torch.cuda.OutOfMemoryError:   # encode in two halves that overlap by one frame (approximate for a causal video VAE)
            torch.cuda.empty_cache(); h = vid.shape[2] // 2
            za = vae_encode(config.vae.vae_type, vae, vid[:, :, :h + 1].to(device, dtype=vae_dtype), device=device)
            zb = vae_encode(config.vae.vae_type, vae, vid[:, :, h:].to(device, dtype=vae_dtype), device=device)
            zg = torch.cat([za, zb[:, :, 1:]], dim=2)
    print(f"[layout steering] guide latent {tuple(zg.shape)}, K={K}, sigma={sig} cells, w0={w0}, tone={tone}, ramp={ramp_on} (end {ramp_end})", flush=True)
    r = max(1, int(3 * sig)); xs = torch.arange(-r, r + 1, device=device, dtype=torch.float32); ker = torch.exp(-xs ** 2 / (2 * sig ** 2)); ker = ker / ker.sum()
    def lowpass(x):
        B, C, Fh, H, W = x.shape; y = x.float().reshape(B * C * Fh, 1, H, W)
        y = _F.conv2d(_F.pad(y, (r, r, 0, 0), mode="replicate"), ker.view(1, 1, 1, -1)); y = _F.conv2d(_F.pad(y, (0, 0, r, r), mode="replicate"), ker.view(1, 1, -1, 1))
        return y.reshape(B, C, Fh, H, W).to(x.dtype)
    gen = torch.Generator(device=device).manual_seed(int(seed) + 7)
    def hook(i, sigma_next, latents):
        if i + 1 >= K or latents.shape[2:] != zg.shape[2:]:
            return latents
        zgl = zg.to(latents.dtype)
        eps = torch.randn(zgl.shape, generator=gen, device=device, dtype=zgl.dtype)
        zt = (1.0 - sigma_next) * zgl + sigma_next * eps
        w = w0 * (1.0 - (i + 1) / K)
        lp_g = lowpass(zt).float(); lp_l = lowpass(latents[:1]).float()
        if tone == "keep":   # the prior's low-pass mean and contrast per frame and channel
            mg, sg = lp_g.mean(dim=(-2, -1), keepdim=True), lp_g.std(dim=(-2, -1), keepdim=True); ml, sl = lp_l.mean(dim=(-2, -1), keepdim=True), lp_l.std(dim=(-2, -1), keepdim=True)
            corr = (lp_g - mg) / (sg + 1e-6) * sl + ml - lp_l
        elif tone == "mean":
            corr = lp_g - lp_l; corr = corr - corr.mean(dim=(-2, -1), keepdim=True)
        else:   # raw: SANA's layout lives in the channel means
            corr = lp_g - lp_l
        if ramp_on:
            Fh = corr.shape[2]; ramp = (torch.linspace(0.0, 1.0, Fh, device=corr.device) / max(ramp_end, 1e-3)).clamp(max=1.0)
            corr = corr * ramp.view(1, 1, Fh, 1, 1)
        return latents + w * corr.to(latents.dtype)
    return hook


'''
    assert p.count(old) == 1
    p = p.replace(old, builder + old)
    # c) build after the first-frame latent is encoded, attach to the solvers
    old = """                for frame_idx in list(condition_frame_info.keys()):
                    z[:, :, frame_idx : frame_idx + 1] = image_vae_embeds  # 1,C,F,H,W, first frame is the image
"""
    assert p.count(old) == 1
    p = p.replace(
        old,
        old
        + """                layout_hook = None
                if getattr(args, "layout_guide", None):
                    layout_hook = build_layout_hook(config, args, vae, vae_dtype, device, (int(latent_size_w * config.vae.vae_downsample_rate), int(latent_size_h * config.vae.vae_downsample_rate)), args.num_frames, cur_seed)
""",
    )
    n = 0
    for solver in ("LTXFlowEuler(", "FlowEuler("):
        pat = re.compile(
            r"(                flow_solver = " + re.escape(solver) + r".*?\n                \)\n)", re.S
        )
        m = pat.search(p)
        if m and "layout_hook" not in m.group(1):
            p = (
                p[: m.end()]
                + "                flow_solver.layout_hook = locals().get('layout_hook')\n"
                + p[m.end() :]
            )
            n += 1
    print("solvers wired:", n)
    g.write_text(p)
    print("inference script patched")
else:
    print("inference script already patched")
