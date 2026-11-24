"""Turn VideoX-Fun's examples/wan2.2_fun/predict_v2v_control_camera.py (a flat script of constants) into wanfun_run.py that takes
--image --prompt --camera --out --seed --size on the command line (--steps --guidance --frames override the card settings), with the
model-card sampler settings kept (Flow, shift 5, 50 steps, guidance 6, 81 frames, 16 fps), TeaCache off (exact sampling) and
model_cpu_offload (one 80 GB GPU per clip). The camera file is a RealEstate10K pose text file, see wanfun_poses.py.

    python baselines/wanfun_make_runner.py <VideoX-Fun>/examples/wan2.2_fun/predict_v2v_control_camera.py <VideoX-Fun>/examples/wan2.2_fun/wanfun_run.py
"""

import re
import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()
out = Path(sys.argv[2])
head = """import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--image", required=True); _ap.add_argument("--prompt", required=True); _ap.add_argument("--camera", required=True)
_ap.add_argument("--out", required=True); _ap.add_argument("--seed", type=int, default=42); _ap.add_argument("--size", type=int, nargs=2, default=[704, 1280])
_ap.add_argument("--steps", type=int, default=50); _ap.add_argument("--guidance", type=float, default=6.0); _ap.add_argument("--frames", type=int, default=81)
_a = _ap.parse_args()
"""
reps = [
    (r"^GPU_memory_mode\s*=.*$", 'GPU_memory_mode     = "model_cpu_offload"'),
    (r"^enable_teacache\s*=.*$", "enable_teacache     = False"),
    (r"^sample_size\s*=.*$", "sample_size         = list(_a.size)"),
    (r"^video_length\s*=.*$", "video_length        = _a.frames"),
    (r"^control_camera_txt\s*=.*$", "control_camera_txt      = _a.camera"),
    (r"^start_image\s*=.*$", "start_image             = _a.image"),
    (r"^prompt\s*=.*$", "prompt                  = _a.prompt"),
    (r"^guidance_scale\s*=.*$", "guidance_scale          = _a.guidance"),
    (r"^seed\s*=.*$", "seed                    = _a.seed"),
    (r"^num_inference_steps\s*=.*$", "num_inference_steps     = _a.steps"),
    (r"^save_path\s*=.*$", "save_path               = _a.out"),
]
for pat, rep in reps:
    src, n = re.subn(pat, rep, src, flags=re.M)
    assert n == 1, (pat, n)
# save as video_raw.mp4 in --out instead of a numbered file
src = src.replace(
    'video_path = os.path.join(save_path, prefix + ".mp4")',
    'video_path = os.path.join(save_path, "video_raw.mp4")',
)
assert "video_raw.mp4" in src
# insert the argparse block after the sys.path setup (before the videox_fun imports)
i = src.index("from videox_fun.data import process_pose_file")
src = src[:i] + head + src[i:]
out.write_text(src)
print(out)
