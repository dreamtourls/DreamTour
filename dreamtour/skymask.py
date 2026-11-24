"""Sky mask for depth backends without a sky head (Pi3X, MapAnything): SAM 3.1's "sky" / "clouds" masks (dreamtour.segment._sky_of); when the
segmenter cannot be loaded no pixel is sky (and the far rule of the canvas still catches the far limit)."""

from __future__ import annotations

import torch


class SkySegmenter:
    def __init__(self, device="cuda", enabled: bool = True):
        self.dev, self.enabled, self._segmenter = device, enabled, None

    def __call__(self, imgs01: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) in [0, 1] -> (N, H, W) bool."""
        N, _, H, W = imgs01.shape
        if not self.enabled:
            return torch.zeros((N, H, W), dtype=torch.bool, device=self.dev)
        if self._segmenter is None:
            try:
                from .segment import Segmenter

                self._segmenter = Segmenter(self.dev)
            except Exception as e:  # noqa: BLE001
                print(f"[sky] no SAM 3 segmenter ({str(e)[:120]}); no pixel is sky", flush=True)
                self.enabled = False
                return torch.zeros((N, H, W), dtype=torch.bool, device=self.dev)
        from .segment import _sky_of

        return torch.stack([_sky_of(self._segmenter, im).to(self.dev) for im in imgs01])
