"""Pull-push hole filling for layout guides and canvases (fill_holes)."""

import torch
import torch.nn.functional as F


def fill_holes(image, valid, depth=None, far_bias=2.0, coarse=1):
    """Pull-push fill: every unknown pixel takes the mean of the nearest known colours (coarse-to-fine).

    With `depth`, known pixels are weighted by (depth / median depth) ** far_bias, so a disocclusion hole behind a
    silhouette is filled from the far side (the background) instead of bleeding the foreground's colour into it.
    With `coarse` > 1 the fill is computed on a `coarse`-times smaller grid and upsampled: a full-resolution pull-push
    extends each boundary row or column on its own, so a band next to a vertical edge comes out as horizontal streaks,
    which the prior then reproduces as stripes; the coarse fill is a smooth colour field that only states the layout.
    """
    weight = valid[None].float()
    if depth is not None and far_bias and valid.any():
        reference = depth[valid & torch.isfinite(depth) & (depth > 0)]
        if len(reference):
            scale = (
                torch.nan_to_num(depth, nan=0.0, posinf=0.0).clamp_min(0) / reference.median().clamp_min(1e-6)
            ).clamp(0.25, 4.0)
            weight = weight * scale[None] ** far_bias
    if coarse > 1:
        low_rgb = F.avg_pool2d((image * weight)[None], coarse, ceil_mode=True)[0]
        low_weight = F.avg_pool2d(weight[None], coarse, ceil_mode=True)[0]
        low = fill_holes(low_rgb / low_weight.clamp_min(1e-8), low_weight[0] > 0)
        up = F.interpolate(low[None], size=image.shape[-2:], mode="bilinear", align_corners=False)[0]
        return torch.where(valid[None], image, up)
    rgb = image * weight
    levels = []
    # Pool down to a single cell: stopping at one row or column would leave a hole along an edge in its own column
    # lineage (ceil-mode pooling never mixes it with a known cell), so it would take the 0.5 default instead of the
    # known mean, and a canvas built on it would copy a flat gray strip there.
    while (weight == 0).any() and max(weight.shape[-2:]) > 1:
        levels.append((rgb, weight))
        kernel = (min(2, weight.shape[-2]), min(2, weight.shape[-1]))
        rgb = F.avg_pool2d(rgb[None], kernel, ceil_mode=True)[0]
        weight = F.avg_pool2d(weight[None], kernel, ceil_mode=True)[0]
    known_mean = (
        (rgb.sum((-2, -1)) / weight.sum((-2, -1)).clamp_min(1e-8))[:, None, None]
        if (weight > 0).any()
        else torch.full_like(rgb[:, :1, :1], 0.5)
    )
    filled = torch.where(weight > 0, rgb / weight.clamp_min(1e-8), known_mean.expand_as(rgb))
    for rgb, weight in reversed(levels):
        up = F.interpolate(filled[None], size=rgb.shape[-2:], mode="bilinear", align_corners=False)[0]
        filled = torch.where(weight > 0, rgb / weight.clamp_min(1e-8), up)
    return filled
