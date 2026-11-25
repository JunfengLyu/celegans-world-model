"""Bend the Virtual Worm sagittal cuticle profile around a planar centerline."""

import json
from pathlib import Path

from matplotlib.colors import to_rgb
from matplotlib.patches import Polygon
import numpy as np

ATLAS = Path(__file__).resolve().parents[1] / "data" / "virtual-worm-atlas.json"


def load_cuticle_profile(path=ATLAS):
    """Return normalized arclength and sagittal radii from the cuticle mesh."""
    sections = np.asarray(json.loads(Path(path).read_text())["crossSections"], dtype=float)
    return sections[:, :2]


def _body_coordinates(centerline, profile, width_scale):
    centerline = np.asarray(centerline, dtype=float)
    if centerline.ndim != 2 or centerline.shape[1] != 2 or len(centerline) < 2:
        raise ValueError("A centerline must contain at least two planar points.")
    distances = np.linalg.norm(np.diff(centerline, axis=0), axis=1)
    length = distances.sum()
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("A centerline must have positive, finite length.")
    arclength = np.r_[0.0, distances.cumsum()] / length
    keep = np.r_[True, distances > 0.0]
    u, widths = profile.T
    center = np.column_stack([np.interp(u, arclength[keep], coordinate[keep]) for coordinate in centerline.T])
    tangent = np.gradient(center, u, axis=0)
    norm = np.linalg.norm(tangent, axis=1)
    if np.any(norm <= 1e-12):
        good = norm > 1e-12
        tangent = np.column_stack([np.interp(u, u[good], component[good]) for component in tangent.T])
        norm = np.linalg.norm(tangent, axis=1)
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0])) / np.maximum(norm[:, None], 1e-12)
    offset = normal * (length * width_scale * widths[:, None])
    return center, offset


def _band(center, offset, lo, hi):
    return np.concatenate((center + hi * offset, (center + lo * offset)[::-1]))


def silhouette(centerline, profile=None, *, width_scale=1.0):
    """Return a closed planar outline as head-to-tail positive and reverse negative flanks."""
    profile = load_cuticle_profile() if profile is None else profile
    return _band(*_body_coordinates(centerline, profile, width_scale), -1.0, 1.0)


class WormSilhouette:
    """Mutable, vector-safe Matplotlib artists for one arclength-registered cuticle."""

    def __init__(self, ax, centerline, profile, color, width_scale, linewidth, alpha, zorder):
        self.profile = profile
        self.width_scale = width_scale
        ink = np.asarray(to_rgb(color))
        light = 0.38 * ink + 0.62
        middle = 0.62 * ink + 0.38
        dark = 0.88 * ink + 0.12
        center, offset = _body_coordinates(centerline, profile, width_scale)
        body = _band(center, offset, -1.0, 1.0)
        self.base = Polygon(body, facecolor=middle, edgecolor="none", alpha=alpha, zorder=zorder)
        self.shade = Polygon(_band(center, offset, -1.0, -0.56), facecolor=dark,
                             edgecolor="none", alpha=alpha, zorder=zorder + 0.01)
        self.light = Polygon(_band(center, offset, -0.1, 0.55), facecolor=light,
                             edgecolor="none", alpha=alpha, zorder=zorder + 0.02)
        self.edge = Polygon(body, facecolor="none", edgecolor=ink, linewidth=linewidth,
                            alpha=alpha, joinstyle="round", zorder=zorder + 0.03)
        self.artists = (self.base, self.shade, self.light, self.edge)
        for artist in self.artists:
            ax.add_patch(artist)
        self.shade.set_clip_path(self.base)
        self.light.set_clip_path(self.base)

    def set_centerline(self, centerline):
        """Update the outline and shading in place for an animation frame."""
        center, offset = _body_coordinates(centerline, self.profile, self.width_scale)
        body = _band(center, offset, -1.0, 1.0)
        self.base.set_xy(body)
        self.edge.set_xy(body)
        self.shade.set_xy(_band(center, offset, -1.0, -0.56))
        self.light.set_xy(_band(center, offset, -0.1, 0.55))


def draw_worm(ax, centerline, profile, color, *, width_scale=1.0, linewidth=0.55, alpha=1.0, zorder=2):
    """Draw a shaded sagittal cuticle using the projected centerline as its midline."""
    profile = load_cuticle_profile() if profile is None else profile
    return WormSilhouette(ax, centerline, profile, color, width_scale, linewidth, alpha, zorder)
