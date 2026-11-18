"""
Custom colormaps
----------------
'ibm_seq': sequential map through the hues of the IBM colour-blind-safe palette
(blue -> violet -> magenta -> orange -> amber), the same palette used for the
saSPT state figures.

The raw IBM colours are not ordered by lightness (L* = 61, 50, 50, 61, 77), so
the map is built in CIELAB: hue/chroma (a*, b*) follow the IBM anchors while
lightness L* rises linearly from L_START to L_END. Equal steps in the data are
therefore equal steps in perceived brightness, which keeps the scale readable
for colour-blind viewers and in greyscale.

'abbelight_seq': same construction with the Abbelight brand colours (from the
Abbelight Technical Doc template theme): dark teal #00647B -> cyan #00C9F7 ->
magenta #FF73FF -> orange #FF9B35, L* rising 45 -> 82. The cyan and magenta
anchors sit at 0.20 and 0.55 of the scale so that, with the skewed track-length
distribution, mid-length tracks pick up magenta rather than staying cyan.

'abbelight_gradient': the four Abbelight accents green #00D295 -> cyan #00C9F7
-> magenta #FF73FF -> orange #FF9B35, interpolated in CIELAB at their own
lightness (vivid, but not lightness-ordered; used for the MSD cover images).

Importing this module registers all three maps (and their '_r' reversed
versions) with matplotlib.
"""

import numpy as np
import matplotlib
import matplotlib.colors as mcolors

IBM_ANCHORS = ('#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000')
L_START, L_END = 40.0, 80.0

# Abbelight brand palette (Technical Doc template theme)
ABBELIGHT_COLORS = {
    'teal': '#00647B', 'blue': '#0096B9', 'cyan': '#00C9F7', 'green': '#00D295',
    'magenta': '#FF73FF', 'orange': '#FF9B35', 'light_grey': '#DEE3E6',
}
ABBELIGHT_ANCHORS = ('#00647B', '#00C9F7', '#FF73FF', '#FF9B35')
ABBELIGHT_POSITIONS = (0.0, 0.20, 0.55, 1.0)
ABBELIGHT_L_START, ABBELIGHT_L_END = 45.0, 82.0

_WHITE = np.array([0.95047, 1.0, 1.08883])
_M_RGB2XYZ = np.array([[0.4124, 0.3576, 0.1805],
                       [0.2126, 0.7152, 0.0722],
                       [0.0193, 0.1192, 0.9505]])


def _srgb_to_lab(rgb):
    c = np.asarray(rgb, dtype=float)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M_RGB2XYZ.T / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _lab_to_srgb(lab):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    f = np.stack([fy + a / 500, fy, fy - b / 200], -1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * _WHITE
    lin = xyz @ np.linalg.inv(_M_RGB2XYZ).T
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)


def lightness_ramp_cmap(name, anchors, l_start, l_end, n=256, positions=None):
    """Colormap whose a*/b* follow `anchors` (placed at `positions` in [0, 1],
    evenly spaced by default) and whose L* rises linearly."""
    lab = _srgb_to_lab([mcolors.to_rgb(c) for c in anchors])
    pos = np.linspace(0, 1, len(anchors)) if positions is None else np.asarray(positions, dtype=float)
    t = np.linspace(0, 1, n)
    ramp = np.stack([np.linspace(l_start, l_end, n),
                     np.interp(t, pos, lab[:, 1]),
                     np.interp(t, pos, lab[:, 2])], -1)
    return mcolors.ListedColormap(np.clip(_lab_to_srgb(ramp), 0, 1), name=name)


IBM_SEQ = lightness_ramp_cmap('ibm_seq', IBM_ANCHORS, L_START, L_END)
ABBELIGHT_SEQ = lightness_ramp_cmap('abbelight_seq', ABBELIGHT_ANCHORS, ABBELIGHT_L_START, ABBELIGHT_L_END,
                                    positions=ABBELIGHT_POSITIONS)


def lab_gradient_cmap(name, anchors, n=256):
    """Colormap interpolated in CIELAB that keeps each anchor's own lightness
    (decorative gradients; not a perceptually ordered data scale)."""
    lab = _srgb_to_lab([mcolors.to_rgb(c) for c in anchors])
    pos, t = np.linspace(0, 1, len(anchors)), np.linspace(0, 1, n)
    ramp = np.stack([np.interp(t, pos, lab[:, i]) for i in range(3)], -1)
    return mcolors.ListedColormap(np.clip(_lab_to_srgb(ramp), 0, 1), name=name)


ABBELIGHT_GRADIENT = lab_gradient_cmap('abbelight_gradient', ('#00D295', '#00C9F7', '#FF73FF', '#FF9B35'))

for _cmap in (IBM_SEQ, IBM_SEQ.reversed(), ABBELIGHT_SEQ, ABBELIGHT_SEQ.reversed(),
              ABBELIGHT_GRADIENT, ABBELIGHT_GRADIENT.reversed()):
    if _cmap.name not in matplotlib.colormaps:
        matplotlib.colormaps.register(_cmap)
