"""Shared constants and helpers."""

import numpy as np
import pandas as pd
import matplotlib.patheffects as pe
from PIL import Image

# Acquisition parameters
PIXEL_SIZE_UM = 0.0975    # camera pixel in the sample plane (97.5 nm)
FRAME_INTERVAL_S = 0.04   # 40 ms per frame
LOC_ERROR_UM = 0.035      # localization precision
MIN_LENGTH = 5            # minimum localizations per trajectory

# saSPT populations and their colours (IBM colour-blind-safe palette)
CATEGORIES = ('Immobile', 'Multimers', 'Monomers')
STATE_COLORS = ('#648FFF', '#DC267F', '#FFB000')


def load_tracks(csv_path, min_length=MIN_LENGTH):
    """Trajectory table (x, y in pixels; 1-based frame; trajectory id), keeping
    trajectories with at least `min_length` localizations."""
    df = pd.read_csv(csv_path)
    counts = df['trajectory'].value_counts()
    return df[df['trajectory'].isin(counts[counts >= min_length].index)]


def load_frame(tif_path, page=0):
    """One page of a TIFF movie as float32."""
    img = Image.open(tif_path)
    img.seek(page)
    return np.array(img, dtype=np.float32)


def contrast_stretch(frame, low=1.0, high=99.8):
    """Scale `frame` to [0, 1] between two intensity percentiles."""
    lo, hi = np.percentile(frame, (low, high))
    return np.clip((frame - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(frame)


def add_scale_bar(ax, length_um, pixel_size_um=PIXEL_SIZE_UM, fontsize=14, linewidth=4,
                  margin_frac=0.04, color='white'):
    """Scale bar with label in the bottom-right corner of an image axis (pixel coordinates)."""
    x0, x1 = sorted(ax.get_xlim())
    y_top, y_bottom = sorted(ax.get_ylim())  # image axes: y increases downwards
    length_px = length_um / pixel_size_um
    xe = x1 - margin_frac * (x1 - x0)
    y = y_bottom - margin_frac * (y_bottom - y_top)
    ax.plot([xe - length_px, xe], [y, y], color=color, linewidth=linewidth, solid_capstyle='butt',
            zorder=30, path_effects=[pe.withStroke(linewidth=linewidth + 2, foreground='black')])
    ax.text(xe - length_px / 2, y - 0.012 * (y_bottom - y_top), f"{length_um:g} µm", color=color,
            fontsize=fontsize, fontweight='bold', ha='center', va='bottom', zorder=30,
            path_effects=[pe.withStroke(linewidth=3, foreground='black')])
