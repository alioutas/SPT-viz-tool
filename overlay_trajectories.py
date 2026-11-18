"""
Trajectory overlay
------------------
Draws every trajectory on the first frame of the movie (or on black), with an
optional white zoom box shown magnified in a top-right inset.

Colouring (color_by):
    id      random colour per trajectory
    length  number of localizations (log scale)
    frame   acquisition frame
    class   'class' column of the trajectory table
    state   saSPT population: Immobile / Multimers / Monomers (needs saspt_csv)
    msd     per-track diffusion coefficient from the MSD (see msd.py)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import colormaps  # noqa: F401  registers the custom colormaps
from common import (CATEGORIES, STATE_COLORS, PIXEL_SIZE_UM, LOC_ERROR_UM, MIN_LENGTH,
                    load_tracks, load_frame, contrast_stretch, add_scale_bar)
from saspt_classify import classify_trajectories
from msd import msd_diffusion

COLOR_MODES = ("id", "length", "frame", "class", "state", "msd")
COLORBAR_LABELS = {"length": "Track length", "frame": "Frame", "class": "Class", "msd": "D$_{MSD}$ (µm²/s)"}
LENGTH_TICKS = (5, 10, 20, 50, 100, 200, 500)
MSD_TICKS = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5)


def quantile_norm(values, low=0.01, high=0.99):
    """Rank scale: each part of the colormap holds the same number of values
    (between the `low` and `high` quantiles); a colourbar still shows real values."""
    levels = np.linspace(low, high, 99)
    qs = np.maximum.accumulate(np.quantile(values, levels))
    qs = qs + np.arange(len(qs)) * 1e-9  # strictly increasing for interpolation
    u = (levels - levels[0]) / (levels[-1] - levels[0])
    return mcolors.FuncNorm((lambda v: np.interp(v, qs, u), lambda w: np.interp(w, u, qs)),
                            vmin=qs[0], vmax=qs[-1])


def overlay_trajectories(
    tif_path: str,
    csv_path: str,
    output_path: str = "overlay_trajectories_frame0.png",
    color_by: str = "id",
    cmap_name: str = "turbo",
    saspt_csv: str = None,           # required for color_by='state'
    min_length: int = MIN_LENGTH,
    linewidth: float = 0.8,
    alpha: float = 0.7,
    show_image: bool = True,         # False: trajectories on black
    show_inset: bool = True,
    x_center: int = 250,             # zoom box centre and size (pixels)
    y_center: int = 650,
    box_size: int = 150,
    zoom_roi: tuple = None,          # or explicit (xmin, xmax, ymin, ymax)
    box_color: str = "white",
    scale_bars: bool = True,         # 10 µm on the full image, 1 µm on the inset
    show_colorbar: bool = True,
    show_title: bool = False,
    msd_scale: str = "quantile",     # color_by='msd': 'quantile' (equal tracks per colour) or 'log'
    msd_range: tuple = (0.01, 0.5),  # colour range (µm²/s) for msd_scale='log'
    pixel_size_um: float = PIXEL_SIZE_UM,
    loc_error_um: float = LOC_ERROR_UM,
    percentile_low: float = 1.0,     # background contrast
    percentile_high: float = 99.8,
    dpi: int = 300,
):
    if color_by not in COLOR_MODES:
        raise ValueError(f"color_by must be one of {COLOR_MODES}")
    frame0 = load_frame(tif_path)
    height, width = frame0.shape
    background = contrast_stretch(frame0, percentile_low, percentile_high) if show_image else np.zeros_like(frame0)

    if zoom_roi is None:
        h = box_size // 2
        zoom_roi = (max(0, x_center - h), min(width, x_center + h), max(0, y_center - h), min(height, y_center + h))
    z_xmin, z_xmax, z_ymin, z_ymax = zoom_roi

    df = load_tracks(csv_path, min_length)
    if df.empty:
        raise ValueError(f"No trajectories with at least {min_length} localizations.")
    print(f"{df['trajectory'].nunique():,} trajectories (>= {min_length} localizations)")

    if color_by == "state":
        if saspt_csv is None:
            raise ValueError("color_by='state' requires saspt_csv.")
        states = classify_trajectories(df, saspt_csv, pixel_size_um=pixel_size_um, loc_error_um=loc_error_um)
        df = df[df['trajectory'].isin(states.index)]
        state_counts = np.bincount(states['category'].values, minlength=3)
        for name, cnt in zip(CATEGORIES, state_counts):
            print(f"  {name}: {cnt:,} ({cnt / state_counts.sum() * 100:.1f}%)")
    elif color_by == "msd":
        msd_d = msd_diffusion(df, pixel_size_um=pixel_size_um, loc_error_um=loc_error_um)['D_msd']
        print(f"  D_MSD median {msd_d.median():.3f} um^2/s "
              f"(5-95%: {msd_d.quantile(.05):.3f}-{msd_d.quantile(.95):.3f})")

    # One line segment per pair of consecutive localizations
    df = df.sort_values(['trajectory', 'frame'])
    t = df['trajectory'].values
    xy = df[['x', 'y']].values
    same = t[1:] == t[:-1]
    segments = np.stack([xy[:-1][same], xy[1:][same]], axis=1)
    seg_tid = t[1:][same]

    if color_by == "id":
        values = ((seg_tid * 2654435761) % 2 ** 32) / 2 ** 32
    elif color_by == "length":
        values = df.groupby('trajectory').size().reindex(seg_tid).values
    elif color_by == "frame":
        values = df['frame'].values[:-1][same]
    elif color_by == "class":
        values = df['class'].values[:-1][same] if 'class' in df else np.full(len(seg_tid), 0.5)
    elif color_by == "state":
        values = states['category'].reindex(seg_tid).values
    else:  # msd
        values = msd_d.reindex(seg_tid).values

    cmap = plt.get_cmap(cmap_name)
    color_norm = None
    if color_by in ("state", "msd"):
        # Draw Monomers / fast tracks first so the compact, slow ones stay visible on top
        order = np.argsort(-values, kind='stable')
        segments, values = segments[order], values[order]
    if color_by == "state":
        colors = np.array([mcolors.to_rgba(c) for c in STATE_COLORS])[values.astype(int)]
    elif color_by == "id":
        colors = cmap(values)
    else:
        if color_by == "length" and values.min() > 0:
            color_norm = mcolors.LogNorm(vmin=values.min(), vmax=values.max())  # lengths are heavily skewed
        elif color_by == "msd" and msd_scale == "quantile":
            color_norm = quantile_norm(msd_d.values)
        elif color_by == "msd":
            color_norm = mcolors.LogNorm(vmin=msd_range[0], vmax=msd_range[1], clip=True)
        else:
            color_norm = plt.Normalize(vmin=values.min(), vmax=values.max())
        colors = cmap(np.clip(color_norm(values), 0, 1))

    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
    ax.imshow(background, cmap='gray', vmin=0, vmax=1, origin='upper')
    ax.add_collection(LineCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha))

    if show_inset:
        ax.add_patch(patches.Rectangle((z_xmin, z_ymin), z_xmax - z_xmin, z_ymax - z_ymin, linewidth=2.5,
                                       edgecolor=box_color, facecolor='none', zorder=10))
        ax_in = ax.inset_axes([0.62, 0.62, 0.35, 0.35])
        ax_in.imshow(background, cmap='gray', vmin=0, vmax=1, origin='upper')
        ax_in.add_collection(LineCollection(segments, colors=colors, linewidths=linewidth * 2.0,
                                            alpha=min(alpha + 0.15, 1.0)))
        ax_in.set_xlim(z_xmin, z_xmax)
        ax_in.set_ylim(z_ymax, z_ymin)
        for spine in ax_in.spines.values():
            spine.set_edgecolor(box_color)
            spine.set_linewidth(2.5)
        ax_in.text(0.04, 0.94, "Zoom", transform=ax_in.transAxes, color="white", fontsize=14,
                   fontweight="bold", va="top", ha="left", zorder=20)
        ax_in.set_xticks([])
        ax_in.set_yticks([])
        if scale_bars:
            add_scale_bar(ax_in, 1, pixel_size_um, fontsize=14, linewidth=4)

    if show_title:
        ax.set_title(f"Trajectories (>= {min_length} localizations, N = {df['trajectory'].nunique():,})\n"
                     f"{os.path.basename(tif_path)}", fontsize=12, pad=12, fontweight='semibold')
    ax.axis('off')
    if scale_bars:
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        add_scale_bar(ax, 10, pixel_size_um, fontsize=16, linewidth=5, margin_frac=0.03)

    if color_by == "state":
        handles = [Line2D([0], [0], color=c, lw=3, label=f"{n} ({cnt / state_counts.sum() * 100:.1f}%)")
                   for c, n, cnt in zip(STATE_COLORS, CATEGORIES, state_counts)]
        ax.legend(handles=handles, loc='lower left', fontsize=13, frameon=True, facecolor='black',
                  edgecolor='white', labelcolor='white', framealpha=0.7)

    if color_norm is not None:
        fig.set_facecolor('black')
        if show_colorbar:
            cax = make_axes_locatable(ax).append_axes("right", size=0.3, pad=0.4)
            cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=color_norm), cax=cax)
            cbar.set_label(COLORBAR_LABELS[color_by], fontsize=18, color='white', labelpad=10)
            if color_by == "msd" or isinstance(color_norm, mcolors.LogNorm):
                ticks = [v for v in (MSD_TICKS if color_by == "msd" else LENGTH_TICKS)
                         if color_norm.vmin <= v <= color_norm.vmax]
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{v:g}" for v in ticks])
                cbar.ax.minorticks_off()
            cbar.ax.tick_params(labelsize=15, colors='white')
            cbar.outline.set_edgecolor('white')

    fig.tight_layout()
    output_abs = os.path.abspath(output_path)
    fig.savefig(output_abs, dpi=dpi, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved {output_abs}")
    return output_abs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Overlay trajectories on the first movie frame, with a zoom inset.")
    p.add_argument("--tif", required=True, help="TIFF movie")
    p.add_argument("--csv", required=True, help="Trajectory table (x, y, frame, trajectory)")
    p.add_argument("--saspt-csv", help="saSPT distribution (needed for --color-by state)")
    p.add_argument("--out", default="overlay_trajectories_frame0.png", help="Output PNG")
    p.add_argument("--color-by", choices=COLOR_MODES, default="id")
    p.add_argument("--cmap", default="turbo", help="Matplotlib colormap, e.g. ibm_seq, abbelight_gradient")
    p.add_argument("--min-length", type=int, default=MIN_LENGTH, help="Minimum localizations per trajectory")
    p.add_argument("--linewidth", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--x-center", type=int, default=250, help="Zoom box centre x (pixels)")
    p.add_argument("--y-center", type=int, default=650, help="Zoom box centre y (pixels)")
    p.add_argument("--box-size", type=int, default=150, help="Zoom box size (pixels)")
    p.add_argument("--zoom-roi", type=int, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                   help="Explicit zoom region instead of centre/size")
    p.add_argument("--msd-scale", choices=["quantile", "log"], default="quantile",
                   help="Colour scale for --color-by msd")
    p.add_argument("--no-inset", action="store_true", help="No zoom inset")
    p.add_argument("--no-image", action="store_true", help="Black background instead of the movie frame")
    p.add_argument("--no-scale-bar", action="store_true")
    p.add_argument("--no-colorbar", action="store_true")
    p.add_argument("--show-title", action="store_true")
    p.add_argument("--pixel-size", type=float, default=PIXEL_SIZE_UM, help="Pixel size (µm)")
    p.add_argument("--loc-error", type=float, default=LOC_ERROR_UM, help="Localization error (µm)")
    p.add_argument("--dpi", type=int, default=300)
    a = p.parse_args()

    overlay_trajectories(
        a.tif, a.csv, a.out, color_by=a.color_by, cmap_name=a.cmap, saspt_csv=a.saspt_csv,
        min_length=a.min_length, linewidth=a.linewidth, alpha=a.alpha,
        show_image=not a.no_image, show_inset=not a.no_inset,
        x_center=a.x_center, y_center=a.y_center, box_size=a.box_size,
        zoom_roi=tuple(a.zoom_roi) if a.zoom_roi else None,
        scale_bars=not a.no_scale_bar, show_colorbar=not a.no_colorbar, show_title=a.show_title,
        msd_scale=a.msd_scale, pixel_size_um=a.pixel_size, loc_error_um=a.loc_error, dpi=a.dpi)
