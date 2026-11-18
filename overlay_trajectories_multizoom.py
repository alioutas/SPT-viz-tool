"""
Multi-Zoom Trajectory Figure
----------------------------
Full field of view with three numbered zoom boxes, plus the three zoomed panels
(numbered 1-3) either in a row on top (layout='top') or in a column directly to
the right of the FOV (layout='right', colourbar/legend at the far right).

Trajectories are coloured by saSPT state (color_by='state', categorical legend)
or by track length (color_by='length', log-scaled colourbar; length is the
number of localizations, i.e. frames, in the trajectory).

With examples=True (state colouring only) the zooms are replaced by the most
representative track of each state (Immobile, Multimers, Monomers, top to bottom),
each shown on its first-blink frame as in plot_state_examples.py; their crop
regions are boxed and numbered on the full FOV. Add match_length=N to make the
three example tracks length-matched (N localizations, similar duration; see
plot_state_examples.py).

The default zones are 1 = the original zoom (250, 650) and 2-3 = two zones that
were picked at random with seed 0. Any zones not given are picked at random
(reproducible seed) inside areas that contain trajectories, spread apart.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image

import colormaps  # noqa: F401  registers the custom colormaps
from common import (CATEGORIES, STATE_COLORS, PIXEL_SIZE_UM, LOC_ERROR_UM, MIN_LENGTH,
                    load_tracks, load_frame, contrast_stretch, add_scale_bar)
from saspt_classify import classify_trajectories
from plot_state_examples import pick_examples, common_half_size, draw_example

DEFAULT_ZONES = [(250, 650), (628, 1230), (1160, 1099)]

NUMBER_STYLE = dict(color='white', fontweight='bold', zorder=20,
                    path_effects=[pe.withStroke(linewidth=3, foreground='black')])


def pick_random_zones(df, width, height, box_size, fixed_centres, n_zones, seed=0,
                      margin=20, min_separation=None, max_tries=20000):
    """Random box centres in trajectory-dense areas, kept at least `min_separation`
    pixels apart (centre to centre; default 3 box widths) so zones spread across the FOV."""
    rng = np.random.default_rng(seed)
    # Point density per box-sized tile, used to reject empty regions
    H, xe, ye = np.histogram2d(df['x'], df['y'], bins=[width // 10, height // 10],
                               range=[[0, width], [0, height]])
    half = box_size // 2
    if min_separation is None:
        min_separation = 3 * box_size

    def density(cx, cy):
        ix = slice(int((cx - half) / 10), int((cx + half) / 10))
        iy = slice(int((cy - half) / 10), int((cy + half) / 10))
        return H[ix, iy].sum()

    lo_x, hi_x = half + margin, width - half - margin
    lo_y, hi_y = half + margin, height - half - margin
    samples = [density(cx, cy) for cx, cy in
               rng.integers([lo_x, lo_y], [hi_x, hi_y], size=(500, 2))]
    min_density = np.median(samples)

    centres = list(fixed_centres)
    for _ in range(max_tries):
        if len(centres) >= n_zones:
            break
        cx, cy = rng.integers(lo_x, hi_x), rng.integers(lo_y, hi_y)
        clear = all(np.hypot(cx - ox, cy - oy) >= min_separation for ox, oy in centres)
        if clear and density(cx, cy) >= min_density:
            centres.append((int(cx), int(cy)))
    if len(centres) < n_zones:
        raise RuntimeError("Could not place all zoom zones; try a smaller box size.")
    return centres


def plot_multizoom(
    tif_path: str,
    csv_path: str,
    saspt_csv: str = None,
    output_path: str = "trajectories_by_state_multizoom.png",
    color_by: str = "state",     # 'state' or 'length'
    layout: str = "top",         # 'top' or 'right'
    cmap_name: str = "ibm_seq",  # used for color_by='length'
    zone_centres: list = tuple(DEFAULT_ZONES),
    examples: bool = False,      # zooms = representative track per state (color_by='state')
    match_length: int = None,    # with examples: all example tracks have this many localizations
    length_tol: int = 0,
    span_tol: int = 3,           # with match_length: max deviation (frames) of track duration
    n_zones: int = 3,
    box_size: int = 150,
    seed: int = 0,
    min_length: int = MIN_LENGTH,
    linewidth: float = 0.8,
    alpha: float = 0.8,
    show_image: bool = False,
    scale_bars: bool = True,     # 10 µm bar on the full FOV, 1 µm bar on the last zoom panel
    pixel_size_um: float = PIXEL_SIZE_UM,
    loc_error_um: float = LOC_ERROR_UM,
    percentile_low: float = 1.0,
    percentile_high: float = 99.8,
    dpi: int = 300,
):
    img = Image.open(tif_path)  # kept open: example panels read their first-blink frames
    frame0 = load_frame(tif_path)
    height, width = frame0.shape
    background = contrast_stretch(frame0, percentile_low, percentile_high) if show_image else np.zeros_like(frame0)

    df = load_tracks(csv_path, min_length)
    counts = df['trajectory'].value_counts()
    if color_by == "state":
        if saspt_csv is None:
            raise ValueError("color_by='state' requires saspt_csv.")
        states = classify_trajectories(df, saspt_csv, pixel_size_um=pixel_size_um, loc_error_um=loc_error_um)
        df = df[df['trajectory'].isin(states.index)]
        state_counts = np.bincount(states['category'].values, minlength=3)
    df = df.sort_values(['trajectory', 'frame'])

    t = df['trajectory'].values
    xy = df[['x', 'y']].values
    same = t[1:] == t[:-1]
    segments = np.stack([xy[:-1][same], xy[1:][same]], axis=1)
    if color_by == "state":
        seg_cat = states['category'].reindex(t[1:][same]).values
        # Draw Monomers first and Immobile last so the rarer populations stay visible
        order = np.argsort(-seg_cat, kind='stable')
        segments, seg_cat = segments[order], seg_cat[order]
        colors = np.array([mcolors.to_rgba(c) for c in STATE_COLORS])[seg_cat]
    elif color_by == "length":
        seg_len = counts.reindex(t[1:][same]).values
        # Track lengths are heavily skewed (median ~7, max ~600), so use a log scale
        color_norm = mcolors.LogNorm(vmin=seg_len.min(), vmax=seg_len.max())
        cmap = plt.get_cmap(cmap_name)
        colors = cmap(color_norm(seg_len))
    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    # Zoom zones
    if examples:
        if color_by != "state":
            raise ValueError("examples=True needs color_by='state'.")
        traj_ids = pick_examples(df, states, np.random.default_rng(seed), 15, 40, 0.9, width, height,
                                 img=img, min_blink_snr=5.0, selection='representative',
                                 target_length=match_length, length_tol=length_tol,
                                 span_tol=span_tol if match_length else None)
        ex_tracks = [df[df['trajectory'] == t].sort_values('frame') for t in traj_ids]
        half = common_half_size(ex_tracks)
        centres = [(tr['x'].values[0], tr['y'].values[0]) for tr in ex_tracks]
        n_zones = len(centres)
        for i, (tid, tr) in enumerate(zip(traj_ids, ex_tracks), 1):
            print(f"Zone {i}: {CATEGORIES[int(states.at[tid, 'category'])]} trajectory {tid}, "
                  f"start ({tr['x'].values[0]:.1f}, {tr['y'].values[0]:.1f}), crop {2 * half} px")
    else:
        centres = pick_random_zones(df, width, height, box_size, zone_centres, n_zones, seed=seed)
        half = box_size // 2
        for i, (cx, cy) in enumerate(centres, 1):
            print(f"Zone {i}: centre=({cx}, {cy}), X=[{cx - half}:{cx + half}], Y=[{cy - half}:{cy + half}]")
    zones = [(cx - half, cx + half, cy - half, cy + half) for cx, cy in centres]

    if layout == "top":
        # Three square zooms on top, full FOV below (same total width)
        fig = plt.figure(figsize=(12, 16.3), dpi=dpi, facecolor='black')
        gs = fig.add_gridspec(2, n_zones, height_ratios=[1, n_zones], hspace=0.03, wspace=0.03,
                              left=0, right=1, top=1, bottom=0)
        ax_full = fig.add_subplot(gs[1, :])
        zoom_slots = [gs[0, i] for i in range(n_zones)]
    elif layout == "right":
        # Full FOV on the left, zooms stacked directly against its right edge,
        # colourbar strip at the far right (length only; the state legend sits inside the FOV)
        fov, col, cbar_w = 12.0, 12.0 / n_zones, (2.0 if color_by == "length" else 0.0)
        fig_w = fov + col + cbar_w
        fig = plt.figure(figsize=(fig_w, fov), dpi=dpi, facecolor='black')
        gs = fig.add_gridspec(n_zones, 2, width_ratios=[n_zones, 1], hspace=0, wspace=0,
                              left=0, right=(fov + col) / fig_w, top=1, bottom=0)
        ax_full = fig.add_subplot(gs[:, 0])
        zoom_slots = [gs[i, 1] for i in range(n_zones)]
    else:
        raise ValueError(f"Unknown layout: {layout}")

    ax_full.imshow(background, cmap='gray', vmin=0, vmax=1, origin='upper')
    ax_full.add_collection(LineCollection(segments, colors=colors, linewidths=linewidth, alpha=alpha))
    ax_full.set_xlim(0, width)
    ax_full.set_ylim(height, 0)
    ax_full.axis('off')
    if scale_bars:
        add_scale_bar(ax_full, 10, pixel_size_um, fontsize=16, linewidth=5, margin_frac=0.03)

    for i, (x0, x1, y0, y1) in enumerate(zones, 1):
        ax_full.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2.5,
                                            edgecolor='white', facecolor='none', zorder=10))
        if examples:
            # Example crops are small, so the number goes just outside the box (top-left)
            ax_full.text(x0 - 4, y0 - 4, str(i), ha='right', va='bottom', fontsize=16, **NUMBER_STYLE)
            ax_z = fig.add_subplot(zoom_slots[i - 1])
            tid = traj_ids[i - 1]
            draw_example(ax_z, img, ex_tracks[i - 1], half, int(states.at[tid, 'category']),
                         states.at[tid, 'D_mean'], title_inside=True, number=i,
                         scale_bar_um=1 if (scale_bars and i == len(zones)) else None,
                         pixel_size_um=pixel_size_um, show_length=match_length is not None)
            continue
        ax_full.text(x0 + 8, y0 + 8, str(i), ha='left', va='top', fontsize=16, **NUMBER_STYLE)

        ax_z = fig.add_subplot(zoom_slots[i - 1])
        ax_z.imshow(background, cmap='gray', vmin=0, vmax=1, origin='upper')
        ax_z.add_collection(LineCollection(segments, colors=colors, linewidths=linewidth * 2.0,
                                           alpha=min(alpha + 0.15, 1.0)))
        ax_z.set_xlim(x0, x1)
        ax_z.set_ylim(y1, y0)
        ax_z.set_xticks([])
        ax_z.set_yticks([])
        for spine in ax_z.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(2.5)
        ax_z.text(0.05, 0.95, str(i), transform=ax_z.transAxes, ha='left', va='top',
                  fontsize=20, **NUMBER_STYLE)
        if scale_bars and i == len(zones):
            add_scale_bar(ax_z, 1, pixel_size_um, fontsize=14, linewidth=4)

    if color_by == "state":
        handles = [Line2D([0], [0], color=c, lw=3, label=f"{n} ({cnt / state_counts.sum() * 100:.1f}%)")
                   for c, n, cnt in zip(STATE_COLORS, CATEGORIES, state_counts)]
        if layout == "right":
            # Legend in the top-right corner of the full FOV
            for h in handles:
                h.set_linewidth(6)
            ax_full.legend(handles=handles, loc='upper right', fontsize=16, frameon=True, facecolor='black',
                           edgecolor='white', labelcolor='white', framealpha=0.75, handlelength=1.6)
        else:
            ax_full.legend(handles=handles, loc='lower left', fontsize=13, frameon=True, facecolor='black',
                           edgecolor='white', labelcolor='white', framealpha=0.7)
    else:
        sm = plt.cm.ScalarMappable(norm=color_norm, cmap=cmap)
        if layout == "right":
            # Colourbar spans the full figure height
            cax = fig.add_axes([(fov + col + 0.4) / fig_w, 0, 0.3 / fig_w, 1])
            cbar = fig.colorbar(sm, cax=cax)
        else:
            cbar = fig.colorbar(sm, ax=ax_full, fraction=0.035, pad=0.02)
        ticks = [v for v in (5, 10, 20, 50, 100, 200, 500) if color_norm.vmin <= v <= color_norm.vmax]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([str(v) for v in ticks])
        cbar.ax.minorticks_off()
        cbar.set_label("Track length", fontsize=18, color='white', labelpad=10)
        cbar.ax.tick_params(labelsize=15, colors='white')
        cbar.outline.set_edgecolor('white')

    output_abs = os.path.abspath(output_path)
    fig.savefig(output_abs, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"Saved multi-zoom figure to: {output_abs}")
    return output_abs, centres


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full FOV with three numbered zoom panels, trajectories coloured by saSPT state or length")
    parser.add_argument("--tif", required=True, help="TIFF movie")
    parser.add_argument("--csv", required=True, help="Trajectory table (x, y, frame, trajectory)")
    parser.add_argument("--saspt-csv", help="saSPT distribution (needed for --color-by state)")
    parser.add_argument("--out", default="trajectories_by_state_multizoom.png")
    parser.add_argument("--zone", type=int, nargs=2, action="append", metavar=("X", "Y"),
                        help="Zoom centre in pixels; repeat up to 3 times. Missing zones are picked at random.")
    parser.add_argument("--color-by", choices=["state", "length"], default="state")
    parser.add_argument("--layout", choices=["top", "right"], default="top",
                        help="Zoom panels in a row on top, or in a column right of the FOV")
    parser.add_argument("--cmap", default="ibm_seq", help="Colormap for --color-by length")
    parser.add_argument("--examples", action="store_true",
                        help="Zooms show the most representative track of each state (with --color-by state)")
    parser.add_argument("--match-length", type=int, default=None, metavar="N",
                        help="With --examples: all example tracks have N localizations and a similar duration")
    parser.add_argument("--length-tol", type=int, default=0, help="Allowed deviation from --match-length")
    parser.add_argument("--span-tol", type=int, default=3,
                        help="With --match-length: max deviation (frames) of the track duration from the common median")
    parser.add_argument("--box-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=0, help="Seed for the randomly placed zones")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--show-image", action="store_true", help="Draw the movie frame behind the trajectories")
    parser.add_argument("--no-scale-bar", action="store_true", help="Omit the 10 µm / 1 µm scale bars")
    parser.add_argument("--loc-error", type=float, default=LOC_ERROR_UM)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    plot_multizoom(args.tif, args.csv, args.saspt_csv, args.out,
                   color_by=args.color_by, layout=args.layout, cmap_name=args.cmap,
                   zone_centres=[tuple(z) for z in args.zone] if args.zone else DEFAULT_ZONES,
                   examples=args.examples, match_length=args.match_length,
                   length_tol=args.length_tol, span_tol=args.span_tol,
                   box_size=args.box_size, seed=args.seed, alpha=args.alpha,
                   show_image=args.show_image, scale_bars=not args.no_scale_bar,
                   loc_error_um=args.loc_error, dpi=args.dpi)
