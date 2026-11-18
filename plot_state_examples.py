"""
Example Track per saSPT State
-----------------------------
Three panels (Immobile / Multimers / Monomers), one representative trajectory each.
The background of each panel is the raw movie frame of that trajectory's FIRST
localization (the first blink), cropped around it; the full trajectory is drawn
on top in its state colour. All crops share the same size, chosen so every
track fits, with the first localization at the centre. One 1 µm scale bar.

Representative track (default): among confidently assigned tracks of displayable
length (min_points-max_points), the one closest to its state's median diffusion
coefficient AND median radius of gyration, each distance measured in log10 and
scaled by that state's MAD:  score = |dlogD|/MAD_logD + |dlogRg|/MAD_logRg.
The lowest-scoring track with a clearly visible first blink is used.
selection='random' instead draws a random eligible track (reproducible seed).

Length-matched examples (target_length=N): the three tracks have the same number
of localizations (N +/- length_tol) and a similar duration (first-to-last frame
span within span_tol frames of the common median span), so differences between
panels come from mobility, not track length. "Representative" is then judged
within that same-length population of each state, which also makes the Rg
comparison fair (Rg grows with track length).

Frame mapping: CSV frames are 1-based (223-973 in a 973-page TIFF), so the
TIFF page is frame - 1. CSV x/y are pixel-centre coordinates, matching imshow.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as patches
from PIL import Image

from common import (CATEGORIES, STATE_COLORS, PIXEL_SIZE_UM, FRAME_INTERVAL_S, LOC_ERROR_UM,
                    load_tracks, add_scale_bar)
from saspt_classify import classify_trajectories


def blink_snr(img, x, y, frame):
    """Peak of the first blink above the local background, in background-noise units."""
    img.seek(frame - 1)  # CSV frames are 1-based
    a = np.array(img, dtype=np.float32)
    xi, yi = int(round(x)), int(round(y))
    win = a[yi - 10:yi + 11, xi - 10:xi + 11]
    bg = np.median(win)
    noise = 1.4826 * np.median(np.abs(win - bg))
    return (a[yi - 1:yi + 2, xi - 1:xi + 2].max() - bg) / max(noise, 1e-6)


def radius_of_gyration(df):
    """Per-trajectory radius of gyration (pixels)."""
    centred = df[['x', 'y']] - df.groupby('trajectory')[['x', 'y']].transform('mean')
    return np.sqrt((centred ** 2).sum(axis=1).groupby(df['trajectory']).mean())


def pick_examples(df, states, rng, min_points, max_points, min_confidence, width, height,
                  img=None, min_blink_snr=0.0, selection='representative', edge=40, max_tries=300,
                  target_length=None, length_tol=0, span_tol=None):
    """One confidently assigned trajectory per state, away from the FOV edges, whose
    first blink is clearly visible (SNR >= min_blink_snr). selection='representative'
    ranks candidates by closeness to the state's median D and radius of gyration;
    selection='random' tries them in random order.
    With target_length, all three tracks have target_length +/- length_tol
    localizations and (if span_tol is set) a frame span within span_tol frames of
    the median span of all confident tracks of that length."""
    lengths = df.groupby('trajectory').size()
    spans = df.groupby('trajectory')['frame'].agg(lambda f: f.max() - f.min() + 1)
    target_span = None
    if target_length is not None:
        min_points, max_points = target_length - length_tol, target_length + length_tol
        if span_tol is not None:
            conf_all = states[states['confidence'] >= min_confidence].index
            same_len = [t for t in conf_all if min_points <= lengths.get(t, 0) <= max_points]
            target_span = float(np.median(spans.loc[same_len]))
            print(f"Length-matched examples: {min_points}-{max_points} localizations, "
                  f"frame span {target_span:.0f} +/- {span_tol} frames")
    bounds = df.groupby('trajectory')[['x', 'y']].agg(['min', 'max'])
    inside = ((bounds[('x', 'min')] > edge) & (bounds[('x', 'max')] < width - edge) &
              (bounds[('y', 'min')] > edge) & (bounds[('y', 'max')] < height - edge))
    rg = radius_of_gyration(df)
    picks = []
    for cat in range(3):
        conf = states[(states['category'] == cat) & (states['confidence'] >= min_confidence)].index
        # Reference population: confident tracks of displayable length (Rg grows with length)
        ref = [t for t in conf if min_points <= lengths.get(t, 0) <= max_points]
        if target_span is not None:
            ref = [t for t in ref if abs(spans[t] - target_span) <= span_tol]
        ok = [t for t in ref if inside.get(t, False)]
        if not ok:
            raise RuntimeError(f"No candidate trajectory for {CATEGORIES[cat]}; relax the selection.")
        if selection == 'representative':
            logd = np.log10(states.loc[ref, 'D_mean'])
            logrg = np.log10(rg.loc[ref].clip(lower=1e-3))
            mad = lambda v: max(np.median(np.abs(v - np.median(v))), 1e-6)
            score = (np.abs(logd - np.median(logd)) / mad(logd) +
                     np.abs(logrg - np.median(logrg)) / mad(logrg)).loc[ok]
            order = score.sort_values().index.values
            print(f"{CATEGORIES[cat]:9s} reference ({len(ref):,} tracks): median D = {10 ** np.median(logd):.3f} um^2/s, "
                  f"median Rg = {10 ** np.median(logrg) * PIXEL_SIZE_UM * 1000:.0f} nm")
        else:
            order = rng.permutation(ok)
        first = df[df['trajectory'].isin(ok)].sort_values('frame').groupby('trajectory').first()
        for t in order[:max_tries]:
            r = first.loc[t]
            if img is None or blink_snr(img, r['x'], r['y'], int(r['frame'])) >= min_blink_snr:
                picks.append(int(t))
                break
        else:
            raise RuntimeError(f"No {CATEGORIES[cat]} trajectory with a clear first blink; lower min_blink_snr.")
    return picks


def common_half_size(tracks, pad_px=6):
    """Half-width (px) of a square crop centred on each track's first localization
    that fits every track in `tracks`."""
    half = max(np.abs(tr[['x', 'y']].values - tr[['x', 'y']].values[0]).max() for tr in tracks)
    return int(np.ceil(half)) + pad_px


def draw_example(ax, img, tr, half, cat, d_value, title_inside=False, number=None,
                 scale_bar_um=None, pixel_size_um=PIXEL_SIZE_UM, show_length=False):
    """Draw one example: first-blink frame cropped around the first localization,
    the full track in its state colour, a dashed ring around the first blink and
    the track's diffusion coefficient. With title_inside=True the state name (and
    optional zone number) is written inside the panel instead of above it.
    show_length adds the number of localizations and the track duration."""
    color = STATE_COLORS[cat]
    x0, y0, f0 = tr['x'].values[0], tr['y'].values[0], int(tr['frame'].values[0])

    img.seek(f0 - 1)  # CSV frames are 1-based
    frame = np.array(img, dtype=np.float32)
    xs, ys = int(round(x0)), int(round(y0))
    crop = frame[max(0, ys - half):ys + half + 1, max(0, xs - half):xs + half + 1]
    # Contrast: background floor from the crop, top set by the first blink so it reads white
    vmin = np.percentile(crop, 1)
    vmax = frame[ys - 1:ys + 2, xs - 1:xs + 2].max()

    ax.imshow(frame, cmap='gray', vmin=vmin, vmax=vmax, origin='upper', interpolation='nearest')
    ax.plot(tr['x'], tr['y'], color=color, linewidth=1.8, alpha=0.9, zorder=5,
            path_effects=[pe.withStroke(linewidth=3.0, foreground='black')])
    ax.plot(tr['x'], tr['y'], 'o', color=color, markersize=2.5, zorder=6)
    # Ring around the first localization, sized to sit outside the blink (radius 3 px)
    ax.add_patch(patches.Circle((x0, y0), 3.0, fill=False, edgecolor='white', linewidth=1.6,
                                linestyle=(0, (3, 2)), zorder=7))

    ax.set_xlim(x0 - half, x0 + half)
    ax.set_ylim(y0 + half, y0 - half)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)

    stroke = [pe.withStroke(linewidth=3, foreground='black')]
    d_text = f"{d_value:.3f} \u00b5m\u00b2/s"
    if show_length:
        n_frames = int(tr['frame'].values[-1] - tr['frame'].values[0] + 1)
        d_text += f"\n{len(tr)} loc. / {n_frames * FRAME_INTERVAL_S:.2f} s"
    if title_inside:
        name = f"{number}  {CATEGORIES[cat]}" if number is not None else CATEGORIES[cat]
        ax.text(0.04, 0.96, name, transform=ax.transAxes, ha='left', va='top', color=color,
                fontsize=20, fontweight='bold', path_effects=stroke, zorder=10)
        ax.text(0.04, 0.84, d_text, transform=ax.transAxes, ha='left', va='top', color='white',
                fontsize=16, fontweight='bold', path_effects=stroke, zorder=10)
    else:
        ax.set_title(CATEGORIES[cat], color=color, fontsize=20, fontweight='bold', pad=8)
        ax.text(0.04, 0.96, d_text, transform=ax.transAxes, ha='left', va='top', color='white',
                fontsize=16, fontweight='bold', path_effects=stroke, zorder=10)
    if scale_bar_um:
        add_scale_bar(ax, scale_bar_um, pixel_size_um, fontsize=14, linewidth=4)


def plot_state_examples(
    tif_path: str,
    csv_path: str,
    saspt_csv: str,
    output_path: str = "trajectories_by_state_examples.png",
    traj_ids: list = None,         # explicit [immobile, multimers, monomers] ids; else picked
    selection: str = 'representative',  # 'representative' or 'random'
    seed: int = 0,
    min_points: int = 15,
    max_points: int = 40,
    min_confidence: float = 0.9,
    min_blink_snr: float = 5.0,
    pad_px: int = 6,
    scale_bar_um: float = 1.0,
    pixel_size_um: float = PIXEL_SIZE_UM,
    loc_error_um: float = LOC_ERROR_UM,
    target_length: int = None,     # length-matched examples (number of localizations)
    length_tol: int = 0,
    span_tol: int = None,          # max deviation of the frame span from the common median
    dpi: int = 300,
):
    img = Image.open(tif_path)
    width, height = img.size

    df = load_tracks(csv_path)
    states = classify_trajectories(df, saspt_csv, pixel_size_um=pixel_size_um, loc_error_um=loc_error_um)

    if traj_ids is None:
        traj_ids = pick_examples(df, states, np.random.default_rng(seed), min_points, max_points,
                                 min_confidence, width, height, img=img, min_blink_snr=min_blink_snr,
                                 selection=selection, target_length=target_length,
                                 length_tol=length_tol, span_tol=span_tol)
    tracks = [df[df['trajectory'] == t].sort_values('frame') for t in traj_ids]
    rg = radius_of_gyration(df[df['trajectory'].isin(traj_ids)])

    # Common crop: first localization centred, size set so every track fits
    half = common_half_size(tracks, pad_px)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), dpi=dpi, facecolor='black')
    fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.005, wspace=0.03)

    for i, (ax, tid, tr) in enumerate(zip(axes, traj_ids, tracks)):
        cat = int(states.at[tid, 'category'])
        x0, y0, f0 = tr['x'].values[0], tr['y'].values[0], int(tr['frame'].values[0])
        draw_example(ax, img, tr, half, cat, states.at[tid, 'D_mean'],
                     scale_bar_um=scale_bar_um if i == len(axes) - 1 else None, pixel_size_um=pixel_size_um,
                     show_length=target_length is not None)

        print(f"{CATEGORIES[cat]:9s} trajectory {tid}: {len(tr)} localizations, "
              f"span {int(tr['frame'].values[-1]) - f0 + 1} frames, first frame {f0}, "
              f"start (x={x0:.1f}, y={y0:.1f}) px, confidence {states.at[tid, 'confidence']:.2f}, "
              f"D_mean {states.at[tid, 'D_mean']:.3f} um^2/s, Rg {rg.at[tid] * pixel_size_um * 1000:.0f} nm")
    print(f"Common crop: {2 * half} x {2 * half} px ({2 * half * pixel_size_um:.2f} um)")

    output_abs = os.path.abspath(output_path)
    fig.savefig(output_abs, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved example-track panel to: {output_abs}")
    return output_abs, traj_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One example trajectory per saSPT state on its first-blink frame")
    parser.add_argument("--tif", required=True, help="TIFF movie")
    parser.add_argument("--csv", required=True, help="Trajectory table (x, y, frame, trajectory)")
    parser.add_argument("--saspt-csv", required=True, help="saSPT distribution")
    parser.add_argument("--out", default="trajectories_by_state_examples.png")
    parser.add_argument("--traj-ids", type=int, nargs=3, metavar=("IMMOBILE", "MULTIMERS", "MONOMERS"),
                        help="Use these trajectory ids instead of picking them")
    parser.add_argument("--selection", choices=["representative", "random"], default="representative",
                        help="Pick the most representative track per state (default) or a random one")
    parser.add_argument("--seed", type=int, default=0, help="Seed for --selection random")
    parser.add_argument("--min-points", type=int, default=15)
    parser.add_argument("--max-points", type=int, default=40)
    parser.add_argument("--min-confidence", type=float, default=0.9)
    parser.add_argument("--min-blink-snr", type=float, default=5.0,
                        help="Minimum first-blink signal-to-noise")
    parser.add_argument("--match-length", type=int, default=None, metavar="N",
                        help="Pick length-matched examples: all three tracks have N localizations")
    parser.add_argument("--length-tol", type=int, default=0, help="Allowed deviation from --match-length")
    parser.add_argument("--span-tol", type=int, default=3,
                        help="With --match-length: max deviation (frames) of the track duration from the common median")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    plot_state_examples(args.tif, args.csv, args.saspt_csv, args.out, traj_ids=args.traj_ids,
                        selection=args.selection, seed=args.seed,
                        min_points=args.min_points, max_points=args.max_points,
                        min_confidence=args.min_confidence, min_blink_snr=args.min_blink_snr,
                        target_length=args.match_length, length_tol=args.length_tol,
                        span_tol=args.span_tol if args.match_length else None,
                        dpi=args.dpi)
