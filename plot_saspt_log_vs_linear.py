"""
saSPT Diffusion Coefficient Distribution: linear D vs log10(D)
--------------------------------------------------------------
The saSPT distribution plotted against linear D and against log10(D), saved
as two figures (<prefix>_linear.png, <prefix>_log.png).

The CSV column 'Probability_Density' holds per-bin probabilities on a
log-spaced D grid (it sums to 1), so each panel converts it to a proper
density for its own x variable:
    linear panel : P(D)        = p_i / dD_i          (per µm²/s)
    log panel    : P(log10 D)  = p_i / dlog10(D)_i   (per decade)
Population fractions are the bin sums, which are identical in both views.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from common import CATEGORIES, STATE_COLORS

C_IMMOBILE, C_MULTIMERS, C_MONOMERS = (np.array(mcolors.to_rgb(c)) for c in STATE_COLORS)
# Darker label shades for legibility on white
LABEL_COLORS = ('#3D6FE0', '#DC267F', '#B37A00')


def log_bin_edges(D):
    """Bin edges for a log-spaced grid of bin centres."""
    logD = np.log10(D)
    mid = (logD[:-1] + logD[1:]) / 2
    first = logD[0] - (mid[0] - logD[0])
    last = logD[-1] + (logD[-1] - mid[-1])
    return 10 ** np.concatenate([[first], mid, [last]])


def blended_colors(D, immobile_cutoff, multimer_cutoff, blend_width):
    """Smooth 3-population colour ramp, blended in log10(D) space."""
    log_d1, log_d2 = np.log10(immobile_cutoff), np.log10(multimer_cutoff)
    colors = []
    for lD in np.log10(D):
        if lD <= log_d1 - blend_width / 2:
            c = C_IMMOBILE
        elif lD <= log_d1 + blend_width / 2:
            t = (lD - (log_d1 - blend_width / 2)) / blend_width
            c = (1 - t) * C_IMMOBILE + t * C_MULTIMERS
        elif lD <= log_d2 - blend_width / 2:
            c = C_MULTIMERS
        elif lD <= log_d2 + blend_width / 2:
            t = (lD - (log_d2 - blend_width / 2)) / blend_width
            c = (1 - t) * C_MULTIMERS + t * C_MONOMERS
        else:
            c = C_MONOMERS
        colors.append(c)
    return colors


def draw_panel(ax, x, y, colors, cutoffs, label_x, fracs, xlabel, ylabel, title,
               label_y=(0.80, 0.80, 0.80), ymax=None):
    for i in range(len(x) - 1):
        c_mid = (colors[i] + colors[i + 1]) / 2.0
        for lvl in range(6):
            f1, f2 = lvl / 6.0, (lvl + 1) / 6.0
            alpha = 0.08 + 0.42 * ((lvl + 1) / 6.0) ** 0.8
            ax.fill_between(x[i:i + 2], y[i:i + 2] * f1, y[i:i + 2] * f2,
                            color=c_mid, alpha=alpha, linewidth=0, zorder=2)
    ax.plot(x, y, color='#1A202C', linewidth=2.0, zorder=5)

    for xc in cutoffs:
        ax.axvline(xc, color='#718096', linestyle='--', linewidth=1.2, zorder=4)

    for lx, ly, name, frac, col in zip(label_x, label_y, CATEGORIES, fracs, LABEL_COLORS):
        ax.text(lx, ly, f'{name}\n({frac * 100:.1f}%)', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=9, fontweight='bold', color=col, zorder=6)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, (ymax if ymax is not None else y.max()) * 1.25)
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel(ylabel, fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color='#1A202C')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.4, color='#CBD5E0')


def plot_log_vs_linear(
    csv_path: str,
    output_prefix: str = "saSPT_diffusion",
    immobile_cutoff: float = 0.04,
    multimer_cutoff: float = 0.18,
    blend_width: float = 0.20,
    lin_xmax: float = 0.5,
    dpi: int = 300
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"saSPT CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    D = df['Diffusion_Coefficient'].values
    p = df['Probability_Density'].values
    p = p / p.sum()

    edges = log_bin_edges(D)
    dD = np.diff(edges)
    dlogD = np.diff(np.log10(edges))
    dens_lin = p / dD          # per µm²/s
    dens_log = p / dlogD       # per decade

    immobile = D <= immobile_cutoff
    multimer = (D > immobile_cutoff) & (D <= multimer_cutoff)
    monomer = D > multimer_cutoff
    fracs = (p[immobile].sum(), p[multimer].sum(), p[monomer].sum())
    for name, f in zip(CATEGORIES, fracs):
        print(f"{name}: {f * 100:.1f}%")

    colors = blended_colors(D, immobile_cutoff, multimer_cutoff, blend_width)
    # Label at the log-centre of each region
    lo, hi = D.min(), D.max()
    centres = (np.sqrt(lo * immobile_cutoff), np.sqrt(immobile_cutoff * multimer_cutoff),
               np.sqrt(multimer_cutoff * hi))

    unit = r'$\mu\mathrm{m}^2\,\mathrm{s}^{-1}$'
    title = 'saSPT Diffusion Coefficient Distribution'
    outputs = []

    def save(fig, suffix):
        plt.tight_layout()
        out = os.path.abspath(f"{output_prefix}_{suffix}.png")
        fig.savefig(out, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {suffix} figure to: {out}")
        outputs.append(out)

    style = {'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
             'axes.edgecolor': '#2D3748', 'axes.linewidth': 1.2}

    # --- Linear D figure ---
    with plt.rc_context(style):
        fig, ax_lin = plt.subplots(figsize=(11, 5.5), dpi=dpi)
        draw_panel(ax_lin, D, dens_lin, colors, (immobile_cutoff, multimer_cutoff),
                   (0.03, centres[1], (multimer_cutoff + lin_xmax) / 2), fracs,
                   xlabel=f'Diffusion coefficient D ({unit})',
                   ylabel=r'Probability density $P(D)$ (per ' + unit + ')',
                   title=title, label_y=(0.80, 0.55, 0.80),
                   ymax=dens_lin[1:].max())
        # Show 0 to lin_xmax (µm²/s); fractions still use the full range
        ax_lin.set_xlim(0, lin_xmax)
        # The first bin (D = D.min) is the saSPT grid-edge pile-up; on a linear axis its
        # tiny dD makes it ~40x taller than the rest, so the y-axis is scaled to the other bins.
        save(fig, 'linear')

        # --- log10(D) figure ---
        fig, ax_log = plt.subplots(figsize=(9, 5.5), dpi=dpi)
        draw_panel(ax_log, np.log10(D), dens_log, colors,
                   (np.log10(immobile_cutoff), np.log10(multimer_cutoff)),
                   np.log10(centres), fracs,
                   xlabel=r'$\log_{10}$(D / ' + unit + ')',
                   ylabel=r'Probability density $P(\log_{10} D)$ (per decade)',
                   title=f'{title} ' + r'($\log_{10}$ D)')
        save(fig, 'log')

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot saSPT distribution as separate linear D and log10(D) figures")
    parser.add_argument("--csv", required=True, help="saSPT distribution CSV (Diffusion_Coefficient, Probability_Density)")
    parser.add_argument("--out-prefix", type=str, default="saSPT_diffusion", help="Output prefix; writes <prefix>_linear.png and <prefix>_log.png")
    parser.add_argument("--immobile-cutoff", type=float, default=0.04, help="Upper cutoff for Immobile population (µm²/s)")
    parser.add_argument("--multimer-cutoff", type=float, default=0.18, help="Upper cutoff for Multimers population (µm²/s)")
    parser.add_argument("--lin-xmax", type=float, default=0.5, help="Upper D limit (µm²/s) for the linear D figure")
    parser.add_argument("--dpi", type=int, default=300, help="DPI resolution for saved plots")
    args = parser.parse_args()
    plot_log_vs_linear(args.csv, args.out_prefix, args.immobile_cutoff, args.multimer_cutoff,
                       lin_xmax=args.lin_xmax, dpi=args.dpi)
