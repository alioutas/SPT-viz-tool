"""
Per-trajectory saSPT population assignment
------------------------------------------
Assigns each trajectory to Immobile / Multimers / Monomers with the Bayesian
step of state-array SPT: the population saSPT distribution p(D) is the prior,
each trajectory's jumps give a Brownian-with-localization-error likelihood on
the same D grid, and the trajectory goes to the population holding the largest
share of its posterior.

Per jump spanning k frames (2D):  r^2 ~ 4 (D k dt + sigma^2)
Only jumps of up to `max_gap` frames are used (default 1: consecutive frames,
which best reproduces the saSPT population fractions for this dataset).
"""

import numpy as np
import pandas as pd

from common import CATEGORIES, PIXEL_SIZE_UM, FRAME_INTERVAL_S, LOC_ERROR_UM  # noqa: F401  CATEGORIES re-exported


def classify_trajectories(
    df: pd.DataFrame,
    saspt_csv: str,
    pixel_size_um: float = PIXEL_SIZE_UM,
    frame_interval_s: float = FRAME_INTERVAL_S,
    loc_error_um: float = LOC_ERROR_UM,
    immobile_cutoff: float = 0.04,
    multimer_cutoff: float = 0.18,
    max_gap: int = 1,
) -> pd.DataFrame:
    """Return a DataFrame indexed by trajectory with 'category' (0 Immobile,
    1 Multimers, 2 Monomers), 'confidence' (posterior mass of that category)
    and 'D_mean' (posterior mean D, µm²/s)."""
    sp = pd.read_csv(saspt_csv)
    grid = sp['Diffusion_Coefficient'].values
    prior = sp['Probability_Density'].values
    prior = prior / prior.sum()
    grid_cat = np.digitize(grid, [immobile_cutoff, multimer_cutoff], right=True)

    d = df.sort_values(['trajectory', 'frame'])
    t = d['trajectory'].values
    f = d['frame'].values
    x = d['x'].values * pixel_size_um
    y = d['y'].values * pixel_size_um

    gap = np.diff(f)
    ok = (t[1:] == t[:-1]) & (gap >= 1) & (gap <= max_gap)
    jumps = pd.DataFrame({
        'trajectory': t[1:][ok],
        'k': gap[ok],
        'r2': (np.diff(x) ** 2 + np.diff(y) ** 2)[ok],
    })
    stats = jumps.groupby(['trajectory', 'k'])['r2'].agg(['sum', 'count']).unstack('k', fill_value=0)
    traj_ids = stats.index.values

    log_post = np.tile(np.log(prior + 1e-300), (len(traj_ids), 1))
    for k in range(1, max_gap + 1):
        if ('sum', k) not in stats.columns:
            continue
        S = stats[('sum', k)].values[:, None]
        N = stats[('count', k)].values[:, None]
        var = grid[None, :] * k * frame_interval_s + loc_error_um ** 2
        log_post += -N * np.log(var) - S / (4 * var)

    log_post -= log_post.max(axis=1, keepdims=True)
    post = np.exp(log_post)
    post /= post.sum(axis=1, keepdims=True)

    cat_mass = np.stack([post[:, grid_cat == c].sum(axis=1) for c in range(3)], axis=1)
    return pd.DataFrame({
        'category': cat_mass.argmax(axis=1),
        'confidence': cat_mass.max(axis=1),
        'D_mean': post @ grid,
    }, index=pd.Index(traj_ids, name='trajectory'))
