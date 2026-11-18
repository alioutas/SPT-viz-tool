"""
Per-trajectory MSD diffusion coefficient
----------------------------------------
For every trajectory the time-averaged MSD is computed at lags of 1..max_lag
frames, using the true frame difference between localizations (so blinking
gaps are handled). D is then fitted from

    MSD(k) = 4 D k dt + 4 sigma^2          (2D Brownian motion)

with the localization-error offset fixed at 4 sigma^2 (sigma = loc_error_um),
weighting each lag by its number of displacements:

    D = sum_k n_k k (MSD_k - 4 sigma^2) / (4 dt sum_k n_k k^2)

Fixing the offset keeps the fit defined for short tracks (5+ localizations).
Values are floored at d_min so they can be shown on a log scale.
"""

import numpy as np
import pandas as pd

from common import PIXEL_SIZE_UM, FRAME_INTERVAL_S, LOC_ERROR_UM


def msd_diffusion(df: pd.DataFrame, pixel_size_um: float = PIXEL_SIZE_UM,
                  frame_interval_s: float = FRAME_INTERVAL_S, loc_error_um: float = LOC_ERROR_UM,
                  max_lag: int = 4, d_min: float = 1e-3) -> pd.DataFrame:
    """Return a DataFrame indexed by trajectory with 'D_msd' (µm²/s) and
    'n_disp' (number of displacements used)."""
    d = df.sort_values(['trajectory', 'frame'])
    tid, codes = np.unique(d['trajectory'].values, return_inverse=True)
    f = d['frame'].values
    x = d['x'].values * pixel_size_um
    y = d['y'].values * pixel_size_um

    n = len(tid)
    num = np.zeros(n)   # sum_k n_k k (MSD_k - 4 s^2)  ==  sum over displacements of k (r2 - 4 s^2)
    den = np.zeros(n)   # sum_k n_k k^2
    cnt = np.zeros(n)
    off = 4 * loc_error_um ** 2
    # A lag of k frames is at most k rows apart (fewer if frames are missing)
    for j in range(1, max_lag + 1):
        same = codes[j:] == codes[:-j]
        k = f[j:] - f[:-j]
        ok = same & (k >= 1) & (k <= max_lag)
        r2 = (x[j:] - x[:-j]) ** 2 + (y[j:] - y[:-j]) ** 2
        c, kk = codes[j:][ok], k[ok]
        num += np.bincount(c, weights=kk * (r2[ok] - off), minlength=n)
        den += np.bincount(c, weights=kk.astype(float) ** 2, minlength=n)
        cnt += np.bincount(c, minlength=n)

    with np.errstate(invalid='ignore', divide='ignore'):
        D = num / (4 * frame_interval_s * den)
    D = np.where(den > 0, np.maximum(D, d_min), np.nan)
    return pd.DataFrame({'D_msd': D, 'n_disp': cnt.astype(int)},
                        index=pd.Index(tid, name='trajectory'))
