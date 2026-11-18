"""
Render all application-note figures
-----------------------------------
    python render_figures.py                        # every figure into figures/
    python render_figures.py --only cover msd_zoom  # a subset
    python render_figures.py --list                 # figure keys

Data paths are read from config.json (copy config.example.json) and can be
overridden with --tif / --tracks / --saspt.
"""

import os
import json
import argparse
import matplotlib
matplotlib.use("Agg")

from overlay_trajectories import overlay_trajectories
from overlay_trajectories_multizoom import plot_multizoom
from plot_state_examples import plot_state_examples
from plot_saspt_log_vs_linear import plot_log_vs_linear

HERE = os.path.dirname(os.path.abspath(__file__))
MATCH_LENGTH = 24        # localizations per track in the length-matched examples
CENTRE_ZOOM = dict(x_center=698, y_center=698, box_size=150)  # 14.6 µm box at the FOV centre


def figures(out, tif, csv, saspt):
    """Figure key -> function that renders it into `out`."""
    p = lambda name: os.path.join(out, name)
    overlay = lambda name, **kw: lambda: overlay_trajectories(tif, csv, p(name), **kw)
    state = dict(color_by="state", saspt_csv=saspt, alpha=0.8)
    msd = dict(color_by="msd", cmap_name="abbelight_gradient", show_image=False)
    cover = dict(msd, scale_bars=False, show_colorbar=False)
    return {
        # saSPT distribution and populations (Immobile / Multimers / Monomers)
        "saspt": lambda: plot_log_vs_linear(saspt, p("saSPT_diffusion")),
        "state_overlay": overlay("overlay_trajectories_by_state.png", **state),
        "state_no_image": overlay("trajectories_by_state_no_image.png", **state, show_image=False),
        "state_no_image_no_zoom": overlay("trajectories_by_state_no_image_no_zoom.png", **state,
                                          show_image=False, show_inset=False),
        "state_multizoom": lambda: plot_multizoom(
            tif, csv, saspt, p("trajectories_by_state_multizoom.png"), layout="right", examples=True),
        # One representative track per population
        "examples": lambda: plot_state_examples(tif, csv, saspt, p("trajectories_by_state_examples.png")),
        "examples_matched": lambda: plot_state_examples(
            tif, csv, saspt, p("trajectories_by_state_examples_matched_length.png"),
            target_length=MATCH_LENGTH, span_tol=3),
        "multizoom_matched": lambda: plot_multizoom(
            tif, csv, saspt, p("trajectories_by_state_multizoom_matched_length.png"), layout="right",
            examples=True, match_length=MATCH_LENGTH, span_tol=3),
        # Abbelight-coloured overlays (front page)
        "abbelight": overlay("overlay_trajectories_frame0_abbelight_seq_no_zoom_no_image.png", color_by="length",
                             cmap_name="abbelight_seq", show_image=False, show_inset=False),
        "msd": overlay("overlay_trajectories_frame0_abbelight_gradient_msd.png", **msd, show_inset=False),
        "cover": overlay("overlay_trajectories_frame0_abbelight_seq_cover.png", **cover, show_inset=False),
        "msd_zoom": overlay("overlay_trajectories_frame0_abbelight_gradient_msd_zoom.png", **msd, **CENTRE_ZOOM),
        "cover_zoom": overlay("overlay_trajectories_frame0_abbelight_seq_cover_zoom.png", **cover, **CENTRE_ZOOM),
    }


def load_config(path):
    """Data paths from a JSON config; relative paths are taken from this folder."""
    cfg = {"tif": "data/movie.tif", "tracks": "data/tracks.csv", "saspt": "data/saspt_distribution.csv"}
    if os.path.exists(path):
        with open(path, encoding="utf8") as f:
            cfg.update(json.load(f))
    return {k: os.path.join(HERE, v) for k, v in cfg.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Render the application-note figures")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"), help="JSON file with data paths")
    ap.add_argument("--tif", help="TIFF movie (overrides config)")
    ap.add_argument("--tracks", help="Trajectory CSV (overrides config)")
    ap.add_argument("--saspt", help="saSPT distribution CSV (overrides config)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--only", nargs="+", metavar="KEY", help="Render only these figures")
    ap.add_argument("--list", action="store_true", help="List figure keys and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = {k: getattr(args, k) or cfg[k] for k in ("tif", "tracks", "saspt")}
    figs = figures(args.out_dir, paths["tif"], paths["tracks"], paths["saspt"])
    if args.list:
        print("\n".join(figs))
        raise SystemExit
    missing = [f"{k}: {v}" for k, v in paths.items() if not os.path.exists(v)]
    if missing:
        ap.error("data file(s) not found - edit config.json or pass --tif/--tracks/--saspt\n  " +
                 "\n  ".join(missing))
    keys = args.only or list(figs)
    unknown = [k for k in keys if k not in figs]
    if unknown:
        ap.error(f"unknown figure key(s) {unknown}; run with --list")

    os.makedirs(args.out_dir, exist_ok=True)
    for k in keys:
        print(f"\n##### {k}")
        figs[k]()
