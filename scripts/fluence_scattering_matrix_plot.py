#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from tqdm import tqdm

import chimefrb_selection as cfsf

def build_selection_function():
    return cfsf.SelectionFunction(
        predictor_names=["fluence", "scattering_time", "width", "dm"],
        degree=3,
        snr_cut=12.0,
        exclude_sidelobes=True,
        sidelobe_cut=5.0,
        reweighted=False,
    )


def compute_selection_grid(sf, fluence_mesh, scat_mesh, width_ms, dm):
    p = np.full_like(fluence_mesh, np.nan, dtype=float)
    n_y, n_x = fluence_mesh.shape
    for i in tqdm(range(n_y), desc=f"Computing p_det (W={width_ms:g} ms, DM={dm})", leave=False):
        for j in range(n_x):
            try:
                p[i, j] = sf.calculate_selection_probability(
                    {
                        "fluence_jy_ms": float(fluence_mesh[i, j]),
                        "tau_1_ghz_ms": float(scat_mesh[i, j]),
                        "pulse_width_ms": float(width_ms),
                        "dm": float(dm),
                    }
                )
            except Exception:
                p[i, j] = np.nan
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="./plots/selection_probability_grid.pdf")
    parser.add_argument("--n", type=int, default=200)
    args = parser.parse_args()

    fluence_grid = np.logspace(-1, 4, args.n)      # Jy ms
    scattering_grid = np.logspace(-3, 2, args.n)   # ms
    fluence_mesh, scat_mesh = np.meshgrid(fluence_grid, scattering_grid)

    dms = [100, 500, 3000]
    widths_ms = [1e-1, 5e0, 2e1]

    sf = build_selection_function()

    fig = plt.figure(figsize=(18, 18))
    gs = fig.add_gridspec(
        nrows=4, ncols=3,
        height_ratios=[1, 1, 1, 0.10],   # last row reserved for colorbar
        left=0.07, right=0.95, top=0.95, bottom=0.05,
        wspace=0.28, hspace=0.32,
    )

    axes = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])

    cax = fig.add_subplot(gs[3, :])  # colorbar axis spanning all columns

    levels = np.arange(0.0, 1.0001, 0.05)
    last_contour = None

    x_locator = LogLocator(base=10.0, subs=(1.0,))
    y_locator = LogLocator(base=10.0, subs=(1.0,))
    log_fmt = LogFormatterMathtext(base=10.0)

    for r, w in tqdm(list(enumerate(widths_ms)), desc="Generating rows"):
        for c, dm in enumerate(dms):
            ax = axes[r, c]
            p = compute_selection_grid(sf, fluence_mesh, scat_mesh, width_ms=w, dm=dm)

            last_contour = ax.contourf(
                fluence_mesh, scat_mesh, p,
                levels=levels, vmin=0.0, vmax=1.0,
                cmap="viridis", extend="neither",
            )

            ax.set_xscale("log")
            ax.set_yscale("log")

            # make each axes box square
            try:
                ax.set_box_aspect(1)
            except Exception:
                ax.set_aspect("equal", adjustable="box")

            ax.set_xlabel("Fluence (Jy ms)", fontsize=22)
            ax.set_ylabel("Scattering timescale (ms)", fontsize=22)
            ax.tick_params(axis="both", which="both", labelsize=18)

            ax.xaxis.set_major_locator(x_locator)
            ax.yaxis.set_major_locator(y_locator)
            ax.xaxis.set_major_formatter(log_fmt)
            ax.yaxis.set_major_formatter(log_fmt)

            if r == 0:
                ax.set_title(f"DM = {dm} pc/cm$^3$", fontsize=22, pad=10)

        # row label on right
        ax_right = axes[r, -1]
        ax_right.annotate(
            f"Width = {w:g} ms",
            xy=(1.10, 0.5),
            xycoords="axes fraction",
            rotation=-90,
            va="center",
            ha="left",
            fontsize=22,
        )

    # --- Colorbar: drawn into reserved cax (no overlap possible) ---
    cbar = fig.colorbar(
        last_contour,
        cax=cax,
        orientation="horizontal",
        ticks=np.arange(0.0, 1.01, 0.1),
    )
    cbar.set_label("Selection probability", fontsize=24)
    cbar.ax.tick_params(labelsize=18)

    # Optional: make the colorbar axis a bit “cleaner”
    cax.yaxis.set_visible(False)

    # IMPORTANT: don't use bbox_inches="tight" here; it can re-pack axes unexpectedly
    fig.savefig(args.out, format="pdf")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
