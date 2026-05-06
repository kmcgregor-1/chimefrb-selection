import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from matplotlib.colors import Normalize
from tqdm import tqdm

import chimefrb_selection as cfsf


def build_selection_function(order=3, marginalized=True):
    """
    Load the 2D marginalized selection function for scattering_time and DM.
    """
    return cfsf.SelectionFunction(
        predictor_names=["fluence", "dm"],
        degree=order,  # Adjust based on your best-fit order
        snr_cut=12.0,
        exclude_sidelobes=True,
        sidelobe_cut=5.0,
        reweighted=marginalized # marginalization
    )

def compute_selection_grid(sf, fluence_mesh, dm_mesh):
    """
    Compute selection probability over a 2D grid of fluence and DM.
    """
    p = np.full_like(fluence_mesh, np.nan, dtype=float)
    n_y, n_x = fluence_mesh.shape
    
    for i in tqdm(range(n_y), desc="Computing p_det", leave=True):
        for j in range(n_x):
            try:
                p[i, j] = sf.calculate_selection_probability(
                    {
                        "fluence": float(fluence_mesh[i, j]),
                        "dm": float(dm_mesh[i, j]),
                    }
                )
            except (ValueError, KeyError):
                # Outside KNN envelope or other error
                p[i, j] = np.nan
    return p


def main():
    parser = argparse.ArgumentParser(
        description="Plot 2D marginalized selection function (fluence vs DM)"
    )
    parser.add_argument("--out", default="fluence_dm_selection_marg.pdf",
                        help="Output filename")
    parser.add_argument("--n", type=int, default=150,
                        help="Grid resolution per axis")
    parser.add_argument("--order", type=int, default=5,
                        help="Polynomial order of the selection function")
    args = parser.parse_args()

    # Define grid ranges
    fluence_grid = np.geomspace(1e-1, 2e3, args.n)
    dm_grid = np.geomspace(100, 5000, args.n)
    
    fluence_mesh, dm_mesh = np.meshgrid(fluence_grid, dm_grid)

    # Load selection function
    print(f"Loading 2D selection function (order={args.order})...")
    sf = build_selection_function(order=args.order, marginalized=True)
    
    print(f"Model loaded from: {sf.npz_path()}")
    print(f"KNN mask from: {sf.knn_path()}")

    # Compute selection probability grid
    print(f"Computing selection probability on {args.n}x{args.n} grid...")
    p = compute_selection_grid(sf, fluence_mesh, dm_mesh)
    
    # Count valid points
    n_valid = np.sum(~np.isnan(p))
    n_total = p.size
    print(f"Valid grid points: {n_valid}/{n_total} ({100*n_valid/n_total:.1f}%)")

    # --- Create figure ---
    fig, ax = plt.subplots(figsize=(12, 9))

    # Contour levels - limited to 0.5 max
    levels = np.arange(0.0, 1.01, 0.05)
    
    # Filled contours
    cf = ax.contourf(
        fluence_mesh, dm_mesh, p,
        levels=levels,
        vmin=0.0, vmax=1.0,
        cmap="viridis",
        extend="max",  # Show values > 0.5 as the max color
    )
    
    # Add contour lines for key probabilities
    contour_lines = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cs = ax.contour(
        fluence_mesh, dm_mesh, p,
        levels=contour_lines,
        colors='white',
        linewidths=0.8,
        linestyles='--',
        alpha=0.7,
    )
    ax.clabel(cs, inline=True, fontsize=10, fmt='%.2f')

    # Log scales
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Axis labels
    ax.set_xlabel(r"Fluence (Jy ms)", fontsize=18)
    ax.set_ylabel(r"Dispersion Measure (pc cm$^{-3}$)", fontsize=18)

    # Tick formatting
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.tick_params(axis="both", which="both", labelsize=12)

    # Grid
    ax.grid(True, which="major", linestyle="-", alpha=0.3, color="white")
    ax.grid(True, which="minor", linestyle=":", alpha=0.2, color="white")

    # Colorbar - limited to 0.5
    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, shrink=0.9)
    cbar.set_label("Selection Probability", fontsize=18)
    cbar.set_ticks(np.arange(0.0, 1.05, 0.05))
    cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    
    # Save
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()