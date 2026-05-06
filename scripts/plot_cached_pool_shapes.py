#!/usr/bin/env python3

import os
import itertools
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# USER PARAMS
# ============================================================

MODE = "cache"  # "cache" or "prior"

CACHE_DIR = "/data/user-data/kmcgregor/bootstrap_downturn/sbi_3param/cache_3param"
OUTDIR    = "/data/user-data/kmcgregor/bootstrap_downturn/sbi_3param"

OUT_PDF = os.path.join(OUTDIR, f"functional_forms_{MODE}_with_kde.pdf")

TAU_MIN_S = 1e-4
TAU_MAX_S = 0.30
Y_CRIT = 15.0

N_GRID = 2000
PLOT_EVERY = 1

# ---------- Prior/grid to plot if MODE = "prior" ----------
KAPPA1_MIN, KAPPA1_MAX = 4.0, 6.0
TAU_CRIT_MIN_S, TAU_CRIT_MAX_S = 5e-3, 3e-2
KAPPA2_MIN, KAPPA2_MAX = -5.0, 5.0

N_KAPPA1_GRID = 10
N_TAU_CRIT_GRID = 5
N_KAPPA2_GRID = 10

# ---------- KDE overlay ----------
OVERPLOT_KDE = True
KDE_PICKLE_PATH = "/data/user-data/kmcgregor/catalog2-outputs/kde_scat__gaussian.pkl"

# Normalize both model curves and KDE to this point
RESCALE_TO_ANCHOR = True
ANCHOR_X = 13.1e-3   # seconds
ANCHOR_Y = 0.5

KDE_COLOR = "k"
KDE_LINEWIDTH = 3.0
KDE_ALPHA = 1.0

# ============================================================

def parse_cached_filename(fname):
    base = os.path.basename(fname)

    if base.startswith("pool_"):
        base = base[len("pool_"):]
    if base.endswith(".npz"):
        base = base[:-len(".npz")]

    parts = base.split("_")
    if len(parts) != 3:
        raise ValueError(f"Could not parse filename: {fname}")

    def parse_part(part, prefix):
        if not part.startswith(prefix):
            raise ValueError(f"Expected prefix {prefix} in {part}")

        s = part[len(prefix):]

        if s.startswith("p"):
            sign = 1.0
            s = s[1:]
        elif s.startswith("m"):
            sign = -1.0
            s = s[1:]
        else:
            raise ValueError(f"Expected p/m sign in {part}")

        return sign * float(s.replace("d", "."))

    kappa_1 = parse_part(parts[0], "k1")
    tau_crit = parse_part(parts[1], "tc")
    kappa_2 = parse_part(parts[2], "k2")

    return kappa_1, tau_crit, kappa_2


class TwoSegmentSemiLogLine:
    """
    Density-per-dex model:

        g(u) = p(log10 tau),   u = log10(tau)

    with two straight-line segments joined at tau_crit.
    """

    def __init__(
        self,
        kappa_1,
        tau_crit,
        kappa_2,
        *,
        tau_min=TAU_MIN_S,
        tau_max=TAU_MAX_S,
        y_crit=Y_CRIT,
    ):
        self.kappa_1 = float(kappa_1)
        self.tau_crit = float(tau_crit)
        self.kappa_2 = float(kappa_2)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.y_crit = float(y_crit)

        if not (self.tau_min < self.tau_crit < self.tau_max):
            raise ValueError(
                f"Need tau_min < tau_crit < tau_max; "
                f"got {self.tau_min}, {self.tau_crit}, {self.tau_max}"
            )

        self.u_min = np.log10(self.tau_min)
        self.u_crit = np.log10(self.tau_crit)
        self.u_max = np.log10(self.tau_max)

        self.y_min = self.y_crit + self.kappa_1 * (self.u_min - self.u_crit)
        self.y_max = self.y_crit + self.kappa_2 * (self.u_max - self.u_crit)

        if self.y_min <= 0 or self.y_crit <= 0 or self.y_max <= 0:
            raise ValueError(
                f"Non-positive density: "
                f"y_min={self.y_min:.4g}, "
                f"y_crit={self.y_crit:.4g}, "
                f"y_max={self.y_max:.4g}"
            )

        self.left_mass = 0.5 * (self.y_min + self.y_crit) * (
            self.u_crit - self.u_min
        )
        self.right_mass = 0.5 * (self.y_crit + self.y_max) * (
            self.u_max - self.u_crit
        )
        self.total_mass = self.left_mass + self.right_mass

    def density_per_dex(self, tau):
        tau = np.asarray(tau, dtype=float)
        u = np.log10(tau)

        y = np.zeros_like(tau, dtype=float)
        in_support = (tau >= self.tau_min) & (tau <= self.tau_max)

        left = in_support & (tau <= self.tau_crit)
        right = in_support & (tau > self.tau_crit)

        y[left] = self.y_crit + self.kappa_1 * (u[left] - self.u_crit)
        y[right] = self.y_crit + self.kappa_2 * (u[right] - self.u_crit)

        return y / self.total_mass


def get_parameter_grid_from_cache():
    files = sorted(
        f for f in os.listdir(CACHE_DIR)
        if f.startswith("pool_") and f.endswith(".npz")
    )

    if len(files) == 0:
        raise RuntimeError(f"No cached pool files found in {CACHE_DIR}")

    return [parse_cached_filename(fname) for fname in files]


def get_parameter_grid_from_prior():
    k1_grid = np.linspace(KAPPA1_MIN, KAPPA1_MAX, N_KAPPA1_GRID)
    tc_grid = np.linspace(TAU_CRIT_MIN_S, TAU_CRIT_MAX_S, N_TAU_CRIT_GRID)
    k2_grid = np.linspace(KAPPA2_MIN, KAPPA2_MAX, N_KAPPA2_GRID)

    return list(itertools.product(k1_grid, tc_grid, k2_grid))


def load_saved_kde_curve(kde_pickle_path, xgrid):
    """
    Load Catalog2 KDE pickle storing bootstrap bands for x * pdf(x).

    Returns interpolated:
      y_med, y_lo, y_hi

    These are in x * pdf(x), matching the reference script's plotted quantity.
    """

    with open(kde_pickle_path, "rb") as f:
        payload = pickle.load(f)

    if "xgrid" not in payload:
        raise KeyError("KDE pickle missing 'xgrid'")

    x_saved = np.asarray(payload["xgrid"], dtype=float).reshape(-1)

    boot = payload.get("curve_x_times_pdf", {}).get("bootstrap", {})

    y_med = boot.get("median", None)
    y_lo  = boot.get("lo", None)
    y_hi  = boot.get("hi", None)

    if y_med is None:
        raise KeyError("KDE pickle missing curve_x_times_pdf/bootstrap/median")

    y_med = np.asarray(y_med, dtype=float).reshape(-1)

    if len(x_saved) != len(y_med):
        raise ValueError("Lengths of KDE xgrid and median curve do not match")

    y_interp = np.interp(xgrid, x_saved, y_med, left=np.nan, right=np.nan)

    y_lo_interp = None
    y_hi_interp = None

    if y_lo is not None and y_hi is not None:
        y_lo = np.asarray(y_lo, dtype=float).reshape(-1)
        y_hi = np.asarray(y_hi, dtype=float).reshape(-1)

        if not (len(x_saved) == len(y_lo) == len(y_hi)):
            raise ValueError("Lengths of KDE xgrid/lo/hi do not match")

        y_lo_interp = np.interp(xgrid, x_saved, y_lo, left=np.nan, right=np.nan)
        y_hi_interp = np.interp(xgrid, x_saved, y_hi, left=np.nan, right=np.nan)

    return y_interp, y_lo_interp, y_hi_interp, payload

def rescale_curve_at_x(xgrid, y, x0, y_target):
    y = np.asarray(y, dtype=float)
    y0 = np.interp(x0, xgrid, y)

    if not np.isfinite(y0) or y0 <= 0:
        return y.copy()

    return y * (y_target / y0)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    if MODE == "cache":
        params = get_parameter_grid_from_cache()
        title_source = "cached filenames"
    elif MODE == "prior":
        params = get_parameter_grid_from_prior()
        title_source = "user-defined prior grid"
    else:
        raise ValueError("MODE must be either 'cache' or 'prior'.")

    print(f"Plotting {len(params)} functional forms from {title_source}.")
    print(f"Normalization anchor: tau={ANCHOR_X:.4g} s, y={ANCHOR_Y:.3g}")

    xgrid = np.logspace(np.log10(TAU_MIN_S), np.log10(TAU_MAX_S), N_GRID)

    fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)

    n_plotted = 0
    failed = []

    for i, (k1, tc, k2) in enumerate(tqdm(params, desc="Plotting functional forms")):
        if i % PLOT_EVERY != 0:
            continue

        try:
            model = TwoSegmentSemiLogLine(
                kappa_1=k1,
                tau_crit=tc,
                kappa_2=k2,
                y_crit=Y_CRIT,
            )

            y = model.density_per_dex(xgrid)

            if RESCALE_TO_ANCHOR:
                y = rescale_curve_at_x(xgrid, y, ANCHOR_X, ANCHOR_Y)

            ax.plot(
                xgrid,
                y,
                color="C0",
                alpha=0.05,
                linewidth=1.0,
            )

            n_plotted += 1

        except Exception as e:
            failed.append((k1, tc, k2, str(e)))

    # ------------------------------------------------------------
    # KDE overlay
    # ------------------------------------------------------------
    if OVERPLOT_KDE:
        try:
            y_kde, y_kde_lo, y_kde_hi, payload = load_saved_kde_curve(
                KDE_PICKLE_PATH,
                xgrid,
            )

            # Only rescale the median (optional)
            if RESCALE_TO_ANCHOR:
                scale = ANCHOR_Y / np.interp(ANCHOR_X, xgrid, y_kde)

                y_kde = y_kde * scale

                if y_kde_lo is not None and y_kde_hi is not None:
                    y_kde_lo = y_kde_lo * scale
                    y_kde_hi = y_kde_hi * scale

            good = np.isfinite(y_kde) & (y_kde > 0)

            label = "Selection-corrected KDE"
            kernel = payload.get("kde_kernel", None)
            if kernel is not None:
                label += f" ({kernel})"

            if y_kde_lo is not None and y_kde_hi is not None:
                band_good = (
                    good
                    & np.isfinite(y_kde_lo)
                    & np.isfinite(y_kde_hi)
                    & (y_kde_lo > 0)
                    & (y_kde_hi > 0)
                )

                ax.fill_between(
                    xgrid[band_good],
                    y_kde_lo[band_good],
                    y_kde_hi[band_good],
                    color=KDE_COLOR,
                    alpha=0.15,
                    linewidth=0,
                    zorder=9,
                    label="KDE interval",
                )

            if np.any(good):
                ax.plot(
                    xgrid[good],
                    y_kde[good],
                    color=KDE_COLOR,
                    linewidth=KDE_LINEWIDTH,
                    alpha=KDE_ALPHA,
                    zorder=10,
                    label=label,
                )

                print(f"Overplotted KDE from: {KDE_PICKLE_PATH}")
            else:
                print("WARNING: KDE curve had no finite positive values to plot.")

        except Exception as e:
            print(f"WARNING: Could not overplot KDE: {e}")

    ax.scatter(
        [ANCHOR_X],
        [ANCHOR_Y],
        color="red",
        s=45,
        zorder=20,
        label=rf"anchor: $\tau={ANCHOR_X*1e3:.1f}$ ms, $y={ANCHOR_Y:.2f}$",
    )

    ax.axvline(
        ANCHOR_X,
        color="red",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
    )

    ax.set_xscale("log")
    ax.set_xlim(TAU_MIN_S, TAU_MAX_S)

    ax.set_xlabel(r"Scattering timescale $\tau_{600\,{\rm MHz}}$ (s)", fontsize=15)
    ax.set_ylabel(r"Density per dex / arbitrary normalization", fontsize=15)

    ax.set_title(
        rf"Functional forms from {title_source} "
        rf"($N={n_plotted}$, anchor at {ANCHOR_X*1e3:.1f} ms)",
        fontsize=13,
    )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=10, loc="best", framealpha=1.0)

    fig.savefig(OUT_PDF, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {OUT_PDF}")

    if failed:
        print(f"\nFailed to plot {len(failed)} parameter combinations:")
        for k1, tc, k2, err in failed[:20]:
            print(
                f"  k1={k1:.3f}, tc={tc*1e3:.2f} ms, "
                f"k2={k2:.3f}: {err}"
            )
        if len(failed) > 20:
            print("  ...")


if __name__ == "__main__":
    main()