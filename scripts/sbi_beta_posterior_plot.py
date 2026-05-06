#!/usr/bin/env python3

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

### NOTE: Beta has been renamed to kappa on the plots

# ============================================================
# USER PARAMS
# ============================================================

INDIR     = "/data/user-data/kmcgregor/bootstrap_downturn/sbi"
JSON_PATH = os.path.join(INDIR, "beta_posterior.json")
NPY_PATH  = os.path.join(INDIR, "beta_posterior_samples.npy")

OUT_PNG            = os.path.join(INDIR, "beta_posterior.pdf")
OUT_PNG_BETA_SHAPE = os.path.join(INDIR, "beta_shape_reference.pdf")
OUT_PNG_POST_SHAPE = os.path.join(INDIR, "beta_shape_posterior_draws.pdf")

# for the shape plots
TAU_MIN = 1e-5
TAU_MAX = 1.0
SIGMA   = 1.94
SCALE   = 0.0131

BETAS_REF = [-0.10, 0.0, 0.10]

N_POST_CURVES = 2_000
POST_RNG_SEED = 12345

# normalization point: below the pivot so left sides line up
NORM_X = SCALE / 3.0

# analytic grid for smooth curves
N_GRID = 2000

# ============================================================


class LognormalSemiLogTail:
    def __init__(self, beta=0.0, *, sigma=1.94, scale=0.0131,
                 tau_min=1e-5, tau_max=1.0):
        self.tail_k  = float(beta)
        self.sigma   = float(sigma)
        self.scale   = float(scale)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self._ln     = lognorm(s=self.sigma, scale=self.scale)
        self.pivot   = float(np.clip(self.scale, self.tau_min, self.tau_max))
        self._w_left = self._compute_w_left()

    def _compute_w_left(self):
        c_lo = self._ln.cdf(self.tau_min)
        c_p  = self._ln.cdf(self.pivot)
        left_mass = float(np.clip(c_p - c_lo, 0.0, 1.0))
        f_p = float(self._ln.pdf(self.pivot))
        A   = (self.pivot ** (1.0 - self.tail_k)) * f_p
        if self.tau_max <= self.pivot * (1.0 + 1e-15):
            right_mass = 0.0
        elif abs(self.tail_k) < 1e-12:
            right_mass = A * np.log(self.tau_max / self.pivot)
        else:
            right_mass = (A / self.tail_k) * (
                self.tau_max**self.tail_k - self.pivot**self.tail_k)
            right_mass = float(max(right_mass, 0.0))
        total = left_mass + right_mass
        return left_mass / total

    def rvs(self, size=1, random_state=None):
        n   = int(size)
        rng = np.random.default_rng(random_state)
        u   = rng.uniform(0.0, 1.0, size=n)
        left    = u < self._w_left
        n_left  = int(np.sum(left))
        n_right = n - n_left
        out = np.empty(n, dtype=float)

        if n_left > 0:
            c_lo = self._ln.cdf(self.tau_min)
            c_p  = self._ln.cdf(self.pivot)
            if c_p <= c_lo:
                out[left] = self.tau_min
            else:
                u_left    = rng.uniform(c_lo, c_p, size=n_left)
                out[left] = np.clip(self._ln.ppf(u_left), self.tau_min, self.pivot)

        if n_right > 0:
            u_right = rng.uniform(0.0, 1.0, size=n_right)
            if self.tau_max <= self.pivot * (1.0 + 1e-15):
                out[~left] = self.pivot
            elif abs(self.tail_k) < 1e-12:
                out[~left] = self.pivot * (self.tau_max / self.pivot) ** u_right
            else:
                xk = (self.pivot**self.tail_k
                      + u_right * (self.tau_max**self.tail_k - self.pivot**self.tail_k))
                out[~left] = np.power(xk, 1.0 / self.tail_k)

        return np.clip(out, self.tau_min, self.tau_max)


def analytic_pdf(model, x):
    """
    Analytic linear-x PDF implied by the class construction.

    On [tau_min, pivot], this is just the original lognormal PDF.
    On (pivot, tau_max], this is A * x^(beta-1), where A is chosen so
    the tail joins continuously at the pivot.
    """
    x = np.asarray(x, dtype=float)
    pdf = np.zeros_like(x)

    in_support = (x >= model.tau_min) & (x <= model.tau_max)
    if not np.any(in_support):
        return pdf

    left = in_support & (x <= model.pivot)
    right = in_support & (x > model.pivot)

    if np.any(left):
        pdf[left] = model._ln.pdf(x[left])

    if np.any(right):
        f_p = float(model._ln.pdf(model.pivot))
        A = (model.pivot ** (1.0 - model.tail_k)) * f_p
        pdf[right] = A * np.power(x[right], model.tail_k - 1.0)

    return pdf


def density_per_dex_from_pdf(x, pdf):
    """
    Convert linear-x PDF f(x) into density per dex:
        p(log10 x) = x ln(10) f(x)
    """
    return x * np.log(10.0) * pdf


def rescale_at_x(xgrid, ygrid, x0, y_target):
    """
    Multiply ygrid by a constant so that y(x0) = y_target.
    """
    y0 = np.interp(x0, xgrid, ygrid)
    if y0 <= 0 or not np.isfinite(y0):
        return ygrid.copy()
    return ygrid * (y_target / y0)


def get_shape_curve(beta, xgrid):
    model = LognormalSemiLogTail(
        beta=beta,
        sigma=SIGMA,
        scale=SCALE,
        tau_min=TAU_MIN,
        tau_max=TAU_MAX,
    )
    pdf = analytic_pdf(model, xgrid)
    yc = density_per_dex_from_pdf(xgrid, pdf)
    return xgrid, yc, model.pivot


def main():
    # ------------------------------------------------------------
    # Load posterior
    # ------------------------------------------------------------
    with open(JSON_PATH) as f:
        meta = json.load(f)
    samples = np.load(NPY_PATH)

    beta_lo, beta_hi = meta["params"]["beta_prior"]
    b_med            = meta["posterior"]["median"]
    b_lo68, b_hi68   = meta["posterior"]["ci_16_84"]
    b_lo95, b_hi95   = meta["posterior"]["ci_2p5_97p5"]

    # ------------------------------------------------------------
    # Posterior histogram
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5,5), constrained_layout=True)

    ax.hist(
        samples,
        bins=np.linspace(beta_lo, beta_hi, 201),
        density=True,
        histtype="step",
        linewidth=3,
    )

    ax.axvline(
        b_med, linestyle="--", linewidth=3,
        label=f"Median = {b_med:+.3f}"
    )

    ax.axvspan(
        b_lo68, b_hi68, alpha=0.2,
        label=f"68% CI [{b_lo68:+.3f}, {b_hi68:+.3f}]"
    )

    ax.axvspan(
        b_lo95, b_hi95, alpha=0.10,
        label=f"95% CI [{b_lo95:+.3f}, {b_hi95:+.3f}]"
    )

    ax.set_xlabel(r"$\kappa$", fontsize=18)
    ax.set_ylabel("Posterior density", fontsize=18)
    ax.set_xticks(np.arange(-1, 1.001, 0.25))
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(-1, 1)
    #ax.legend(loc='upper left', fontsize=12)

    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {OUT_PNG}")

    # ------------------------------------------------------------
    # Analytic smooth curves
    # ------------------------------------------------------------
    xgrid = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), N_GRID)

    fig, ax = plt.subplots(figsize=(5.5,5), constrained_layout=True)

    x0, y0, pivot = get_shape_curve(beta=0.0, xgrid=xgrid)
    y_target = np.interp(NORM_X, x0, y0)
    # manual annotation positions
    label_positions = {
        -0.10: (7e-2, 0.30),
        0.00: (3e-1, 0.42),
        0.10: (8e-2, 0.65),
    }

    for i, beta in enumerate(BETAS_REF):

        xc, yc, pivot = get_shape_curve(beta=beta, xgrid=xgrid)
        yc = rescale_at_x(xc, yc, NORM_X, y_target)

        left  = xc <= pivot
        right = xc > pivot

        color = f"C{i}"

        ax.plot(xc[left], yc[left], color=color, linewidth=4)
        ax.plot(xc[right], yc[right], color=color, linewidth=4, linestyle="--")

        # manual label placement
        x_ann, y_ann = label_positions[beta]

        ax.text(
            x_ann,
            y_ann,
            rf"$\kappa = {beta:+.2f}$",
            color=color,
            fontsize=14,
            ha="center",
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Scattering timescale $\tau_{600 {\rm MHz}}$ (s)", fontsize=18)
    ax.set_ylabel("Arbitrary normalization", fontsize=18)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(1e-4, TAU_MAX)
    ax.set_ylim(0, 0.8)

    fig.savefig(OUT_PNG_BETA_SHAPE, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {OUT_PNG_BETA_SHAPE}")

    # ------------------------------------------------------------
    # Posterior beta draws
    # ------------------------------------------------------------
    rng = np.random.default_rng(POST_RNG_SEED)
    n_show = min(N_POST_CURVES, len(samples))
    beta_draws = rng.choice(samples, size=n_show, replace=False)

    fig, ax = plt.subplots(figsize=(5.5,5), constrained_layout=True)

    xref, yref, pivot = get_shape_curve(beta=0.0, xgrid=xgrid)
    y_target = np.interp(NORM_X, xref, yref)

    for beta in beta_draws:

        xc, yc, pivot = get_shape_curve(beta=float(beta), xgrid=xgrid)
        yc = rescale_at_x(xc, yc, NORM_X, y_target)

        ax.plot(
            xc,
            yc,
            linewidth=4,
            alpha=0.01,
            color="C0"
        )

    ax.set_xscale("log")
    ax.set_xlabel(r"Scattering timescale $\tau_{600 {\rm MHz}}$ (s)", fontsize=18)
    ax.set_ylabel("Arbitrary normalization", fontsize=18)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(1e-4, TAU_MAX)
    ax.set_ylim(0, 0.8)

    fig.savefig(OUT_PNG_POST_SHAPE, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {OUT_PNG_POST_SHAPE}")


if __name__ == "__main__":
    main()