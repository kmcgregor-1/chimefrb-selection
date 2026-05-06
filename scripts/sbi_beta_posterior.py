#!/usr/bin/env python3
"""
sbi_beta_posterior.py

Estimate a posterior p(beta | catalog) using Neural Posterior Estimation (NPE)
from the `sbi` package. Reuses cached model pools from build_downturn_cache.py.

Adds training-data/cache diagnostics:
  1. Weighted-pool effective sample size per cached beta
  2. Nearest-neighbor cache-grid distance for random prior beta draws
"""

import json
import os
from typing import Dict, Tuple, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from sbi import inference as sbi_inference
from sbi import utils as sbi_utils
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# USER PARAMS
# ============================================================

OUTDIR    = "/data/user-data/kmcgregor/bootstrap_downturn/sbi"
CACHE_DIR = "/data/user-data/kmcgregor/bootstrap_downturn/beta_hat_cache"

CAT2_JSON = "/data/user-data/ssiegel/catalog2/table/20251121/chimefrbcat2.json"

# Prior range on beta
BETA_MIN = -2.0
BETA_MAX =  2.0

# SBI training
N_SIMULATIONS = 100_000
N_RESAMPLE_MULT = 1.0
N_POOL_PER_BETA = 1_000_000
N_POSTERIOR_SAMPLES = 100_000
N_QUANTILES = 12

# Catalog cuts
SNR_CUT = 12.0
DM_LO   = 100.0
DM_HI   = 5000.0

# Diagnostics
RUN_DIAGNOSTICS = True
RUN_ESS_DIAGNOSTIC = True
RUN_GRID_DIAGNOSTIC = True
N_GRID_DIAG_DRAWS = 10_000

RNG_SEED = 99999


# ============================================================
# Catalog loading
# ============================================================

def to_float_or_nan(x):
    try:
        if isinstance(x, str):
            x = x.strip()
            if x.startswith("<"):
                x = x[1:]
        if isinstance(x, list) and len(x) > 0:
            return to_float_or_nan(x[0])
        return float(x)
    except Exception:
        return np.nan


def load_catalog_scattering(cat2_json, snr_cut, dm_lo, dm_hi):
    with open(cat2_json) as f:
        cat = json.load(f)

    cut = []
    for frb in cat:
        # Use bonsai_snr if available, otherwise fall back to snr_fitb.
        snr = to_float_or_nan(frb.get("bonsai_snr", frb.get("snr_fitb", np.nan)))
        dm  = to_float_or_nan(frb.get("dm_fitb", np.nan))

        if (
            np.isfinite(snr)
            and snr >= snr_cut
            and np.isfinite(dm)
            and dm_lo <= dm <= dm_hi
        ):
            cut.append(frb)

    if len(cut) == 0:
        raise RuntimeError("No catalog entries passed cuts.")

    scat_key = None
    for k in ["scat_time", "scattering_time", "tau"]:
        if k in cut[0]:
            scat_key = k
            break

    scat_s = (
        np.array([to_float_or_nan(frb.get(scat_key, np.nan)) for frb in cut], float)
        if scat_key
        else np.full(len(cut), np.nan)
    )

    # Heuristic unit fix: ms -> s if values look like ms.
    if np.nanmedian(scat_s) > 0.5:
        scat_s *= 1e-3

    # Catalog scattering assumed at 400 MHz.
    # Convert seconds at 400 MHz to milliseconds at 1000 MHz.
    tau_ms_1000 = scat_s * (400.0 / 1000.0) ** 4 * 1e3

    m = np.isfinite(tau_ms_1000) & (tau_ms_1000 > 0)

    if np.sum(m) < 10:
        raise RuntimeError(f"Too few valid scattering times after cuts: n={np.sum(m)}")

    return tau_ms_1000[m]


# ============================================================
# Cache loading
# ============================================================

def parse_beta_from_pool_filename(fname: str) -> float:
    """
    Parse beta from filenames like:
      pool_betam2p000.npz
      pool_betap0p500.npz
      pool_beta0p000.npz
    """
    tag = fname.replace("pool_beta", "").replace(".npz", "")

    if tag.startswith("m"):
        sign = -1.0
        tag = tag[1:]
    elif tag.startswith("p"):
        sign = 1.0
        tag = tag[1:]
    else:
        sign = 1.0

    tag = tag.replace("p", ".")
    return sign * float(tag)


def load_cached_pools(cache_dir: str) -> Dict[float, Tuple[np.ndarray, np.ndarray]]:
    pool_files = [
        f for f in os.listdir(cache_dir)
        if f.startswith("pool_") and f.endswith(".npz")
    ]

    if len(pool_files) == 0:
        raise RuntimeError(
            f"No cached pools found in {cache_dir}.\n"
            "Run build_downturn_cache.py first."
        )

    loaded_pools: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

    for fname in sorted(pool_files):
        pth = os.path.join(cache_dir, fname)
        dat = np.load(pth)

        x = dat["x_pool_ms_1000"].astype(float)
        w = dat["w_pool"].astype(float)

        beta_val = parse_beta_from_pool_filename(fname)
        loaded_pools[beta_val] = (x, w)

    return loaded_pools


# ============================================================
# Summary statistic and simulation
# ============================================================

def weighted_resample(
    rng: np.random.Generator,
    x_pool: np.ndarray,
    w_pool: np.ndarray,
    size: int,
) -> np.ndarray:
    """Draw `size` samples from x_pool with replacement, weighted by w_pool."""
    w = np.asarray(w_pool, dtype=float)
    w = np.clip(w, 0.0, None)

    total = w.sum()
    if total <= 0.0:
        raise ValueError("All pool weights are zero or negative.")

    p = w / total
    idx = rng.choice(len(x_pool), size=int(size), replace=True, p=p)
    return x_pool[idx]


def compute_summary(tau_ms: np.ndarray, n_quantiles: int = N_QUANTILES) -> np.ndarray:
    """
    Summary statistic:
      quantiles of log10(tau_ms) + mean + std.

    Returns length n_quantiles + 2.
    """
    x = np.log10(tau_ms[np.isfinite(tau_ms) & (tau_ms > 0)])

    if x.size < 5:
        return np.zeros(n_quantiles + 2, dtype=np.float32)

    probs = np.linspace(5, 95, n_quantiles)
    qs = np.percentile(x, probs)

    return np.concatenate([qs, [x.mean(), x.std()]]).astype(np.float32)


SUMMARY_DIM = N_QUANTILES + 2


def nearest_pool(
    beta: float,
    loaded_pools: Dict[float, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return nearest cached beta and corresponding pool."""
    cached_betas = np.array(list(loaded_pools.keys()), dtype=float)
    nearest_beta = float(cached_betas[np.argmin(np.abs(cached_betas - beta))])
    x_pool, w_pool = loaded_pools[nearest_beta]
    return nearest_beta, x_pool, w_pool


def simulate_summary(
    beta: float,
    rng: np.random.Generator,
    n_cat: int,
    n_resample_mult: float,
    loaded_pools: Dict[float, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    Draw a mock catalog by importance-resampling from the nearest cached beta pool.
    """
    _, x_pool, w_pool = nearest_pool(beta, loaded_pools)

    n_resamp = int(max(50, round(n_resample_mult * n_cat)))
    x_mock = weighted_resample(rng, x_pool, w_pool, size=n_resamp)

    return compute_summary(x_mock)


# ============================================================
# Diagnostics
# ============================================================

def effective_n(w: np.ndarray) -> float:
    """Effective sample size of a non-negative weighted pool."""
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]

    if w.size == 0:
        return 0.0

    denom = np.sum(w ** 2)
    if denom <= 0.0:
        return 0.0

    return float((np.sum(w) ** 2) / denom)


def diagnose_pools(
    loaded_pools: Dict[float, Tuple[np.ndarray, np.ndarray]],
    outdir: str,
) -> List[dict]:
    """
    Diagnostic 1:
    Compute weighted effective sample size for every cached beta pool.
    """
    print("\n" + "=" * 60)
    print("Diagnostic 1 – Weighted-pool effective sample size")
    print("=" * 60)

    rows: List[dict] = []

    for beta, (x, w) in sorted(loaded_pools.items()):
        n_raw = int(len(w))
        n_eff = effective_n(w)
        eff_frac = n_eff / max(n_raw, 1)

        w_arr = np.asarray(w, dtype=float)
        nonzero_frac = float(np.mean(w_arr > 0)) if n_raw > 0 else 0.0
        mean_w = float(np.nanmean(w_arr)) if n_raw > 0 else np.nan

        row = {
            "beta": float(beta),
            "n_raw": n_raw,
            "n_eff": float(n_eff),
            "eff_frac": float(eff_frac),
            "nonzero_weight_frac": nonzero_frac,
            "mean_weight": mean_w,
        }
        rows.append(row)

        print(
            f"beta={beta:+7.3f}  "
            f"N_eff={n_eff:10.0f}  ({100 * eff_frac:6.2f}%)  "
            f"nonzero={100 * nonzero_frac:6.2f}%  "
            f"mean_w={mean_w:.3g}"
        )

    if rows:
        n_eff_all = np.array([r["n_eff"] for r in rows], dtype=float)
        eff_frac_all = np.array([r["eff_frac"] for r in rows], dtype=float)

        print("\nESS summary:")
        print(f"  min N_eff:       {np.nanmin(n_eff_all):.0f}")
        print(f"  median N_eff:    {np.nanmedian(n_eff_all):.0f}")
        print(f"  min ESS frac:    {100 * np.nanmin(eff_frac_all):.2f}%")
        print(f"  median ESS frac: {100 * np.nanmedian(eff_frac_all):.2f}%")

        low = [r for r in rows if r["eff_frac"] < 0.05]
        if low:
            print(
                f"\nWARNING: {len(low)} cached beta pools have N_eff/N_raw < 5%. "
                "These beta values are candidates for increasing N_POOL_PER_BETA."
            )
        else:
            print("\nNo cached beta pools have N_eff/N_raw < 5%.")

    os.makedirs(outdir, exist_ok=True)

    out_json = os.path.join(outdir, "diagnostic_pool_ess.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote: {out_json}")

    # Plot ESS fraction versus beta.
    if rows:
        betas = np.array([r["beta"] for r in rows], dtype=float)
        eff_frac = np.array([r["eff_frac"] for r in rows], dtype=float)
        nonzero_frac = np.array([r["nonzero_weight_frac"] for r in rows], dtype=float)

        fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
        ax.plot(betas, eff_frac, marker="o", linewidth=1.8, label=r"$N_{\rm eff}/N_{\rm raw}$")
        ax.plot(betas, nonzero_frac, marker="s", linewidth=1.2, alpha=0.8, label="nonzero weight fraction")
        ax.axhline(0.05, linestyle=":", linewidth=1.5, label="5% heuristic")
        ax.set_xlabel(r"cached $\beta$")
        ax.set_ylabel("fraction")
        ax.set_title("Cached-pool weight diagnostics")
        ax.legend(fontsize=9)

        out_pdf = os.path.join(outdir, "diagnostic_pool_ess.pdf")
        fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote: {out_pdf}")

    return rows


def diagnose_grid_density(
    loaded_pools: Dict[float, Tuple[np.ndarray, np.ndarray]],
    outdir: str,
    rng_seed: int,
    beta_lo: float,
    beta_hi: float,
    n_draws: int = N_GRID_DIAG_DRAWS,
) -> dict:
    """
    Diagnostic 2:
    Draw random beta values from the effective prior and measure distance to
    nearest cached beta.

    In 1D, the normalized distance is:
      |beta - beta_nearest| / (beta_max_cache - beta_min_cache)
    """
    print("\n" + "=" * 60)
    print("Diagnostic 2 – Nearest-neighbor cache-grid distance")
    print("=" * 60)

    cached_betas = np.array(sorted(loaded_pools.keys()), dtype=float)

    if cached_betas.size == 0:
        raise RuntimeError("No cached betas available for grid-density diagnostic.")

    beta_range = float(cached_betas.max() - cached_betas.min())
    if beta_range <= 0:
        beta_range = 1.0

    rng = np.random.default_rng(rng_seed)

    dists = np.empty(int(n_draws), dtype=float)
    nearest_vals = np.empty(int(n_draws), dtype=float)
    draws = np.empty(int(n_draws), dtype=float)

    for i in range(int(n_draws)):
        beta = rng.uniform(beta_lo, beta_hi)
        nearest = cached_betas[np.argmin(np.abs(cached_betas - beta))]

        draws[i] = beta
        nearest_vals[i] = nearest
        dists[i] = abs(beta - nearest) / beta_range

    result = {
        "n_draws": int(n_draws),
        "beta_prior": [float(beta_lo), float(beta_hi)],
        "cached_beta_min": float(cached_betas.min()),
        "cached_beta_max": float(cached_betas.max()),
        "n_cached_betas": int(cached_betas.size),
        "median_nn_dist": float(np.median(dists)),
        "p95_nn_dist": float(np.percentile(dists, 95.0)),
        "max_nn_dist": float(np.max(dists)),
        "median_abs_beta_offset": float(np.median(np.abs(draws - nearest_vals))),
        "p95_abs_beta_offset": float(np.percentile(np.abs(draws - nearest_vals), 95.0)),
        "max_abs_beta_offset": float(np.max(np.abs(draws - nearest_vals))),
    }

    print(f"Cached beta count:       {result['n_cached_betas']}")
    print(f"Median NN dist:          {result['median_nn_dist']:.4f}")
    print(f"95th pct NN dist:        {result['p95_nn_dist']:.4f}")
    print(f"Max NN dist:             {result['max_nn_dist']:.4f}")
    print(f"Median |Δbeta|:          {result['median_abs_beta_offset']:.4f}")
    print(f"95th pct |Δbeta|:        {result['p95_abs_beta_offset']:.4f}")
    print(f"Max |Δbeta|:             {result['max_abs_beta_offset']:.4f}")

    if result["p95_nn_dist"] > 0.05:
        print(
            "WARNING: 95th-percentile normalized NN distance is > 0.05. "
            "If posterior checks look biased or jagged, increase beta grid density."
        )
    else:
        print("Grid spacing looks reasonable by the 95th-percentile < 0.05 heuristic.")

    os.makedirs(outdir, exist_ok=True)

    out_json = os.path.join(outdir, "diagnostic_grid_density.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote: {out_json}")

    fig, ax = plt.subplots(figsize=(5.8, 4.5), constrained_layout=True)
    ax.hist(dists, bins=50, histtype="step", linewidth=2)
    ax.axvline(result["median_nn_dist"], linestyle="--", linewidth=1.5, label="median")
    ax.axvline(result["p95_nn_dist"], linestyle=":", linewidth=2.0, label="95th pct")
    ax.set_xlabel("Nearest cached-β distance, normalized units")
    ax.set_ylabel("Count")
    ax.set_title("Cache-grid distance diagnostic")
    ax.legend(fontsize=9)

    out_pdf = os.path.join(outdir, "diagnostic_grid_density.pdf")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_pdf}")

    return result


# ============================================================
# Main NPE pipeline
# ============================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    # ------------------------------------------------------------
    # Load catalog
    # ------------------------------------------------------------
    print("Loading catalog...")
    x_cat = load_catalog_scattering(CAT2_JSON, SNR_CUT, DM_LO, DM_HI)
    n_cat = int(x_cat.size)

    print(f"  Catalog scattering times: n={n_cat}")

    x_obs = compute_summary(x_cat)
    print(f"  Observed summary length: {len(x_obs)}")
    print(f"  Observed summary: {x_obs}")

    # ------------------------------------------------------------
    # Load cached pools
    # ------------------------------------------------------------
    print("\nLoading cached pools...")
    loaded_pools = load_cached_pools(CACHE_DIR)

    cached_betas = np.array(sorted(loaded_pools.keys()), dtype=float)

    print(
        f"  Loaded {len(cached_betas)} cached pools, "
        f"beta ∈ [{cached_betas.min():.3f}, {cached_betas.max():.3f}]"
    )

    # Clamp prior to cached range with a small margin.
    beta_lo = max(BETA_MIN, float(cached_betas.min()) + 1e-6)
    beta_hi = min(BETA_MAX, float(cached_betas.max()) - 1e-6)

    if not beta_lo < beta_hi:
        raise RuntimeError(
            f"Invalid clamped prior range: [{beta_lo}, {beta_hi}]. "
            "Check BETA_MIN/BETA_MAX and cached beta grid."
        )

    print(f"  Prior range clamped to [{beta_lo:.3f}, {beta_hi:.3f}]")

    # ------------------------------------------------------------
    # Training-data diagnostics
    # ------------------------------------------------------------
    if RUN_DIAGNOSTICS:
        if RUN_ESS_DIAGNOSTIC:
            diagnose_pools(loaded_pools, OUTDIR)

        if RUN_GRID_DIAGNOSTIC:
            diagnose_grid_density(
                loaded_pools=loaded_pools,
                outdir=OUTDIR,
                rng_seed=RNG_SEED,
                beta_lo=beta_lo,
                beta_hi=beta_hi,
                n_draws=N_GRID_DIAG_DRAWS,
            )

    # ------------------------------------------------------------
    # Build training data
    # ------------------------------------------------------------
    print(f"\nGenerating {N_SIMULATIONS} (beta, summary) pairs for NPE training...")

    thetas = []
    xs = []

    for _ in tqdm(range(N_SIMULATIONS), desc="Simulating"):
        beta_i = rng.uniform(beta_lo, beta_hi)

        s_i = simulate_summary(
            beta=beta_i,
            rng=rng,
            n_cat=n_cat,
            n_resample_mult=N_RESAMPLE_MULT,
            loaded_pools=loaded_pools,
        )

        thetas.append([beta_i])
        xs.append(s_i)

    thetas_np = np.array(thetas, dtype=np.float32)
    xs_np = np.array(xs, dtype=np.float32)

    # Basic sanity checks on generated training pairs.
    bad_theta = ~np.isfinite(thetas_np).all(axis=1)
    bad_x = ~np.isfinite(xs_np).all(axis=1)
    bad = bad_theta | bad_x

    if np.any(bad):
        print(f"WARNING: removing {np.sum(bad)} training pairs with non-finite values.")
        thetas_np = thetas_np[~bad]
        xs_np = xs_np[~bad]

    if len(thetas_np) < 100:
        raise RuntimeError(f"Too few valid training pairs after filtering: {len(thetas_np)}")

    print(f"  Final training pairs: {len(thetas_np)}")
    print(f"  Summary dimension:    {xs_np.shape[1]}")

    thetas_t = torch.tensor(thetas_np, dtype=torch.float32)
    xs_t = torch.tensor(xs_np, dtype=torch.float32)

    # ------------------------------------------------------------
    # Set up NPE
    # ------------------------------------------------------------
    print("\nSetting up NPE...")

    prior = sbi_utils.BoxUniform(
        low=torch.tensor([beta_lo], dtype=torch.float32),
        high=torch.tensor([beta_hi], dtype=torch.float32),
    )

    inference = sbi_inference.SNPE(
        prior=prior,
        summary_writer=SummaryWriter(log_dir=os.path.join(OUTDIR, "sbi-logs")),
    )

    inference = inference.append_simulations(thetas_t, xs_t)

    print("Training neural posterior estimator...")
    density_estimator = inference.train(
        training_batch_size=256,
        learning_rate=5e-4,
        max_num_epochs=200,
        stop_after_epochs=20,
        show_train_summary=True,
    )

    posterior = inference.build_posterior(density_estimator)

    # ------------------------------------------------------------
    # Sample posterior at observed summary
    # ------------------------------------------------------------
    print(f"\nSampling {N_POSTERIOR_SAMPLES} posterior samples...")

    x_obs_t = torch.tensor(x_obs, dtype=torch.float32)
    posterior.set_default_x(x_obs_t)

    samples = posterior.sample((N_POSTERIOR_SAMPLES,), x=x_obs_t)
    samples_np = samples.numpy().flatten()

    # ------------------------------------------------------------
    # Summarize posterior
    # ------------------------------------------------------------
    b_med = float(np.median(samples_np))
    b_lo68, b_hi68 = np.percentile(samples_np, [16.0, 84.0])
    b_lo95, b_hi95 = np.percentile(samples_np, [2.5, 97.5])
    b_lo99, b_hi99 = np.percentile(samples_np, [0.5, 99.5])

    prior_width = beta_hi - beta_lo
    edge_frac = 0.05
    near_low = samples_np < beta_lo + edge_frac * prior_width
    near_high = samples_np > beta_hi - edge_frac * prior_width

    print("\nPosterior summary:")
    print(f"  median : {b_med:+.4f}")
    print(f"  68% CI : [{b_lo68:+.4f}, {b_hi68:+.4f}]")
    print(f"  95% CI : [{b_lo95:+.4f}, {b_hi95:+.4f}]")
    print(f"  99% CI : [{b_lo99:+.4f}, {b_hi99:+.4f}]")
    print(f"  mass near low prior edge:  {np.mean(near_low):.4f}")
    print(f"  mass near high prior edge: {np.mean(near_high):.4f}")

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    out_json = os.path.join(OUTDIR, "beta_posterior.json")

    result = {
        "params": {
            "n_simulations": int(N_SIMULATIONS),
            "n_resample_mult": float(N_RESAMPLE_MULT),
            "n_quantiles": int(N_QUANTILES),
            "summary_dim": int(SUMMARY_DIM),
            "beta_prior": [float(beta_lo), float(beta_hi)],
            "snr_cut": float(SNR_CUT),
            "dm_lo": float(DM_LO),
            "dm_hi": float(DM_HI),
            "n_cat": int(n_cat),
            "n_cached_betas": int(len(cached_betas)),
            "cached_beta_range": [float(cached_betas.min()), float(cached_betas.max())],
        },
        "posterior": {
            "mean": float(np.mean(samples_np)),
            "std": float(np.std(samples_np)),
            "median": float(b_med),
            "ci_16_84": [float(b_lo68), float(b_hi68)],
            "ci_2p5_97p5": [float(b_lo95), float(b_hi95)],
            "ci_0p5_99p5": [float(b_lo99), float(b_hi99)],
            "edge_frac_of_prior_width": float(edge_frac),
            "posterior_mass_near_low_edge": float(np.mean(near_low)),
            "posterior_mass_near_high_edge": float(np.mean(near_high)),
            "posterior_mass_beta_lt_0": float(np.mean(samples_np < 0.0)),
            "posterior_mass_beta_gt_0": float(np.mean(samples_np > 0.0)),
        },
    }

    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote: {out_json}")

    out_npy = os.path.join(OUTDIR, "beta_posterior_samples.npy")
    np.save(out_npy, samples_np)
    print(f"Wrote: {out_npy}")

    # ------------------------------------------------------------
    # Posterior plot
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)

    ax.hist(
        samples_np,
        bins=np.linspace(beta_lo, beta_hi, 100),
        density=True,
        histtype="step",
        linewidth=2,
        label="NPE posterior",
    )

    ax.axvline(
        b_med,
        linestyle="--",
        linewidth=2,
        label=f"median = {b_med:+.3f}",
    )

    ax.axvspan(
        b_lo68,
        b_hi68,
        alpha=0.2,
        label=f"68% CI [{b_lo68:+.3f}, {b_hi68:+.3f}]",
    )

    ax.axvspan(
        b_lo95,
        b_hi95,
        alpha=0.10,
        label=f"95% CI [{b_lo95:+.3f}, {b_hi95:+.3f}]",
    )

    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("posterior density")
    ax.set_title(r"NPE posterior $p(\beta \mid \mathrm{catalog})$")
    ax.set_xlim(beta_lo, beta_hi)
    ax.legend(fontsize=9)

    out_png = os.path.join(OUTDIR, "beta_posterior.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out_png}")
    print("\nDone.")


if __name__ == "__main__":
    main()