#!/usr/bin/env python3
"""
sbi_three_param_pipeline.py

Full pipeline for inferring the intrinsic scattering-time distribution using
Simulation-Based Inference (NPE) with a three-parameter model:

  Parameters
  ----------
  kappa_1   : float
      Log-slope of the distribution *below* tau_crit.
      In density-per-dex space (log10 tau on x, tau*p(tau) on y),
      a positive kappa_1 is a rising segment, negative is falling.
      Prior: Uniform[0.5, 2.0]

  tau_crit  : float  (seconds, at 600 MHz)
      Break point separating the two power-law segments.
      Prior: Uniform[0.005, 0.015]  (5–15 ms)

  kappa_2   : float
      Log-slope of the distribution *above* tau_crit.
      Prior: Uniform[-1.0, 1.0]

  Model PDF (on linear tau, at 600 MHz)
  --------------------------------------
  Below tau_crit (and above tau_min):
      f(tau) ∝ tau^(kappa_1 - 1)           [power law, rising in dex space]

  Above tau_crit (and below tau_max):
      f(tau) ∝ tau^(kappa_2 - 1)           [second power law joined continuously]

  The two segments are joined at tau_crit with a continuous (but not smooth)
  normalisation, so that the overall PDF integrates to 1 over [tau_min, tau_max].

Pipeline stages (all run by default; set RUN_* flags to skip stages)
----------------------------------------------------------------------
  Stage 1 – Build / load cache
      For a grid of (kappa_1, tau_crit, kappa_2) parameter tuples, draw a large
      pool of scattering times, compute p_det weights via the CHIME-FRB selection
      function, and save (x_pool_ms_1000, w_pool) per tuple to disk.

  Stage 2 – Train NPE
      Draw random parameter samples from the prior, retrieve the nearest cached
      pool for each, importance-resample a mock catalog, compute summary
      statistics, and train a Neural Posterior Estimator (normalising flow via the
      `sbi` package).

  Stage 3 – Plot results
      (a) Corner plot of the 2-D marginal posteriors for all three parameters.
      (b) Ensemble of distribution curves sampled from the posterior.

Usage
-----
  Edit the USER PARAMS block below, then:
      python sbi_three_param_pipeline.py

Dependencies
------------
  pip install sbi torch scipy matplotlib tqdm corner
  (plus the internal chimefrb_selection package)
"""

import json
import os
import itertools
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import lognorm
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from sbi import inference as sbi_inference
from sbi import utils as sbi_utils

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False

import chimefrb_selection as cfsf


# ============================================================
# USER PARAMS
# ============================================================

OUTDIR      = "/data/user-data/kmcgregor/bootstrap_downturn/sbi_3param"
CACHE_DIR   = os.path.join(OUTDIR, "cache_3param")

CAT2_JSON   = "/data/user-data/ssiegel/catalog2/table/20251121/chimefrbcat2.json"

# ---------- Prior bounds ----------
KAPPA1_MIN, KAPPA1_MAX = 4.0, 6.0
TAU_CRIT_MIN_S, TAU_CRIT_MAX_S = 5e-3, 3e-2
KAPPA2_MIN, KAPPA2_MAX = -5.0, 5.0

# ---------- Model support ----------
TAU_MIN_S = 1e-4
TAU_MAX_S = 0.30

Y_CRIT = 15.0

# ---------- Cache grid ----------
N_KAPPA1_GRID = 20
N_TAU_CRIT_GRID = 10
N_KAPPA2_GRID = 20
N_POOL_PER_CELL = 100_000

# ---------- SBI training ----------
N_SIMULATIONS   = 80_000    # (theta, summary) pairs for NPE training
N_RESAMPLE_MULT = 1.0       # mock catalog size = N_RESAMPLE_MULT * n_cat
N_QUANTILES     = 12        # quantile levels for summary statistic
N_POSTERIOR_SAMPLES = 500_000

# ---------- Plotting ----------
N_POST_CURVES   = 500       # distribution curves drawn from posterior
POST_RNG_SEED   = 12345

# normalization point for shape curves (below tau_crit so all curves share
# the same anchor value)
NORM_X_S = 13.1e-3
NORM_Y   = 0.5

Y_CRIT = 15.0

# ---------- Pipeline control ----------
RUN_CACHE  = True
RUN_SBI    = True
RUN_PLOTS  = True

# ---------- Catalog cuts ----------
SNR_CUT = 12.0
DM_LO, DM_HI = 100.0, 5000.0

RNG_SEED = 12346

# ============================================================


# ────────────────────────────────────────────────────────────
# Section 1  –  Two-segment power-law distribution
# ────────────────────────────────────────────────────────────

class TwoSegmentSemiLogLine:
    """
    Two-segment intrinsic scattering distribution where the density per dex,

        g(u) = p(log10 tau),   u = log10(tau),

    is piecewise linear in u.

    This is the model that appears as two straight lines when plotting
    density per dex versus tau with ax.set_xscale("log") and linear y-axis.

    Parameters
    ----------
    kappa_1 : float
        Slope of density-per-dex below tau_crit, per dex in tau.
        Positive means g rises toward tau_crit from the left.

    tau_crit : float
        Break point in seconds at 600 MHz.

    kappa_2 : float
        Slope of density-per-dex above tau_crit, per dex in tau.
        Negative means a downturn to the right.

    y_crit : float
        Arbitrary unnormalized density-per-dex at tau_crit.
        Only ratios matter because the model is normalized internally.
    """

    def __init__(
        self,
        kappa_1: float,
        tau_crit: float,
        kappa_2: float,
        *,
        tau_min: float = TAU_MIN_S,
        tau_max: float = TAU_MAX_S,
        y_crit: float = 1.0,
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

        # endpoint values of density per dex
        self.y_min = self.y_crit + self.kappa_1 * (self.u_min - self.u_crit)
        self.y_max = self.y_crit + self.kappa_2 * (self.u_max - self.u_crit)

        if self.y_min <= 0 or self.y_crit <= 0 or self.y_max <= 0:
            raise ValueError(
                "Piecewise-linear density became non-positive. "
                f"Got y_min={self.y_min:.3g}, y_crit={self.y_crit:.3g}, "
                f"y_max={self.y_max:.3g}. Adjust slope priors or y_crit."
            )

        # masses in u = log10(tau); trapezoid areas
        self.left_mass = 0.5 * (self.y_min + self.y_crit) * (self.u_crit - self.u_min)
        self.right_mass = 0.5 * (self.y_crit + self.y_max) * (self.u_max - self.u_crit)

        self.total_mass = self.left_mass + self.right_mass
        self.w_left = self.left_mass / self.total_mass

    def density_per_dex(self, tau: np.ndarray) -> np.ndarray:
        """
        Normalized density per dex, g(log10 tau).
        This is the thing to plot on the y-axis.
        """
        tau = np.asarray(tau, dtype=float)
        u = np.log10(tau)

        y = np.zeros_like(tau, dtype=float)
        in_support = (tau >= self.tau_min) & (tau <= self.tau_max)

        left = in_support & (tau <= self.tau_crit)
        right = in_support & (tau > self.tau_crit)

        y[left] = self.y_crit + self.kappa_1 * (u[left] - self.u_crit)
        y[right] = self.y_crit + self.kappa_2 * (u[right] - self.u_crit)

        return y / self.total_mass

    def pdf(self, tau: np.ndarray) -> np.ndarray:
        """
        Linear-tau PDF f(tau).

        Since g(log10 tau) = tau ln(10) f(tau),
        f(tau) = g(log10 tau) / [tau ln(10)].
        """
        tau = np.asarray(tau, dtype=float)
        g = self.density_per_dex(tau)

        out = np.zeros_like(tau, dtype=float)
        m = tau > 0
        out[m] = g[m] / (tau[m] * np.log(10.0))
        return out

    def _sample_linear_density_in_u(self, rng, u0, u1, y0, slope, n):
        """
        Sample u from density y(u) = y0 + slope * (u - u0)
        over [u0, u1].
        """
        if n <= 0:
            return np.empty(0, dtype=float)

        L = u1 - u0
        area = y0 * L + 0.5 * slope * L**2

        r = rng.uniform(0.0, area, size=n)

        if abs(slope) < 1e-14:
            z = r / y0
        else:
            # solve 0.5*slope*z^2 + y0*z - r = 0
            disc = y0**2 + 2.0 * slope * r
            z = (-y0 + np.sqrt(disc)) / slope

        return u0 + z

    def rvs(self, size: int = 1, random_state=None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        n = int(size)

        choose_left = rng.uniform(size=n) < self.w_left
        n_left = int(np.sum(choose_left))
        n_right = n - n_left

        out_u = np.empty(n, dtype=float)

        # Left segment: from u_min to u_crit.
        # y(u) = y_min + kappa_1 * (u - u_min)
        if n_left > 0:
            out_u[choose_left] = self._sample_linear_density_in_u(
                rng,
                self.u_min,
                self.u_crit,
                self.y_min,
                self.kappa_1,
                n_left,
            )

        # Right segment: from u_crit to u_max.
        # y(u) = y_crit + kappa_2 * (u - u_crit)
        if n_right > 0:
            out_u[~choose_left] = self._sample_linear_density_in_u(
                rng,
                self.u_crit,
                self.u_max,
                self.y_crit,
                self.kappa_2,
                n_right,
            )

        return np.clip(10.0**out_u, self.tau_min, self.tau_max)

def density_per_dex(tau: np.ndarray, model: TwoSegmentSemiLogLine, xgrid: np.ndarray) -> np.ndarray:
    return model.density_per_dex(xgrid)


# ────────────────────────────────────────────────────────────
# Section 2  –  Selection function & fiducial samplers
# ────────────────────────────────────────────────────────────

def build_selection_function():
    return cfsf.SelectionFunction(
        predictor_names=["fluence", "scattering_time", "width", "dm"],
        degree=3,
        snr_cut=SNR_CUT,
        exclude_sidelobes=True,
        sidelobe_cut=5.0,
        reweighted=False,
    )


def draw_fluence_powerlaw(N, rng, alpha=-1.203108427072943, fmin=0.1, fmax=10000.0):
    s_max = (fmax / fmin) ** alpha
    u     = rng.uniform(s_max, 1.0, size=N)
    return fmin * (u ** (1.0 / alpha))


def draw_lognorm_truncated(N, rng, shape, scale, lo, hi):
    dist  = lognorm(s=shape, scale=scale)
    c_lo  = dist.cdf(lo)
    c_hi  = dist.cdf(hi)
    u     = rng.uniform(c_lo, c_hi, size=N)
    return np.clip(dist.ppf(u), lo, hi)


def sample_fiducial(N: int, rng: np.random.Generator,
                    kappa_1: float, tau_crit: float, kappa_2: float):
    flu = draw_fluence_powerlaw(N, rng)
    dm  = draw_lognorm_truncated(N, rng, 0.6076516890745998, 534.4727066208081, DM_LO, DM_HI)
    wid = draw_lognorm_truncated(N, rng, 1.1012853240184415, 7.389903164298552e-4, 1e-4, 0.2)

    model   = TwoSegmentSemiLogLine(kappa_1, tau_crit, kappa_2, y_crit=Y_CRIT)
    tau_600 = model.rvs(size=N, random_state=rng)

    mask = np.isfinite(tau_600) & (tau_600 >= 1e-4)
    return (flu[mask], tau_600[mask],
            tau_600[mask] * (600.0 / 1000.0) ** 4,
            wid[mask], dm[mask])


def compute_pdet(sf, flu, tau_1000_s, wid_s, dm):
    props = {
        "fluence_jy_ms":  np.asarray(flu,        float),
        "tau_1_ghz_ms":   np.asarray(tau_1000_s, float) * 1e3,
        "pulse_width_ms": np.asarray(wid_s,      float) * 1e3,
        "dm":             np.asarray(dm,          float),
    }
    try:
        pdet, _ = sf.calculate_selection_probability(props, return_std=True)
    except TypeError:
        pdet = sf.calculate_selection_probability(props)

    pdet  = np.asarray(pdet, float)
    valid = np.isfinite(pdet)
    pdet[valid] = np.clip(pdet[valid], 0.0, 1.0)
    return pdet, valid


# ────────────────────────────────────────────────────────────
# Section 3  –  Catalog loading
# ────────────────────────────────────────────────────────────

def _to_float(x):
    try:
        if isinstance(x, str):
            x = x.strip().lstrip("<")
        if isinstance(x, list) and x:
            return _to_float(x[0])
        return float(x)
    except Exception:
        return np.nan


def load_catalog_scattering(cat2_json: str) -> np.ndarray:
    with open(cat2_json) as f:
        cat = json.load(f)

    cut = [
        frb for frb in cat
        if (np.isfinite(_to_float(frb.get("bonsai_snr", np.nan))) and
            _to_float(frb.get("bonsai_snr", np.nan)) >= SNR_CUT and
            np.isfinite(_to_float(frb.get("dm_fitb", np.nan))) and
            DM_LO <= _to_float(frb.get("dm_fitb", np.nan)) <= DM_HI)
    ]
    if not cut:
        raise RuntimeError("No catalog entries survived the cuts.")

    scat_key = next(
        (k for k in ["scat_time", "scattering_time", "tau"] if k in cut[0]), None
    )
    scat_s = np.array(
        [_to_float(frb.get(scat_key, np.nan)) for frb in cut], float
    ) if scat_key else np.full(len(cut), np.nan)

    # heuristic unit fix (ms → s if median > 0.5)
    if np.nanmedian(scat_s) > 0.5:
        scat_s *= 1e-3

    # scale from 400 MHz to 1000 MHz and convert to ms
    tau_ms = scat_s * (400.0 / 1000.0) ** 4 * 1e3
    m = np.isfinite(tau_ms) & (tau_ms > 0)
    if m.sum() < 10:
        raise RuntimeError(f"Too few valid scattering times: n={m.sum()}")
    return tau_ms[m]


# ────────────────────────────────────────────────────────────
# Section 4  –  Cache building
# ────────────────────────────────────────────────────────────

def _cell_tag(kappa_1: float, tau_crit: float, kappa_2: float) -> str:
    def fmt(v: float) -> str:
        return f"{v:+.4f}".replace("+", "p").replace("-", "m").replace(".", "d")
    return f"k1{fmt(kappa_1)}_tc{fmt(tau_crit)}_k2{fmt(kappa_2)}"


def cache_path(cache_dir: str, kappa_1: float, tau_crit: float, kappa_2: float) -> str:
    return os.path.join(cache_dir, f"pool_{_cell_tag(kappa_1, tau_crit, kappa_2)}.npz")


def build_or_load_cell(
    *,
    sf,
    rng: np.random.Generator,
    kappa_1: float,
    tau_crit: float,
    kappa_2: float,
    n_pool: int,
    cache_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_pool_ms_1000, w_pool), building from simulation if not cached."""
    import os
    import numpy as np
    from tqdm import tqdm

    os.makedirs(cache_dir, exist_ok=True)
    pth = cache_path(cache_dir, kappa_1, tau_crit, kappa_2)

    # --- Load if cached ---
    if os.path.exists(pth):
        print(
            f"[cache hit] k1={kappa_1:.2f}, "
            f"tc={tau_crit*1e3:.1f} ms, k2={kappa_2:.2f}"
        )
        dat = np.load(pth)
        return (
            dat["x_pool_ms_1000"].astype(float),
            dat["w_pool"].astype(float),
        )

    # --- Otherwise build ---
    print(
        f"[building] k1={kappa_1:.2f}, "
        f"tc={tau_crit*1e3:.1f} ms, k2={kappa_2:.2f}"
    )

    x_acc, w_acc = [], []
    chunk = min(1_000, n_pool)

    n_have = 0

    pbar = tqdm(
        total=n_pool,
        desc=f"k1={kappa_1:.2f}, tc={tau_crit*1e3:.1f} ms, k2={kappa_2:.2f}",
        leave=False,
    )

    while n_have < n_pool:
        flu, tau_600, tau_1000, wid, dm = sample_fiducial(
            chunk, rng, kappa_1, tau_crit, kappa_2
        )

        pdet, valid = compute_pdet(sf, flu, tau_1000, wid, dm)

        x_ms = tau_600 * (600.0 / 1000.0) ** 4 * 1e3  # at 1000 MHz, ms

        m = (
            np.isfinite(x_ms)
            & (x_ms > 0)
            & valid
            & np.isfinite(pdet)
            & (pdet >= 0.0)
        )

        if np.any(m):
            x_valid = x_ms[m]
            w_valid = pdet[m]

            x_acc.append(x_valid)
            w_acc.append(w_valid)

            n_new = len(x_valid)
            n_have += n_new

            # update progress bar safely
            pbar.update(min(n_new, n_pool - pbar.n))

            # diagnostics
            pbar.set_postfix(
                valid_frac=f"{n_new / len(m):.3f}",
                mean_pdet=f"{np.nanmean(w_valid):.3g}",
            )

    pbar.close()

    x = np.concatenate(x_acc)[:n_pool]
    w = np.concatenate(w_acc)[:n_pool]

    np.savez_compressed(
        pth,
        x_pool_ms_1000=x.astype(np.float32),
        w_pool=w.astype(np.float32),
    )

    print(
        f"[saved] k1={kappa_1:.2f}, tc={tau_crit*1e3:.1f} ms, k2={kappa_2:.2f} "
        f"(n={len(x)})"
    )

    return x.astype(float), w.astype(float)


def build_cache(sf, cache_dir: str, rng_seed: int):
    """Build the full 3-D grid of cached pools."""
    k1_grid  = np.linspace(KAPPA1_MIN,   KAPPA1_MAX,   N_KAPPA1_GRID)
    tc_grid  = np.linspace(TAU_CRIT_MIN_S, TAU_CRIT_MAX_S, N_TAU_CRIT_GRID)
    k2_grid  = np.linspace(KAPPA2_MIN,   KAPPA2_MAX,   N_KAPPA2_GRID)

    grid = list(itertools.product(k1_grid, tc_grid, k2_grid))
    print(f"Cache grid: {len(grid)} cells "
          f"({N_KAPPA1_GRID} × {N_TAU_CRIT_GRID} × {N_KAPPA2_GRID})")

    pools: Dict = {}
    for k1, tc, k2 in tqdm(grid, desc="Building/loading cache cells"):
        seed = int(rng_seed + 1_000_003 * abs(hash((round(k1, 4), round(tc, 6), round(k2, 4)))) % 2**31)
        cell_rng = np.random.default_rng(seed)
        x, w = build_or_load_cell(
            sf=sf, rng=cell_rng,
            kappa_1=k1, tau_crit=tc, kappa_2=k2,
            n_pool=N_POOL_PER_CELL,
            cache_dir=cache_dir,
        )
        pools[(round(k1, 6), round(tc, 8), round(k2, 6))] = (x, w)

    return pools, k1_grid, tc_grid, k2_grid


def load_cache(cache_dir: str) -> Dict:
    """Load all existing cache files into a pools dict."""
    pools = {}
    files = [f for f in os.listdir(cache_dir) if f.startswith("pool_") and f.endswith(".npz")]
    if not files:
        raise RuntimeError(f"No cache files found in {cache_dir}. Run with RUN_CACHE=True.")

    for fname in sorted(files):
        dat = np.load(os.path.join(cache_dir, fname))
        x   = dat["x_pool_ms_1000"].astype(float)
        w   = dat["w_pool"].astype(float)
        # Parse key from filename: pool_k1{...}_tc{...}_k2{...}.npz
        try:
            tag = fname[len("pool_"):-len(".npz")]
            parts = tag.split("_")
            def parse_val(s):
                # remove leading k1/tc/k2 label
                v = s[2:]  # strip 2-char prefix
                sign = -1.0 if v[0] == 'm' else 1.0
                v = v[1:]  # strip sign char
                return sign * float(v.replace("d", "."))
            k1 = parse_val(parts[0])
            tc = parse_val(parts[1])
            k2 = parse_val(parts[2])
            pools[(round(k1, 6), round(tc, 8), round(k2, 6))] = (x, w)
        except Exception:
            continue  # skip malformed filenames

    print(f"Loaded {len(pools)} cached pools.")
    return pools


# ────────────────────────────────────────────────────────────
# Section 5  –  Summary statistic & weighted resample
# ────────────────────────────────────────────────────────────

def weighted_resample(
    rng: np.random.Generator,
    x_pool: np.ndarray,
    w_pool: np.ndarray,
    size: int,
) -> np.ndarray:
    w = np.clip(np.asarray(w_pool, float), 0.0, None)
    total = w.sum()
    if total <= 0.0:
        raise ValueError("All pool weights are zero.")
    idx = rng.choice(len(x_pool), size=int(size), replace=True, p=w / total)
    return x_pool[idx]


def compute_summary(tau_ms: np.ndarray, n_quantiles: int = N_QUANTILES) -> np.ndarray:
    """
    Summary statistic: quantiles of log10(tau_ms) + mean + std.
    Returns a 1-D float32 array of length n_quantiles + 2.
    """
    x = np.log10(tau_ms[np.isfinite(tau_ms) & (tau_ms > 0)])
    if x.size < 5:
        return np.zeros(n_quantiles + 2, dtype=np.float32)
    probs = np.linspace(5, 95, n_quantiles)
    qs    = np.percentile(x, probs)
    return np.concatenate([qs, [x.mean(), x.std()]]).astype(np.float32)


SUMMARY_DIM = N_QUANTILES + 2


def nearest_pool(
    kappa_1: float, tau_crit: float, kappa_2: float,
    pools: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the closest grid cell (L2 in normalised param space)."""
    keys = np.array(list(pools.keys()))  # shape (n_cells, 3)

    # Normalise each axis to [0, 1] for equal weighting
    ranges = keys.max(axis=0) - keys.min(axis=0)
    ranges[ranges == 0] = 1.0
    q = np.array([kappa_1, tau_crit, kappa_2])
    dists = np.sum(((keys - q) / ranges) ** 2, axis=1)
    idx   = int(np.argmin(dists))
    key   = tuple(keys[idx])
    return pools[key]


# ────────────────────────────────────────────────────────────
# Section 6  –  NPE training & inference
# ────────────────────────────────────────────────────────────

def run_sbi(pools: Dict, x_cat_ms: np.ndarray, outdir: str, rng_seed: int):
    rng   = np.random.default_rng(rng_seed)
    n_cat = int(x_cat_ms.size)
    x_obs = compute_summary(x_cat_ms)
    print(f"Observed summary statistic (length {SUMMARY_DIM}): {x_obs}")

    # --- Prior bounds clamped to cache range ---
    keys = np.array(list(pools.keys()))
    #k1_lo = max(KAPPA1_MIN,     float(keys[:, 0].min()))
    #k1_hi = min(KAPPA1_MAX,     float(keys[:, 0].max()))
    k1_lo = KAPPA1_MIN
    k1_hi = KAPPA1_MAX
    tc_lo = max(TAU_CRIT_MIN_S, float(keys[:, 1].min()))
    tc_hi = min(TAU_CRIT_MAX_S, float(keys[:, 1].max()))
    k2_lo = max(KAPPA2_MIN,     float(keys[:, 2].min()))
    k2_hi = min(KAPPA2_MAX,     float(keys[:, 2].max()))
    print(f"Prior: kappa_1 ∈ [{k1_lo:.3f}, {k1_hi:.3f}], "
          f"tau_crit ∈ [{tc_lo*1e3:.1f}, {tc_hi*1e3:.1f}] ms, "
          f"kappa_2 ∈ [{k2_lo:.3f}, {k2_hi:.3f}]")

    # --- Generate (theta, summary) training pairs ---
    print(f"\nGenerating {N_SIMULATIONS} training simulations...")
    thetas, xs = [], []
    n_resamp = int(max(50, round(N_RESAMPLE_MULT * n_cat)))

    for _ in tqdm(range(N_SIMULATIONS), desc="Simulating"):
        k1  = rng.uniform(k1_lo, k1_hi)
        tc  = rng.uniform(tc_lo, tc_hi)
        k2  = rng.uniform(k2_lo, k2_hi)
        x_p, w_p = nearest_pool(k1, tc, k2, pools)
        x_mock   = weighted_resample(rng, x_p, w_p, size=n_resamp)
        thetas.append([k1, tc, k2])
        xs.append(compute_summary(x_mock))

    thetas_t = torch.tensor(np.array(thetas, dtype=np.float32))
    xs_t     = torch.tensor(np.array(xs,     dtype=np.float32))

    # --- NPE ---
    print("\nSetting up NPE...")
    prior = sbi_utils.BoxUniform(
        low  = torch.tensor([k1_lo, tc_lo, k2_lo], dtype=torch.float32),
        high = torch.tensor([k1_hi, tc_hi, k2_hi], dtype=torch.float32),
    )
    logdir = os.path.join(outdir, "sbi_logs")
    os.makedirs(logdir, exist_ok=True)

    writer = SummaryWriter(log_dir=logdir)

    inference = sbi_inference.SNPE(
        prior=prior,
        summary_writer=writer,
    )
    inference = inference.append_simulations(thetas_t, xs_t)

    print("Training neural posterior estimator...")
    density_estimator = inference.train(
        training_batch_size=256,
        learning_rate=5e-4,
        max_num_epochs=500,
        stop_after_epochs=20,
        show_train_summary=True,
    )
    posterior = inference.build_posterior(density_estimator)

    # --- Sample posterior ---
    print(f"\nSampling {N_POSTERIOR_SAMPLES} posterior samples...")
    x_obs_t = torch.tensor(x_obs, dtype=torch.float32)
    posterior.set_default_x(x_obs_t)
    samples  = posterior.sample((N_POSTERIOR_SAMPLES,), x=x_obs_t)
    samples_np = samples.numpy()   # shape (N_POSTERIOR_SAMPLES, 3)

    k1_s  = samples_np[:, 0]
    tc_s  = samples_np[:, 1]
    k2_s  = samples_np[:, 2]

    # --- Summarise ---
    def ci(arr):
        med = float(np.median(arr))
        lo68, hi68 = np.percentile(arr, [16.0, 84.0])
        lo95, hi95 = np.percentile(arr, [2.5,  97.5])
        return dict(median=med, ci_16_84=[float(lo68), float(hi68)],
                    ci_2p5_97p5=[float(lo95), float(hi95)])

    print("\nPosterior summary:")
    for name, arr in [("kappa_1", k1_s), ("tau_crit_s", tc_s), ("kappa_2", k2_s)]:
        c = ci(arr)
        print(f"  {name:12s}: median = {c['median']:+.4f}, "
              f"68% CI = [{c['ci_16_84'][0]:+.4f}, {c['ci_16_84'][1]:+.4f}]")

    result = {
        "params": dict(
            n_simulations=N_SIMULATIONS,
            n_resample_mult=N_RESAMPLE_MULT,
            n_quantiles=N_QUANTILES,
            prior_kappa1=[k1_lo, k1_hi],
            prior_tau_crit_s=[tc_lo, tc_hi],
            prior_kappa2=[k2_lo, k2_hi],
            n_cat=n_cat,
        ),
        "posterior": dict(
            kappa_1=ci(k1_s),
            tau_crit_s=ci(tc_s),
            kappa_2=ci(k2_s),
        ),
    }

    out_json = os.path.join(outdir, "posterior_3param.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote: {out_json}")

    out_npy = os.path.join(outdir, "posterior_3param_samples.npy")
    np.save(out_npy, samples_np)
    print(f"Wrote: {out_npy}")

    return samples_np, result


# ────────────────────────────────────────────────────────────
# Section 7  –  Plotting
# ────────────────────────────────────────────────────────────

def make_corner_plot(samples_np: np.ndarray, meta: dict, outdir: str):
    """
    Corner plot of the three-parameter posterior.

    Shows 16/50/84 percentile credible intervals on diagonal titles.
    Uses corner.py if available; otherwise falls back to a manual 3x3 plot
    with approximate 2D enclosed-probability contours.
    """
    k1_s = samples_np[:, 0]
    tc_s = samples_np[:, 1] * 1e3   # display tau_crit in ms
    k2_s = samples_np[:, 2]

    labels = [r"$\kappa_1$", r"$\tau_{\rm crit}$ (ms)", r"$\kappa_2$"]
    data = np.column_stack([k1_s, tc_s, k2_s])

    if HAS_CORNER:
        fig = corner.corner(
            data,
            labels=labels,
            bins=50,
            smooth=1.0,
            smooth1d=1.0,

            # Diagonal credible intervals
            quantiles=[0.16, 0.50, 0.84],
            show_titles=True,
            title_quantiles=[0.16, 0.50, 0.84],
            title_fmt=".3g",
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 14},

            # Off-diagonal contours
            plot_contours=True,
            fill_contours=True,
            levels=(0.393, 0.865, 0.989),  # 1, 2, 3 sigma for 2D Gaussian
            contour_kwargs={"linewidths": 1.2},
            contourf_kwargs={"alpha": 0.45},
            plot_datapoints=False,
        )

    else:
        n = 3
        fig, axes = plt.subplots(n, n, figsize=(8, 8), constrained_layout=True)

        def _hist1d(ax, x, label):
            ax.hist(x, bins=60, density=True, histtype="step", linewidth=2)

            q16, q50, q84 = np.percentile(x, [16, 50, 84])
            err_lo = q50 - q16
            err_hi = q84 - q50

            ax.axvline(q50, linestyle="--", linewidth=1.5, color="k")
            ax.axvspan(q16, q84, alpha=0.2)

            ax.set_title(
                rf"{label} = {q50:.3g}$^{{+{err_hi:.2g}}}_{{-{err_lo:.2g}}}$",
                fontsize=11,
            )
            ax.set_xlabel(label, fontsize=13)
            ax.set_yticks([])

        def _hist2d(ax, x, y, xlabel, ylabel):
            H, xedges, yedges = np.histogram2d(x, y, bins=50, density=True)

            xcenters = 0.5 * (xedges[:-1] + xedges[1:])
            ycenters = 0.5 * (yedges[:-1] + yedges[1:])
            XX, YY = np.meshgrid(xcenters, ycenters)

            H = H.T

            Hflat = H.ravel()
            order = np.argsort(Hflat)[::-1]
            Hsort = Hflat[order]

            cumsum = np.cumsum(Hsort)
            cumsum /= cumsum[-1]

            probs = [0.393, 0.865, 0.989]
            levels = [
                Hsort[np.searchsorted(cumsum, p)]
                for p in probs
            ]
            levels = np.sort(levels)

            ax.contourf(
                XX,
                YY,
                H,
                levels=np.r_[levels, H.max()],
                alpha=0.35,
            )
            ax.contour(
                XX,
                YY,
                H,
                levels=levels,
                linewidths=1.2,
            )

            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

        arrays = [k1_s, tc_s, k2_s]

        for i in range(n):
            for j in range(n):
                ax = axes[i][j]

                if i == j:
                    _hist1d(ax, arrays[i], labels[i])
                elif i > j:
                    _hist2d(ax, arrays[j], arrays[i], labels[j], labels[i])
                else:
                    ax.set_visible(False)

    out = os.path.join(outdir, "posterior_corner.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {out}")


def make_ensemble_plot(samples_np: np.ndarray, meta: dict, outdir: str):
    """
    Posterior ensemble plot: N_POST_CURVES draws from the posterior,
    each rendered as a density-per-dex curve on a log10(tau) axis.
    """
    rng = np.random.default_rng(POST_RNG_SEED)
    n_show = min(N_POST_CURVES, len(samples_np))
    idx    = rng.choice(len(samples_np), size=n_show, replace=False)
    draws  = samples_np[idx]   # (n_show, 3)

    xgrid = np.logspace(np.log10(TAU_MIN_S), np.log10(TAU_MAX_S), 2000)

    # Reference normalisation value (at NORM_X_S using the posterior median)
    k1_med = float(np.median(samples_np[:, 0]))
    tc_med = float(np.median(samples_np[:, 1]))
    k2_med = float(np.median(samples_np[:, 2]))

    ref_model = TwoSegmentSemiLogLine(k1_med, tc_med, k2_med, y_crit=Y_CRIT)
    ref_dex   = density_per_dex(None, ref_model, xgrid)
    y_target = NORM_Y

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    # Draw ensemble
    for row in draws:
        k1, tc, k2 = float(row[0]), float(row[1]), float(row[2])
        try:
            model = TwoSegmentSemiLogLine(k1, tc, k2, y_crit=Y_CRIT)
        except ValueError:
            continue
        yd = density_per_dex(None, model, xgrid)
        ax.plot(xgrid, yd, linewidth=1.5, alpha=0.06, color="C0")

    ax.set_xscale("log")
    ax.set_xlabel(r"Scattering timescale $\tau_{600\,\rm MHz}$ (s)", fontsize=14)
    ax.set_ylabel("Arbitrary Normalization", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(TAU_MIN_S, TAU_MAX_S)

    out = os.path.join(outdir, "posterior_ensemble.pdf")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out}")


def make_marginal_1d_plots(samples_np: np.ndarray, meta: dict, outdir: str):
    """
    Individual 1-D marginal posteriors for each parameter.
    """
    names  = ["kappa_1", "tau_crit_ms", "kappa_2"]
    labels = [r"$\kappa_1$", r"$\tau_{\rm crit}$ (ms)", r"$\kappa_2$"]
    arrs   = [samples_np[:, 0], samples_np[:, 1] * 1e3, samples_np[:, 2]]
    priors = [
        (KAPPA1_MIN,       KAPPA1_MAX),
        (TAU_CRIT_MIN_S * 1e3, TAU_CRIT_MAX_S * 1e3),
        (KAPPA2_MIN,       KAPPA2_MAX),
    ]
    fnames = ["marginal_kappa1.pdf", "marginal_taucrit.pdf", "marginal_kappa2.pdf"]

    for name, label, arr, (plo, phi), fname in zip(names, labels, arrs, priors, fnames):
        fig, ax = plt.subplots(figsize=(5, 4.5), constrained_layout=True)

        med          = float(np.median(arr))
        lo68, hi68   = np.percentile(arr, [16.0, 84.0])
        lo95, hi95   = np.percentile(arr, [2.5,  97.5])

        ax.hist(arr, bins=np.linspace(plo, phi, 80), density=True,
                histtype="step", linewidth=2.5)
        ax.axvline(med, linestyle="--", linewidth=2,
                   label=f"Median = {med:.3f}")
        ax.axvspan(lo68, hi68, alpha=0.22,
                   label=f"68% CI [{lo68:.3f}, {hi68:.3f}]")
        ax.axvspan(lo95, hi95, alpha=0.10,
                   label=f"95% CI [{lo95:.3f}, {hi95:.3f}]")

        ax.set_xlabel(label, fontsize=16)
        ax.set_ylabel("Posterior density", fontsize=14)
        ax.tick_params(axis="both", labelsize=11)
        ax.set_xlim(plo, phi)
        ax.legend(fontsize=9)

        out = os.path.join(outdir, fname)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote: {out}")


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTDIR,    exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # --- Stage 1: cache ---
    if RUN_CACHE:
        print("=" * 60)
        print("Stage 1 – Building simulation cache")
        print("=" * 60)
        sf = build_selection_function()
        pools, *_ = build_cache(sf, CACHE_DIR, RNG_SEED)
        print("Cache complete.\n")
    else:
        print("Stage 1 skipped (RUN_CACHE=False); loading existing cache...")
        pools = load_cache(CACHE_DIR)

    # --- Stage 2: SBI ---
    if RUN_SBI:
        print("=" * 60)
        print("Stage 2 – Neural Posterior Estimation")
        print("=" * 60)
        x_cat_ms   = load_catalog_scattering(CAT2_JSON)
        print(f"Catalog scattering times: n={len(x_cat_ms)}")
        samples_np, meta = run_sbi(pools, x_cat_ms, OUTDIR, RNG_SEED)
    else:
        print("Stage 2 skipped (RUN_SBI=False); loading existing posterior...")
        samples_np = np.load(os.path.join(OUTDIR, "posterior_3param_samples.npy"))
        with open(os.path.join(OUTDIR, "posterior_3param.json")) as f:
            meta = json.load(f)

    # --- Stage 3: plots ---
    if RUN_PLOTS:
        print("=" * 60)
        print("Stage 3 – Plotting")
        print("=" * 60)
        make_corner_plot(samples_np, meta, OUTDIR)
        make_ensemble_plot(samples_np, meta, OUTDIR)
        make_marginal_1d_plots(samples_np, meta, OUTDIR)
        print("All plots written.")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()