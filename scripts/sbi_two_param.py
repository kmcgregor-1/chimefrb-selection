#!/usr/bin/env python3
"""
sbi_two_param.py

Full pipeline for inferring the intrinsic scattering-time distribution using
Simulation-Based Inference (NPE) with a two-parameter model:

  Parameters
  ----------
  tau_pivot : float  (seconds, at 600 MHz)
      The pivot point where the lognormal transitions to the power-law tail.
      Replaces the fixed scale parameter (formerly = sigma-mode of lognormal).
      Prior: Uniform[TAU_PIVOT_MIN_S, TAU_PIVOT_MAX_S]

  kappa : float
      Log-slope of the power-law tail above tau_pivot.
      In density-per-dex space (log10 tau on x), positive kappa gives a
      rising tail, negative gives a downturn.
      Prior: Uniform[KAPPA_MIN, KAPPA_MAX]

  Model description
  -----------------
  Below tau_pivot:   lognormal PDF with fixed sigma and scale=tau_pivot
  Above tau_pivot:   power-law tail  f(tau) ∝ tau^(kappa - 1)
  The two segments join continuously at tau_pivot.

Pipeline stages (all run by default; set RUN_* flags to skip stages)
----------------------------------------------------------------------
  Stage 1 – Build / load cache
      Build a grid of (tau_pivot, kappa) pools. Each cell draws a large
      pool of scattering times and computes p_det weights via the CHIME-FRB
      selection function, saving (x_pool_ms_1000, w_pool) per grid cell.

  Stage 2 – Train NPE
      Draw random (tau_pivot, kappa) pairs from the prior, retrieve the
      nearest cached pool, importance-resample a mock catalog, compute
      summary statistics, and train a Neural Posterior Estimator (NPE).

  Stage 3 – Plot results
      (a) Corner plot of the 2-D joint posterior.
      (b) 1-D marginal posteriors for each parameter.
      (c) Ensemble of distribution curves sampled from the posterior.

Usage
-----
  Edit the USER PARAMS block below, then:
      python sbi_two_param.py

Dependencies
------------
  pip install sbi torch scipy matplotlib tqdm corner
  (plus the internal chimefrb_selection package)
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

OUTDIR    = "/data/user-data/kmcgregor/bootstrap_downturn/sbi_2param"
CACHE_DIR = os.path.join(OUTDIR, "cache_2param")

CAT2_JSON = "/data/user-data/ssiegel/catalog2/table/20251121/chimefrbcat2.json"

# ---------- Prior bounds ----------
KAPPA_MIN, KAPPA_MAX           = -2.0, 2.0
TAU_PIVOT_MIN_S, TAU_PIVOT_MAX_S = 5e-3, 3e-2   # seconds at 600 MHz

# ---------- Fixed lognormal shape (left-of-pivot segment) ----------
SIGMA_LN = 1.94   # lognormal shape parameter (kept fixed)

# ---------- Model support ----------
TAU_MIN_S = 1e-4   # seconds at 600 MHz (hard lower bound)
TAU_MAX_S = 0.30    # seconds at 600 MHz (hard upper bound)

# ---------- Cache grid ----------
N_TAU_PIVOT_GRID = 10    # grid points along tau_pivot axis
N_KAPPA_GRID     = 10    # grid points along kappa axis
N_POOL_PER_CELL  = 500_000

# ---------- SBI training ----------
N_SIMULATIONS        = 80_000
N_RESAMPLE_MULT      = 1.0    # mock catalog size = N_RESAMPLE_MULT * n_cat
N_QUANTILES          = 12
N_POSTERIOR_SAMPLES  = 200_000

# ---------- Plotting ----------
N_POST_CURVES  = 500     # ensemble curves drawn from posterior
POST_RNG_SEED  = 12345
NORM_X_S       = 13.1e-3  # normalization anchor for shape plots (seconds)
NORM_Y         = 0.5

# ---------- Pipeline control ----------
RUN_CACHE  = True
RUN_SBI    = True
RUN_PLOTS  = True

# ---------- Catalog cuts ----------
SNR_CUT        = 12.0
DM_LO, DM_HI  = 100.0, 5000.0

RNG_SEED = 12346

# ============================================================


# ────────────────────────────────────────────────────────────
# Section 1  –  Two-parameter scattering distribution
# ────────────────────────────────────────────────────────────

class LognormalPivotPowerLaw:
    """
    Intrinsic scattering-time distribution with a free pivot point.

    Below tau_pivot (and above tau_min):
        Lognormal PDF with shape=SIGMA_LN and scale=tau_pivot.
        The scale is set to tau_pivot so that the lognormal mode aligns with
        the pivot, keeping the left-side shape consistent as tau_pivot varies.

    Above tau_pivot (and below tau_max):
        Power-law tail  f(tau) ∝ tau^(kappa - 1), joined continuously at
        tau_pivot.  In density-per-dex space this corresponds to a slope of
        kappa (positive = upturn, negative = downturn, zero = flat).

    Parameters
    ----------
    kappa : float
        Power-law index of the tail (= tail_k in the original code).
    tau_pivot : float
        Transition point in seconds at 600 MHz.  Replaces the fixed `scale`
        parameter from the single-parameter model.
    sigma : float
        Lognormal shape parameter (fixed; default SIGMA_LN).
    tau_min, tau_max : float
        Hard bounds on the support.
    """

    def __init__(
        self,
        kappa: float,
        tau_pivot: float,
        *,
        sigma: float = SIGMA_LN,
        tau_min: float = TAU_MIN_S,
        tau_max: float = TAU_MAX_S,
    ):
        self.tail_k   = float(kappa)
        self.pivot    = float(tau_pivot)
        self.sigma    = float(sigma)
        self.tau_min  = float(tau_min)
        self.tau_max  = float(tau_max)

        if not (self.tau_min < self.pivot < self.tau_max):
            raise ValueError(
                f"Need tau_min < tau_pivot < tau_max; "
                f"got {self.tau_min}, {self.pivot}, {self.tau_max}"
            )

        # Lognormal with scale = tau_pivot so the mode sits at the pivot
        self._ln     = lognorm(s=self.sigma, scale=self.pivot)
        self._w_left = self._compute_w_left()

    # ------------------------------------------------------------------
    def _compute_w_left(self) -> float:
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
                self.tau_max ** self.tail_k - self.pivot ** self.tail_k
            )
            right_mass = float(max(right_mass, 0.0))

        total = left_mass + right_mass
        if not (np.isfinite(total) and total > 0):
            raise RuntimeError(
                f"Total mass is non-positive for kappa={self.tail_k:.4f}, "
                f"tau_pivot={self.pivot:.5f}; check parameter bounds."
            )
        return left_mass / total

    # ------------------------------------------------------------------
    def pdf(self, tau: np.ndarray) -> np.ndarray:
        """Linear-tau PDF."""
        tau = np.asarray(tau, dtype=float)
        pdf = np.zeros_like(tau)

        in_support = (tau >= self.tau_min) & (tau <= self.tau_max)
        left       = in_support & (tau <= self.pivot)
        right      = in_support & (tau > self.pivot)

        if np.any(left):
            pdf[left] = self._ln.pdf(tau[left])

        if np.any(right):
            f_p = float(self._ln.pdf(self.pivot))
            A   = (self.pivot ** (1.0 - self.tail_k)) * f_p
            pdf[right] = A * np.power(tau[right], self.tail_k - 1.0)

        return pdf

    # ------------------------------------------------------------------
    def density_per_dex(self, tau: np.ndarray) -> np.ndarray:
        """
        Density per dex:  g(log10 tau) = tau * ln(10) * f(tau).
        This is the natural quantity to plot on a log10-tau axis.
        """
        tau = np.asarray(tau, dtype=float)
        p   = self.pdf(tau)
        out = np.zeros_like(tau)
        m   = tau > 0
        out[m] = tau[m] * np.log(10.0) * p[m]
        return out

    # ------------------------------------------------------------------
    def rvs(self, size: int = 1, random_state=None) -> np.ndarray:
        n   = int(size)
        rng = np.random.default_rng(random_state)
        u   = rng.uniform(0.0, 1.0, size=n)

        left    = u < self._w_left
        n_left  = int(np.sum(left))
        n_right = n - n_left
        out     = np.empty(n, dtype=float)

        if n_left > 0:
            c_lo = self._ln.cdf(self.tau_min)
            c_p  = self._ln.cdf(self.pivot)
            if c_p <= c_lo:
                out[left] = self.tau_min
            else:
                u_left     = rng.uniform(c_lo, c_p, size=n_left)
                out[left]  = np.clip(self._ln.ppf(u_left), self.tau_min, self.pivot)

        if n_right > 0:
            u_right = rng.uniform(0.0, 1.0, size=n_right)
            if self.tau_max <= self.pivot * (1.0 + 1e-15):
                out[~left] = self.pivot
            elif abs(self.tail_k) < 1e-12:
                out[~left] = self.pivot * (self.tau_max / self.pivot) ** u_right
            else:
                xk = (
                    self.pivot ** self.tail_k
                    + u_right * (self.tau_max ** self.tail_k - self.pivot ** self.tail_k)
                )
                out[~left] = np.power(xk, 1.0 / self.tail_k)

        return np.clip(out, self.tau_min, self.tau_max)


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


def draw_fluence_powerlaw(N: int, rng: np.random.Generator,
                          alpha: float = -1.203108427072943,
                          fmin: float = 0.1, fmax: float = 10000.0) -> np.ndarray:
    s_max = (fmax / fmin) ** alpha
    u     = rng.uniform(s_max, 1.0, size=N)
    return fmin * (u ** (1.0 / alpha))


def draw_lognorm_truncated(N: int, rng: np.random.Generator,
                           shape: float, scale: float,
                           lo: float, hi: float) -> np.ndarray:
    dist = lognorm(s=shape, scale=scale)
    c_lo = dist.cdf(lo)
    c_hi = dist.cdf(hi)
    u    = rng.uniform(c_lo, c_hi, size=N)
    return np.clip(dist.ppf(u), lo, hi)


def sample_fiducial(N: int, rng: np.random.Generator,
                    kappa: float, tau_pivot: float
                    ) -> Tuple[np.ndarray, ...]:
    flu     = draw_fluence_powerlaw(N, rng)
    dm      = draw_lognorm_truncated(N, rng, 0.6076516890745998, 534.4727066208081,  DM_LO, DM_HI)
    wid     = draw_lognorm_truncated(N, rng, 1.1012853240184415, 7.389903164298552e-4, 1e-4, 0.2)

    model   = LognormalPivotPowerLaw(kappa=kappa, tau_pivot=tau_pivot)
    tau_600 = model.rvs(size=N, random_state=rng)

    mask = np.isfinite(tau_600) & (tau_600 >= TAU_MIN_S)
    return (
        flu[mask],
        tau_600[mask],
        tau_600[mask] * (600.0 / 1000.0) ** 4,   # tau at 1000 MHz (seconds)
        wid[mask],
        dm[mask],
    )


def compute_pdet(sf, flu, tau_1000_s, wid_s, dm
                 ) -> Tuple[np.ndarray, np.ndarray]:
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

def _to_float(x) -> float:
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
        if (
            np.isfinite(_to_float(frb.get("bonsai_snr", np.nan))) and
            _to_float(frb.get("bonsai_snr", np.nan)) >= SNR_CUT and
            np.isfinite(_to_float(frb.get("dm_fitb", np.nan))) and
            DM_LO <= _to_float(frb.get("dm_fitb", np.nan)) <= DM_HI
        )
    ]
    if not cut:
        raise RuntimeError("No catalog entries survived the cuts.")

    scat_key = next(
        (k for k in ["scat_time", "scattering_time", "tau"] if k in cut[0]), None
    )
    scat_s = (
        np.array([_to_float(frb.get(scat_key, np.nan)) for frb in cut], float)
        if scat_key else np.full(len(cut), np.nan)
    )

    # heuristic unit fix (ms → s if median > 0.5)
    if np.nanmedian(scat_s) > 0.5:
        scat_s *= 1e-3

    # scale from 400 MHz to 1000 MHz and convert to ms
    tau_ms = scat_s * (400.0 / 1000.0) ** 4 * 1e3
    m = np.isfinite(tau_ms) & (tau_ms > 0)
    if m.sum() < 10:
        raise RuntimeError(f"Too few valid scattering times after cuts: n={m.sum()}")
    return tau_ms[m]


# ────────────────────────────────────────────────────────────
# Section 4  –  Cache building
# ────────────────────────────────────────────────────────────

def _cell_tag(tau_pivot: float, kappa: float) -> str:
    def fmt(v: float) -> str:
        return f"{v:+.4f}".replace("+", "p").replace("-", "m").replace(".", "d")
    return f"tp{fmt(tau_pivot)}_k{fmt(kappa)}"


def cache_path(cache_dir: str, tau_pivot: float, kappa: float) -> str:
    return os.path.join(cache_dir, f"pool_{_cell_tag(tau_pivot, kappa)}.npz")


def build_or_load_cell(
    *,
    sf,
    rng: np.random.Generator,
    tau_pivot: float,
    kappa: float,
    n_pool: int,
    cache_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_pool_ms_1000, w_pool), building from simulation if not cached."""
    os.makedirs(cache_dir, exist_ok=True)
    pth = cache_path(cache_dir, tau_pivot, kappa)

    if os.path.exists(pth):
        print(f"[cache hit] tau_pivot={tau_pivot*1e3:.1f} ms, kappa={kappa:+.3f}")
        dat = np.load(pth)
        return dat["x_pool_ms_1000"].astype(float), dat["w_pool"].astype(float)

    print(f"[building]  tau_pivot={tau_pivot*1e3:.1f} ms, kappa={kappa:+.3f}")

    x_acc: List[np.ndarray] = []
    w_acc: List[np.ndarray] = []
    chunk = 1000

    n_acc = 0
    n_draw = 0

    with tqdm(
        total=n_pool,
        desc=f"cell tp={tau_pivot*1e3:.1f} ms, k={kappa:+.3f}",
        leave=True,
        colour="green"
    ) as pbar:

        while n_acc < n_pool:
            flu, tau_600, tau_1000, wid, dm = sample_fiducial(
                chunk, rng, kappa, tau_pivot
            )

            pdet, valid = compute_pdet(sf, flu, tau_1000, wid, dm)

            # Convert tau_600 seconds to tau_1000 MHz in ms for storage
            x_ms_1000 = np.asarray(tau_600, float) * (600.0 / 1000.0) ** 4 * 1e3

            m = (
                np.isfinite(x_ms_1000) & (x_ms_1000 > 0) &
                valid & np.isfinite(pdet) & (pdet >= 0.0)
            )

            n_good = int(np.sum(m))
            n_draw += len(tau_600)

            if n_good > 0:
                x_acc.append(x_ms_1000[m])
                w_acc.append(pdet[m])

                n_new = min(n_good, n_pool - n_acc)
                n_acc += n_new
                pbar.update(n_new)

            mean_pdet = np.nanmean(pdet[m]) if n_good > 0 else np.nan

            pbar.set_postfix(
                drawn=n_draw,
                valid=n_acc,
                keep_frac=f"{n_acc / max(n_draw, 1):.3f}",
                mean_pdet=f"{mean_pdet:.3g}" if np.isfinite(mean_pdet) else "nan",
            )

    x = np.concatenate(x_acc)[:n_pool]
    w = np.concatenate(w_acc)[:n_pool]

    np.savez_compressed(
        pth,
        x_pool_ms_1000=x.astype(np.float32),
        w_pool=w.astype(np.float32),
    )

    print(
        f"  Saved: {os.path.basename(pth)}  "
        f"(n={len(x)}, mean_pdet={np.nanmean(w):.3g}, "
        f"nonzero_pdet_frac={np.mean(w > 0):.3f})"
    )

    return x.astype(float), w.astype(float)

def build_cache(sf, cache_dir: str, rng_seed: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Build (or load) the full (tau_pivot, kappa) grid cache."""
    tau_pivot_grid = np.linspace(TAU_PIVOT_MIN_S, TAU_PIVOT_MAX_S, N_TAU_PIVOT_GRID)
    kappa_grid     = np.linspace(KAPPA_MIN, KAPPA_MAX, N_KAPPA_GRID)
    grid           = [(tp, k) for tp in tau_pivot_grid for k in kappa_grid]

    rng   = np.random.default_rng(rng_seed)
    pools: Dict = {}

    for tp, k in tqdm(grid, desc="Building/loading cache cells", colour="blue"):
        seed     = int(rng_seed + 1_000_003 * abs(hash((round(tp, 6), round(k, 4)))) % 2**31)
        cell_rng = np.random.default_rng(seed)
        x, w = build_or_load_cell(
            sf=sf, rng=cell_rng,
            tau_pivot=tp, kappa=k,
            n_pool=N_POOL_PER_CELL,
            cache_dir=cache_dir,
        )
        pools[(round(tp, 8), round(k, 6))] = (x, w)

    return pools, tau_pivot_grid, kappa_grid


def load_cache(cache_dir: str) -> Dict:
    """Load all existing cache files into a pools dict."""
    files = [f for f in os.listdir(cache_dir) if f.startswith("pool_") and f.endswith(".npz")]
    if not files:
        raise RuntimeError(f"No cache files found in {cache_dir}. Run with RUN_CACHE=True.")

    pools: Dict = {}
    for fname in sorted(files):
        dat = np.load(os.path.join(cache_dir, fname))
        x   = dat["x_pool_ms_1000"].astype(float)
        w   = dat["w_pool"].astype(float)
        try:
            # Filename: pool_tp{sign}{int}d{frac}_k{sign}{int}d{frac}.npz
            tag   = fname[len("pool_"):-len(".npz")]
            parts = tag.split("_")

            def parse_val(s: str, prefix_len: int) -> float:
                v    = s[prefix_len:]
                sign = -1.0 if v[0] == "m" else 1.0
                v    = v[1:]
                return sign * float(v.replace("d", "."))

            tp = parse_val(parts[0], 2)   # "tp" prefix = 2 chars
            k  = parse_val(parts[1], 1)   # "k"  prefix = 1 char
            pools[(round(tp, 8), round(k, 6))] = (x, w)
        except Exception:
            continue   # skip malformed filenames

    print(f"Loaded {len(pools)} cached pools from {cache_dir}")
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
    w     = np.clip(np.asarray(w_pool, float), 0.0, None)
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


def nearest_pool(tau_pivot: float, kappa: float, pools: Dict
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Find the closest grid cell (L2 distance in normalised parameter space)."""
    keys   = np.array(list(pools.keys()))   # shape (n_cells, 2): [tau_pivot, kappa]
    ranges = keys.max(axis=0) - keys.min(axis=0)
    ranges[ranges == 0] = 1.0
    q      = np.array([tau_pivot, kappa])
    dists  = np.sum(((keys - q) / ranges) ** 2, axis=1)
    idx    = int(np.argmin(dists))
    return pools[tuple(keys[idx])]


# ────────────────────────────────────────────────────────────
# Section 6  –  NPE training & inference
# ────────────────────────────────────────────────────────────

def run_sbi(pools: Dict, x_cat_ms: np.ndarray, outdir: str, rng_seed: int
            ) -> Tuple[np.ndarray, dict]:
    rng   = np.random.default_rng(rng_seed)
    n_cat = int(x_cat_ms.size)
    x_obs = compute_summary(x_cat_ms)
    print(f"Observed summary statistic (length {SUMMARY_DIM}): {x_obs}")

    # --- Prior bounds clamped to cache range ---
    keys   = np.array(list(pools.keys()))   # (n_cells, 2)
    tp_lo  = max(TAU_PIVOT_MIN_S, float(keys[:, 0].min()))
    tp_hi  = min(TAU_PIVOT_MAX_S, float(keys[:, 0].max()))
    k_lo   = max(KAPPA_MIN,       float(keys[:, 1].min()))
    k_hi   = min(KAPPA_MAX,       float(keys[:, 1].max()))
    print(
        f"Prior: tau_pivot ∈ [{tp_lo*1e3:.1f}, {tp_hi*1e3:.1f}] ms, "
        f"kappa ∈ [{k_lo:.3f}, {k_hi:.3f}]"
    )

    # --- Generate (theta, summary) training pairs ---
    print(f"\nGenerating {N_SIMULATIONS} training simulations...")
    thetas: List = []
    xs: List     = []
    n_resamp     = int(max(50, round(N_RESAMPLE_MULT * n_cat)))

    for _ in tqdm(range(N_SIMULATIONS), desc="Simulating"):
        tp      = rng.uniform(tp_lo, tp_hi)
        k       = rng.uniform(k_lo, k_hi)
        x_p, w_p = nearest_pool(tp, k, pools)
        x_mock  = weighted_resample(rng, x_p, w_p, size=n_resamp)
        thetas.append([tp, k])
        xs.append(compute_summary(x_mock))

    thetas_t = torch.tensor(np.array(thetas, dtype=np.float32))
    xs_t     = torch.tensor(np.array(xs,     dtype=np.float32))

    # --- NPE ---
    print("\nSetting up NPE...")
    prior = sbi_utils.BoxUniform(
        low  = torch.tensor([tp_lo, k_lo], dtype=torch.float32),
        high = torch.tensor([tp_hi, k_hi], dtype=torch.float32),
    )
    logdir = os.path.join(outdir, "sbi_logs")
    os.makedirs(logdir, exist_ok=True)

    inference = sbi_inference.SNPE(
        prior=prior,
        summary_writer=SummaryWriter(log_dir=logdir),
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
    samples    = posterior.sample((N_POSTERIOR_SAMPLES,), x=x_obs_t)
    samples_np = samples.numpy()   # shape (N_POSTERIOR_SAMPLES, 2)

    tp_s = samples_np[:, 0]
    k_s  = samples_np[:, 1]

    # --- Summarise ---
    def ci(arr: np.ndarray) -> dict:
        med = float(np.median(arr))
        lo68, hi68 = np.percentile(arr, [16.0, 84.0])
        lo95, hi95 = np.percentile(arr, [2.5,  97.5])
        return dict(
            median=med,
            ci_16_84=[float(lo68), float(hi68)],
            ci_2p5_97p5=[float(lo95), float(hi95)],
        )

    print("\nPosterior summary:")
    for name, arr in [("tau_pivot_s", tp_s), ("kappa", k_s)]:
        c = ci(arr)
        print(
            f"  {name:14s}: median = {c['median']:+.4f}, "
            f"68% CI = [{c['ci_16_84'][0]:+.4f}, {c['ci_16_84'][1]:+.4f}]"
        )

    result = {
        "params": dict(
            n_simulations=N_SIMULATIONS,
            n_resample_mult=N_RESAMPLE_MULT,
            n_quantiles=N_QUANTILES,
            prior_tau_pivot_s=[tp_lo, tp_hi],
            prior_kappa=[k_lo, k_hi],
            sigma_ln=SIGMA_LN,
            n_cat=n_cat,
        ),
        "posterior": dict(
            tau_pivot_s=ci(tp_s),
            kappa=ci(k_s),
        ),
    }

    out_json = os.path.join(outdir, "posterior_2param.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote: {out_json}")

    out_npy = os.path.join(outdir, "posterior_2param_samples.npy")
    np.save(out_npy, samples_np)
    print(f"Wrote: {out_npy}")

    return samples_np, result


# ────────────────────────────────────────────────────────────
# Section 7  –  Plotting
# ────────────────────────────────────────────────────────────

def make_corner_plot(samples_np: np.ndarray, meta: dict, outdir: str):
    """
    Corner plot of the (tau_pivot, kappa) joint posterior.
    Uses corner.py if available; otherwise falls back to a manual 2×2 grid.
    """
    tp_s = samples_np[:, 0] * 1e3   # display in ms
    k_s  = samples_np[:, 1]

    labels = [r"$\tau_{\rm pivot}$ (ms)", r"$\kappa$"]
    data   = np.column_stack([tp_s, k_s])

    if HAS_CORNER:
        fig = corner.corner(
            data,
            labels=labels,
            bins=50,
            smooth=1.0,
            smooth1d=1.0,
            quantiles=[0.16, 0.50, 0.84],
            show_titles=True,
            title_quantiles=[0.16, 0.50, 0.84],
            title_fmt=".3g",
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 14},
            plot_contours=True,
            fill_contours=True,
            levels=(0.393, 0.865, 0.989),
            contour_kwargs={"linewidths": 1.2},
            contourf_kwargs={"alpha": 0.45},
            plot_datapoints=False,
        )
    else:
        n = 2
        fig, axes = plt.subplots(n, n, figsize=(7, 7), constrained_layout=True)

        def _hist1d(ax, x, label):
            q16, q50, q84 = np.percentile(x, [16, 50, 84])
            plo = meta["params"]["prior_tau_pivot_s"][0] * 1e3 if "tau_pivot" in label.lower() else meta["params"]["prior_kappa"][0]
            phi = meta["params"]["prior_tau_pivot_s"][1] * 1e3 if "tau_pivot" in label.lower() else meta["params"]["prior_kappa"][1]
            ax.hist(x, bins=np.linspace(plo, phi, 80), density=True,
                    histtype="step", linewidth=2)
            ax.axvline(q50, linestyle="--", linewidth=1.5, color="k")
            ax.axvspan(q16, q84, alpha=0.2)
            ax.set_title(
                rf"{label} = {q50:.3g}$^{{+{q84-q50:.2g}}}_{{-{q50-q16:.2g}}}$",
                fontsize=11,
            )
            ax.set_xlabel(label, fontsize=13)
            ax.set_yticks([])

        def _hist2d(ax, x, y, xlabel, ylabel):
            H, xe, ye = np.histogram2d(x, y, bins=50, density=True)
            xc = 0.5 * (xe[:-1] + xe[1:])
            yc = 0.5 * (ye[:-1] + ye[1:])
            XX, YY = np.meshgrid(xc, yc)
            H = H.T
            Hflat = H.ravel()
            order = np.argsort(Hflat)[::-1]
            cumsum = np.cumsum(Hflat[order])
            cumsum /= cumsum[-1]
            probs  = [0.393, 0.865, 0.989]
            levels = np.sort([Hflat[order][np.searchsorted(cumsum, p)] for p in probs])
            ax.contourf(XX, YY, H, levels=np.r_[levels, H.max()], alpha=0.35)
            ax.contour( XX, YY, H, levels=levels, linewidths=1.2)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

        arrays = [tp_s, k_s]
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


def make_marginal_1d_plots(samples_np: np.ndarray, meta: dict, outdir: str):
    """Individual 1-D marginal posteriors for tau_pivot and kappa."""
    names  = ["tau_pivot_ms", "kappa"]
    labels = [r"$\tau_{\rm pivot}$ (ms)", r"$\kappa$"]
    arrs   = [samples_np[:, 0] * 1e3, samples_np[:, 1]]
    priors = [
        (meta["params"]["prior_tau_pivot_s"][0] * 1e3,
         meta["params"]["prior_tau_pivot_s"][1] * 1e3),
        (meta["params"]["prior_kappa"][0],
         meta["params"]["prior_kappa"][1]),
    ]
    fnames = ["marginal_tau_pivot.pdf", "marginal_kappa.pdf"]

    for label, arr, (plo, phi), fname in zip(labels, arrs, priors, fnames):
        fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)

        med        = float(np.median(arr))
        lo68, hi68 = np.percentile(arr, [16.0, 84.0])
        lo95, hi95 = np.percentile(arr, [2.5,  97.5])

        ax.hist(arr, bins=np.linspace(plo, phi, 80), density=True,
                histtype="step", linewidth=2.5)
        ax.axvline(med, linestyle="--", linewidth=2,
                   label=f"Median = {med:.3f}")
        ax.axvspan(lo68, hi68, alpha=0.22,
                   label=f"68% CI [{lo68:.3f}, {hi68:.3f}]")
        ax.axvspan(lo95, hi95, alpha=0.10,
                   label=f"95% CI [{lo95:.3f}, {hi95:.3f}]")

        ax.set_xlabel(label, fontsize=18)
        ax.set_ylabel("Posterior density", fontsize=16)
        ax.tick_params(axis="both", labelsize=12)
        ax.set_xlim(plo, phi)

        out = os.path.join(outdir, fname)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote: {out}")


def make_ensemble_plot(samples_np: np.ndarray, meta: dict, outdir: str):
    """
    Posterior ensemble plot: N_POST_CURVES draws from the posterior,
    each rendered as a density-per-dex curve on a log10(tau) axis.
    """
    rng    = np.random.default_rng(POST_RNG_SEED)
    n_show = min(N_POST_CURVES, len(samples_np))
    idx    = rng.choice(len(samples_np), size=n_show, replace=False)
    draws  = samples_np[idx]   # shape (n_show, 2)

    xgrid = np.logspace(np.log10(TAU_MIN_S), np.log10(TAU_MAX_S), 2000)

    fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)

    for row in draws:
        tp, k = float(row[0]), float(row[1])
        try:
            model = LognormalPivotPowerLaw(kappa=k, tau_pivot=tp)
        except (ValueError, RuntimeError):
            continue
        yd = model.density_per_dex(xgrid)
        ax.plot(xgrid, yd, linewidth=1.5, alpha=0.05, color="C0")

    ax.set_xscale("log")
    ax.set_xlabel(r"Scattering timescale $\tau_{600\,\rm MHz}$ (s)", fontsize=14)
    ax.set_ylabel("Density per dex (arbitrary normalization)", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(TAU_MIN_S, TAU_MAX_S)

    out = os.path.join(outdir, "posterior_ensemble.pdf")
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
        x_cat_ms = load_catalog_scattering(CAT2_JSON)
        print(f"Catalog scattering times: n={len(x_cat_ms)}")
        samples_np, meta = run_sbi(pools, x_cat_ms, OUTDIR, RNG_SEED)
    else:
        print("Stage 2 skipped (RUN_SBI=False); loading existing posterior...")
        samples_np = np.load(os.path.join(OUTDIR, "posterior_2param_samples.npy"))
        with open(os.path.join(OUTDIR, "posterior_2param.json")) as f:
            meta = json.load(f)

    # --- Stage 3: plots ---
    if RUN_PLOTS:
        print("=" * 60)
        print("Stage 3 – Plotting")
        print("=" * 60)
        make_corner_plot(samples_np, meta, OUTDIR)
        make_marginal_1d_plots(samples_np, meta, OUTDIR)
        make_ensemble_plot(samples_np, meta, OUTDIR)
        print("All plots written.")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()