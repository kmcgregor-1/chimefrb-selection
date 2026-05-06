#!/usr/bin/env python3
"""
bootstrap_beta_hat.py

Compute a *bootstrap confidence interval on* β̂ = argmin_β T(β), where T is a
goodness-of-fit statistic (AD or KS) comparing catalog scattering times to an
importance-resampled model population.

Key idea
--------
Instead of bootstrapping the statistic at each β independently, we bootstrap the
*entire β-selection procedure*:

For b=1..B:
  1) resample the catalog (with replacement) -> x_cat_b
  2) for each β, importance-resample a model scattering sample using p_det weights -> x_mod_b(β)
  3) compute T_b(β) for all β and set β̂_b = argmin_β T_b(β)

Then take percentiles of {β̂_b} to get a CI.

Practical implementation
------------------------
To avoid re-simulating the intrinsic population inside each bootstrap replicate,
we precompute (and cache to disk) for each β:
  - x_pool(β): model scattering times (ms at 1000 MHz)
  - w_pool(β): p_det weights for those events
This is the only thing bootstrapping needs.

You can control the pool size with N_POOL_PER_BETA; it does NOT need to be 1e6.

Outputs
-------
Writes to OUTDIR:
  - beta_hat_bootstrap.json
  - beta_hat_hist.png
  - beta_hat_curves.png   (median curve with 68%/95% bands)

Usage
-----
Edit USER PARAMS below, then run:
  python bootstrap_beta_hat.py

Notes
-----
- This uses the same LognormalSemiLogTail and fiducial draw parameters you pasted.
- Catalog scattering assumed at 400 MHz (seconds) and scaled to 1000 MHz (ms)
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, anderson_ksamp, ks_2samp
from tqdm import tqdm

import chimefrb_selection as cfsf

BETAS = np.unique(np.concatenate((
    np.linspace(-0.35, 0.40, 76),  # spacing ~0.01 in the core
    np.linspace(-0.5, 0.5, 31),    # fine grid
    np.linspace(-2.0, 2.0, 21),    # coarse grid
)))

# Pool size PER beta used for the cached model pools (x_pool, w_pool).
# This is the main accuracy/performance knob for β̂ bootstrap.
N_POOL_PER_BETA = 1_000_000

# Model resample size per bootstrap replicate (per beta):
# n_resamp = max(50, round(N_CAT * N_RESAMPLE_MULT))
N_RESAMPLE_MULT = 100.0

# Bootstrap replicates for β̂
N_BOOT_BETAHAT = 10_000

RNG_SEED = 12346

CAT2_JSON = "/data/user-data/ssiegel/catalog2/table/20251121/chimefrbcat2.json"

OUTDIR = "/data/user-data/kmcgregor/bootstrap_downturn"
CACHE_SUBDIR = "beta_hat_cache"  # caches per-beta pools to disk

SNR_CUT = 12.0
DM_LO, DM_HI = 100.0, 5000.0

# Statistic to use for choosing beta_hat: "AD" or "KS"
STATISTIC = "AD"

# Whether to compute stats on log10(tau) (recommended for scattering)
LOG_SPACE = True

# Scattering floor at 600 MHz for fiducial draws (mask-only), as in your script
TAU_MIN_S_600 = 1e-4  # 0.1 ms at 600 MHz

# ===========================


# ---------------------------
# Scattering model: LognormalSemiLogTail (same as your script)
# ---------------------------
class LognormalSemiLogTail:
    """
    Lognormal up to the log-binned peak + log-power tail after the peak.

    beta == tail_k:
      - 0.0  => flat plateau in log-binned counts (log-uniform tail; f ∝ 1/x)
      - <0   => downward-sloping plateau
      - >0   => upward-sloping plateau

    tau is in SECONDS throughout (rvs returns seconds) at 600 MHz.
    """

    def __init__(
        self,
        beta: float = 0.0,
        *,
        sigma: float = 1.94,
        scale: float = 0.0131,   # seconds (SciPy lognorm scale), at 600 MHz
        tau_min: float = 1e-5,   # seconds
        tau_max: float = 1.0,    # seconds
    ):
        self.tail_k = float(beta)
        self.sigma  = float(sigma)
        self.scale  = float(scale)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)

        if not (np.isfinite(self.tail_k) and np.isfinite(self.sigma) and np.isfinite(self.scale)):
            raise ValueError("tail_k, sigma, scale must be finite.")
        if self.sigma <= 0:
            raise ValueError("sigma must be > 0.")
        if not (self.scale > 0 and self.tau_min > 0 and self.tau_max > self.tau_min):
            raise ValueError("Need scale>0 and 0<tau_min<tau_max.")

        self._ln   = lognorm(s=self.sigma, scale=self.scale)
        self.pivot = float(np.clip(self.scale, self.tau_min, self.tau_max))
        self._w_left = self._compute_w_left()

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
            right_mass = (A / self.tail_k) * (self.tau_max**self.tail_k - self.pivot**self.tail_k)
            right_mass = float(max(right_mass, 0.0))

        total = left_mass + right_mass
        if not (np.isfinite(total) and total > 0):
            raise RuntimeError("Total mass is non-positive; check parameters/bounds.")
        return left_mass / total

    def rvs(self, size: int = 1, random_state=None) -> np.ndarray:
        n   = int(size)
        rng = np.random.default_rng(random_state)

        u     = rng.uniform(0.0, 1.0, size=n)
        left  = u < self._w_left
        n_left  = int(np.sum(left))
        n_right = n - n_left
        out   = np.empty(n, dtype=float)

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
                xk         = self.pivot**self.tail_k + u_right * (self.tau_max**self.tail_k - self.pivot**self.tail_k)
                out[~left] = np.power(xk, 1.0 / self.tail_k)

        return np.clip(out, self.tau_min, self.tau_max)


# ---------------------------
# Selection function
# ---------------------------
def build_selection_function():
    return cfsf.SelectionFunction(
        predictor_names=["fluence", "scattering_time", "width", "dm"],
        degree=3,
        snr_cut=12.0,
        exclude_sidelobes=True,
        sidelobe_cut=5.0,
        reweighted=False,
    )


# ---------------------------
# Fiducial samplers
# ---------------------------
def draw_fluence_powerlaw_truncated(N, alpha_tail, xmin, xmax, rng):
    if not (xmin > 0 and xmax > xmin):
        raise ValueError("Need xmin > 0 and xmax > xmin.")
    if not (alpha_tail < 0):
        raise ValueError("alpha_tail must be < 0 for this convention.")
    s_max = (xmax / xmin) ** alpha_tail
    u = rng.uniform(s_max, 1.0, size=N)
    return xmin * (u ** (1.0 / alpha_tail))


def sample_truncated_lognorm_icdf(N, sigma_ln, scale, lo, hi, rng):
    if not (sigma_ln > 0 and scale > 0 and lo > 0 and hi > lo):
        raise ValueError("Need sigma_ln>0, scale>0, lo>0, hi>lo.")
    dist = lognorm(s=sigma_ln, scale=scale)
    c_lo = dist.cdf(lo)
    c_hi = dist.cdf(hi)
    if not (np.isfinite(c_lo) and np.isfinite(c_hi) and c_hi > c_lo):
        raise RuntimeError("Bad truncation window for this lognormal.")
    u = rng.uniform(c_lo, c_hi, size=N)
    return np.clip(dist.ppf(u), lo, hi)


def sample_fiducial_from_params(N: int, rng: np.random.Generator, beta: float):
    alpha_tail  = -1.203108427072943
    dm_shape    = 0.6076516890745998
    dm_scale    = 534.4727066208081
    width_shape = 1.1012853240184415
    width_scale = 0.0007389903164298552
    FMIN, FMAX  = 0.1, 10000.0

    fluence_jy_ms = draw_fluence_powerlaw_truncated(N, alpha_tail, FMIN, FMAX, rng)
    dm_pc_cm3     = sample_truncated_lognorm_icdf(N, dm_shape, dm_scale, DM_LO, DM_HI, rng)
    width_s       = sample_truncated_lognorm_icdf(N, width_shape, width_scale, 1e-4, 0.2, rng)

    scat_model = LognormalSemiLogTail(beta=beta)
    tau_s_600  = scat_model.rvs(size=N, random_state=rng)

    m_tau = np.isfinite(tau_s_600) & (tau_s_600 >= TAU_MIN_S_600)

    fluence_jy_ms = fluence_jy_ms[m_tau]
    tau_s_600     = tau_s_600[m_tau]
    width_s       = width_s[m_tau]
    dm_pc_cm3     = dm_pc_cm3[m_tau]

    tau_s_1000 = tau_s_600 * (600.0 / 1000.0) ** 4
    return fluence_jy_ms, tau_s_600, tau_s_1000, width_s, dm_pc_cm3


def compute_pdet(sf, fluence_jy_ms, tau_s_1000, width_s, dm_pc_cm3):
    burst_properties = {
        "fluence_jy_ms":  np.asarray(fluence_jy_ms, float),
        "tau_1_ghz_ms":   np.asarray(tau_s_1000, float) * 1e3,
        "pulse_width_ms": np.asarray(width_s, float) * 1e3,
        "dm":             np.asarray(dm_pc_cm3, float),
    }
    try:
        pdet, _ = sf.calculate_selection_probability(burst_properties, return_std=True)
    except TypeError:
        pdet = sf.calculate_selection_probability(burst_properties)

    pdet = np.asarray(pdet, float)
    valid = np.isfinite(pdet)
    pdet[valid] = np.clip(pdet[valid], 0.0, 1.0)
    return pdet, valid


# ---------------------------
# Catalog loading
# ---------------------------
def load_cat2_catalog_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


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


def minimal_catalog_cuts(cat, snr_cut=SNR_CUT, dm_lo=DM_LO, dm_hi=DM_HI):
    out = []
    for frb in cat:
        snr = to_float_or_nan(frb.get("bonsai_snr", np.nan))
        dm  = to_float_or_nan(frb.get("dm_fitb", np.nan))
        if not (np.isfinite(snr) and snr >= snr_cut):
            continue
        if not (np.isfinite(dm) and dm_lo <= dm <= dm_hi):
            continue
        out.append(frb)
    return out


def pick_first_existing_key(d: dict, keys: list):
    for k in keys:
        if k in d:
            return k
    return None


def extract_catalog_scattering_ms_1000(cat_cut: List[dict]) -> np.ndarray:
    """
    Your earlier script treats catalog scattering as being at 400 MHz (seconds),
    then scales to 1000 MHz and converts to ms.
    """
    if len(cat_cut) == 0:
        return np.array([], float)

    scat_key = pick_first_existing_key(cat_cut[0], ["scat_time", "scattering_time", "tau"])
    scat_s = np.array([to_float_or_nan(frb.get(scat_key, np.nan)) for frb in cat_cut], float) if scat_key else np.full(len(cat_cut), np.nan)

    # heuristic unit fix consistent with your script (if median > 0.5 -> ms -> s)
    if np.nanmedian(scat_s) > 0.5:
        scat_s *= 1e-3

    tau_ms_1000 = scat_s * (400.0 / 1000.0) ** 4 * 1e3
    m = np.isfinite(tau_ms_1000) & (tau_ms_1000 > 0)
    return tau_ms_1000[m]


# ---------------------------
# Statistics + weighted resample
# ---------------------------
def weighted_resample(rng: np.random.Generator, x: np.ndarray, w: np.ndarray, size: int) -> np.ndarray:
    w = np.clip(np.asarray(w, float), 0.0, np.inf)
    s = np.sum(w)
    if not (np.isfinite(s) and s > 0):
        raise RuntimeError("Non-positive weight sum in weighted_resample.")
    idx = rng.choice(x.size, size=int(size), replace=True, p=w / s)
    return x[idx]


def ad_statistic(x_cat: np.ndarray, x_model: np.ndarray, *, log_space: bool = True) -> float:
    x1 = np.asarray(x_cat, float)
    x2 = np.asarray(x_model, float)
    x1 = x1[np.isfinite(x1) & (x1 > 0)]
    x2 = x2[np.isfinite(x2) & (x2 > 0)]
    if x1.size < 10 or x2.size < 10:
        return np.inf
    if log_space:
        x1 = np.log10(x1)
        x2 = np.log10(x2)
    res = anderson_ksamp([x1, x2], midrank=True)
    return float(res.statistic)


def ks_statistic(x_cat: np.ndarray, x_model: np.ndarray, *, log_space: bool = True) -> float:
    x1 = np.asarray(x_cat, float)
    x2 = np.asarray(x_model, float)
    x1 = x1[np.isfinite(x1) & (x1 > 0)]
    x2 = x2[np.isfinite(x2) & (x2 > 0)]
    if x1.size < 10 or x2.size < 10:
        return np.inf
    if log_space:
        x1 = np.log10(x1)
        x2 = np.log10(x2)
    res = ks_2samp(x1, x2, alternative="two-sided")
    return float(res.statistic)


# ---------------------------
# Pool caching per beta
# ---------------------------
def _beta_tag(beta: float) -> str:
    return "beta" + f"{beta:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")


def cache_path_for_beta(cache_dir: str, beta: float) -> str:
    return os.path.join(cache_dir, f"pool_{_beta_tag(beta)}.npz")


def build_or_load_pool(
    *,
    sf,
    rng: np.random.Generator,
    beta: float,
    n_pool: int,
    cache_dir: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      x_pool_ms_1000 : scattering times (ms @ 1000 MHz)
      w_pool         : p_det weights
    """
    os.makedirs(cache_dir, exist_ok=True)
    pth = cache_path_for_beta(cache_dir, beta)

    if os.path.exists(pth):
        dat = np.load(pth)
        x = dat["x_pool_ms_1000"].astype(float)
        w = dat["w_pool"].astype(float)
        return x, w

    # Simulate until we have n_pool valid weighted entries
    x_acc = []
    w_acc = []

    # We may lose some to invalid pdet; oversample in chunks.
    chunk = min(200_000, max(50_000, n_pool))
    while sum(len(a) for a in x_acc) < n_pool:
        flu, tau600, tau1000, wid, dm = sample_fiducial_from_params(chunk, rng, beta=beta)
        pdet, valid = compute_pdet(sf, flu, tau1000, wid, dm)

        x_ms_1000 = np.asarray(tau600, float) * (600.0 / 1000.0) ** 4 * 1e3

        m = np.isfinite(x_ms_1000) & (x_ms_1000 > 0) & valid & np.isfinite(pdet) & (pdet >= 0.0)
        if np.any(m):
            x_acc.append(x_ms_1000[m])
            w_acc.append(pdet[m])

    x = np.concatenate(x_acc)[:n_pool]
    w = np.concatenate(w_acc)[:n_pool]

    np.savez_compressed(pth, x_pool_ms_1000=x.astype(np.float32), w_pool=w.astype(np.float32))
    return x.astype(float), w.astype(float)


# ---------------------------
# Bootstrap β-hat
# ---------------------------
@dataclass
class BetaHatBootstrapResult:
    beta_hat: np.ndarray
    stat_matrix: np.ndarray  # shape (n_boot, n_beta)
    betas: np.ndarray


def bootstrap_beta_hat(
    rng: np.random.Generator,
    *,
    x_cat: np.ndarray,
    betas: np.ndarray,
    pools: Dict[float, Tuple[np.ndarray, np.ndarray]],
    n_resample_mult: float,
    n_boot: int,
    statistic: str,
    log_space: bool,
) -> BetaHatBootstrapResult:
    x_cat = np.asarray(x_cat, float)
    x_cat = x_cat[np.isfinite(x_cat) & (x_cat > 0)]
    n_cat = int(x_cat.size)
    if n_cat < 10:
        raise RuntimeError("Too few catalog points after cuts to bootstrap beta_hat.")

    n_beta = betas.size
    stat_mat = np.full((n_boot, n_beta), np.nan, float)
    beta_hat = np.full(n_boot, np.nan, float)

    stat_fn = ad_statistic if statistic.upper() == "AD" else ks_statistic

    for b in tqdm(range(n_boot), desc="Bootstrapping β̂"):
        # 1) bootstrap the catalog
        x_cat_b = rng.choice(x_cat, size=n_cat, replace=True)

        # 2) compute T_b(beta) for all betas using cached pools
        for j, beta in enumerate(betas):
            x_pool, w_pool = pools[float(beta)]
            n_resamp = int(max(50, round(n_resample_mult * n_cat)))
            x_mod_b = weighted_resample(rng, x_pool, w_pool, size=n_resamp)
            stat_mat[b, j] = stat_fn(x_cat_b, x_mod_b, log_space=log_space)

        # 3) argmin
        j0 = int(np.nanargmin(stat_mat[b]))
        beta_hat[b] = float(betas[j0])

    return BetaHatBootstrapResult(beta_hat=beta_hat, stat_matrix=stat_mat, betas=betas)


# ---------------------------
# Plotting + saving
# ---------------------------
def percentile_band(arr: np.ndarray, lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    return np.nanpercentile(arr, lo, axis=0), np.nanpercentile(arr, hi, axis=0)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    cache_dir = os.path.join(OUTDIR, CACHE_SUBDIR)
    os.makedirs(cache_dir, exist_ok=True)

    rng = np.random.default_rng(RNG_SEED)

    # Load catalog scattering
    cat = load_cat2_catalog_json(CAT2_JSON)
    cat_cut = minimal_catalog_cuts(cat, snr_cut=SNR_CUT, dm_lo=DM_LO, dm_hi=DM_HI)
    x_cat = extract_catalog_scattering_ms_1000(cat_cut)
    if x_cat.size < 10:
        raise RuntimeError(f"Too few catalog scattering points after cuts: n={x_cat.size}")

    # Build SF
    sf = build_selection_function()

    # Build/load pools per beta
    pools: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    for beta in tqdm(BETAS, desc="Building/loading model pools (per β)"):
        # Make beta-specific RNG stream that is deterministic per beta
        beta_rng = np.random.default_rng(int(RNG_SEED + 1_000_000 * (float(beta) ** 2 + 1.0)))
        x_pool, w_pool = build_or_load_pool(
            sf=sf, rng=beta_rng, beta=float(beta), n_pool=int(N_POOL_PER_BETA), cache_dir=cache_dir
        )
        pools[float(beta)] = (x_pool, w_pool)
        print(f"Pool ready: beta={beta:+.3f}  n={x_pool.size}")

    print("\nCache build complete. Exiting.")
    return


if __name__ == "__main__":
    main()