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
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional

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
N_TAU_PIVOT_GRID = 15    # grid points along tau_pivot axis
N_KAPPA_GRID     = 15    # grid points along kappa axis
N_POOL_PER_CELL  = 1_000_000
N_CACHE_WORKERS   = 4        # parallel cache workers
CACHE_CHUNK_SIZE   = 10_000     # number of intrinsic draws per selection-function call

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
RUN_CACHE       = True
RUN_SBI         = True
RUN_PLOTS       = True
RUN_DIAGNOSTICS = True

# ---------- Diagnostics control ----------
# These diagnostics are designed to run even when RUN_CACHE=False and/or
# RUN_SBI=False, as long as the required saved cache/posterior files exist.
RUN_ESS_DIAGNOSTIC      = True
RUN_GRID_DIAGNOSTIC     = True
RUN_PPC_DIAGNOSTIC      = True
RUN_SBC_DIAGNOSTIC      = True   # requires RUN_SBI=True because the posterior object is in memory

N_GRID_DIAG_DRAWS       = 10_000
N_PPC_POSTERIOR_DRAWS   = 200
N_SBC_DRAWS             = 500
N_SBC_POSTERIOR_SAMPLES = 1_000

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
    chunk: int = CACHE_CHUNK_SIZE,
    progress_queue: Optional[object] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (x_pool_ms_1000, w_pool), building from simulation if not cached.

    If `progress_queue` is supplied, this function does not create its own tqdm
    bar. Instead it emits lightweight progress messages so the parent process can
    update shared progress bars cleanly during parallel cache generation.
    """
    os.makedirs(cache_dir, exist_ok=True)
    pth = cache_path(cache_dir, tau_pivot, kappa)

    key_label = f"tp={tau_pivot*1e3:.1f} ms, k={kappa:+.3f}"

    if os.path.exists(pth):
        if verbose:
            print(f"[cache hit] {key_label}")
        dat = np.load(pth)
        x = dat["x_pool_ms_1000"].astype(float)
        w = dat["w_pool"].astype(float)
        if progress_queue is not None:
            progress_queue.put({
                "type": "hit",
                "tau_pivot": float(tau_pivot),
                "kappa": float(kappa),
                "n": int(len(x)),
                "mean_w": float(np.nanmean(w)),
                "nonzero_frac": float(np.mean(w > 0)),
                "path": pth,
            })
        return x, w

    if verbose:
        print(f"[building]  {key_label}")

    if progress_queue is not None:
        progress_queue.put({
            "type": "start",
            "tau_pivot": float(tau_pivot),
            "kappa": float(kappa),
            "n_pool": int(n_pool),
        })

    x_acc: List[np.ndarray] = []
    w_acc: List[np.ndarray] = []

    n_acc = 0
    n_draw = 0
    last_emit_time = time.time()

    # In serial mode, keep the familiar per-cell progress bar. In parallel mode,
    # the parent process owns the progress bars to avoid garbled output.
    pbar = None
    if progress_queue is None:
        pbar = tqdm(
            total=n_pool,
            desc=f"cell tp={tau_pivot*1e3:.1f} ms, k={kappa:+.3f}",
            leave=True,
            colour="green",
        )

    try:
        while n_acc < n_pool:
            flu, tau_600, tau_1000, wid, dm = sample_fiducial(
                chunk, rng, kappa, tau_pivot
            )

            pdet, valid = compute_pdet(sf, flu, tau_1000, wid, dm)

            # Convert tau_600 seconds to tau_1000 MHz in ms for storage.
            x_ms_1000 = np.asarray(tau_600, float) * (600.0 / 1000.0) ** 4 * 1e3

            m = (
                np.isfinite(x_ms_1000) & (x_ms_1000 > 0) &
                valid & np.isfinite(pdet) & (pdet >= 0.0)
            )

            n_good = int(np.sum(m))
            n_draw += len(tau_600)

            n_new = 0
            if n_good > 0:
                remaining = n_pool - n_acc
                n_new = min(n_good, remaining)

                x_acc.append(x_ms_1000[m][:n_new])
                w_acc.append(pdet[m][:n_new])
                n_acc += n_new

            mean_pdet = np.nanmean(pdet[m]) if n_good > 0 else np.nan

            if pbar is not None and n_new > 0:
                pbar.update(n_new)
                pbar.set_postfix(
                    drawn=n_draw,
                    valid=n_acc,
                    keep_frac=f"{n_acc / max(n_draw, 1):.3f}",
                    mean_pdet=f"{mean_pdet:.3g}" if np.isfinite(mean_pdet) else "nan",
                )

            if progress_queue is not None:
                now = time.time()
                if n_new > 0:
                    progress_queue.put({
                        "type": "progress",
                        "tau_pivot": float(tau_pivot),
                        "kappa": float(kappa),
                        "accepted_inc": int(n_new),
                        "accepted": int(n_acc),
                        "drawn": int(n_draw),
                        "mean_pdet": float(mean_pdet) if np.isfinite(mean_pdet) else None,
                    })
                elif now - last_emit_time > 5:
                    progress_queue.put({
                        "type": "heartbeat",
                        "tau_pivot": float(tau_pivot),
                        "kappa": float(kappa),
                        "accepted": int(n_acc),
                        "drawn": int(n_draw),
                    })
                    last_emit_time = now

    finally:
        if pbar is not None:
            pbar.close()

    if not x_acc:
        raise RuntimeError(f"No valid simulated events for {key_label}.")

    x = np.concatenate(x_acc)[:n_pool]
    w = np.concatenate(w_acc)[:n_pool]

    np.savez_compressed(
        pth,
        x_pool_ms_1000=x.astype(np.float32),
        w_pool=w.astype(np.float32),
    )

    if verbose:
        print(
            f"  Saved: {os.path.basename(pth)}  "
            f"(n={len(x)}, mean_pdet={np.nanmean(w):.3g}, "
            f"nonzero_pdet_frac={np.mean(w > 0):.3f})"
        )

    if progress_queue is not None:
        progress_queue.put({
            "type": "saved",
            "tau_pivot": float(tau_pivot),
            "kappa": float(kappa),
            "n": int(len(x)),
            "mean_w": float(np.nanmean(w)),
            "nonzero_frac": float(np.mean(w > 0)),
            "path": pth,
        })

    return x.astype(float), w.astype(float)


def _cache_worker(args) -> dict:
    """
    Worker for one cache cell.

    The worker builds its own SelectionFunction because that object may not
    pickle cleanly across processes. The large pool arrays are written to disk
    inside the worker; only lightweight metadata are returned to the parent.
    """
    tp, k, cache_dir, rng_seed, progress_queue = args

    seed = int(
        rng_seed
        + 1_000_003 * abs(hash((round(tp, 6), round(k, 4)))) % 2**31
    )
    cell_rng = np.random.default_rng(seed)
    sf = build_selection_function()

    pth = cache_path(cache_dir, tp, k)
    existed_before = os.path.exists(pth)

    x, w = build_or_load_cell(
        sf=sf,
        rng=cell_rng,
        tau_pivot=tp,
        kappa=k,
        n_pool=N_POOL_PER_CELL,
        cache_dir=cache_dir,
        chunk=CACHE_CHUNK_SIZE,
        progress_queue=progress_queue,
        verbose=False,
    )

    return {
        "key": (round(tp, 8), round(k, 6)),
        "tau_pivot": float(tp),
        "kappa": float(k),
        "path": pth,
        "n": int(len(x)),
        "mean_w": float(np.nanmean(w)),
        "nonzero_frac": float(np.mean(w > 0)),
        "existed_before": bool(existed_before),
    }


def _drain_cache_progress_queue(
    queue,
    *,
    sample_pbar,
    cell_pbar,
    active_cells: dict,
    completed_meta: list,
):
    """Drain multiprocessing progress messages and update tqdm bars."""
    while True:
        try:
            msg = queue.get_nowait()
        except Exception:
            break

        mtype = msg.get("type")
        tp = msg.get("tau_pivot")
        k = msg.get("kappa")
        label = None
        if tp is not None and k is not None:
            label = f"tp={tp*1e3:.1f} ms, k={k:+.3f}"

        if mtype == "start":
            active_cells[(round(tp, 8), round(k, 6))] = {"label": label, "accepted": 0, "drawn": 0}
            sample_pbar.set_postfix_str(f"building {label}")

        elif mtype == "progress":
            inc = int(msg.get("accepted_inc", 0))
            if inc > 0:
                sample_pbar.update(inc)
            key = (round(tp, 8), round(k, 6))
            active_cells[key] = {
                "label": label,
                "accepted": int(msg.get("accepted", 0)),
                "drawn": int(msg.get("drawn", 0)),
            }
            mean_pdet = msg.get("mean_pdet")
            if mean_pdet is None:
                sample_pbar.set_postfix_str(f"{label}; accepted={msg.get('accepted', 0)}")
            else:
                sample_pbar.set_postfix_str(
                    f"{label}; accepted={msg.get('accepted', 0)}; mean_pdet={mean_pdet:.3g}"
                )

        elif mtype == "heartbeat":
            key = (round(tp, 8), round(k, 6))
            active_cells[key] = {
                "label": label,
                "accepted": int(msg.get("accepted", 0)),
                "drawn": int(msg.get("drawn", 0)),
            }
            sample_pbar.set_postfix_str(
                f"{label}; accepted={msg.get('accepted', 0)}; drawn={msg.get('drawn', 0)}"
            )

        elif mtype in {"hit", "saved"}:
            completed_meta.append(msg)
            cell_pbar.update(1)
            if mtype == "hit":
                sample_pbar.update(int(msg.get("n", 0)))
            key = (round(tp, 8), round(k, 6))
            active_cells.pop(key, None)
            cell_pbar.set_postfix_str(
                f"{mtype}: {label}; mean_w={msg.get('mean_w', float('nan')):.3g}"
            )


def build_cache(sf, cache_dir: str, rng_seed: int) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Build/load the full (tau_pivot, kappa) grid cache.

    Parallelized over cache cells. Progress is reported by two parent-owned tqdm
    bars: one for completed cells and one for accepted pool samples across all
    cells. This avoids garbled per-worker progress bars.
    """
    tau_pivot_grid = np.linspace(TAU_PIVOT_MIN_S, TAU_PIVOT_MAX_S, N_TAU_PIVOT_GRID)
    kappa_grid     = np.linspace(KAPPA_MIN, KAPPA_MAX, N_KAPPA_GRID)
    grid           = [(float(tp), float(k)) for tp in tau_pivot_grid for k in kappa_grid]

    os.makedirs(cache_dir, exist_ok=True)

    n_workers = int(max(1, min(N_CACHE_WORKERS, len(grid))))
    total_cells = len(grid)
    total_samples = int(total_cells * N_POOL_PER_CELL)

    print(
        f"\nBuilding/loading {total_cells} cache cells with {n_workers} workers "
        f"({N_POOL_PER_CELL:,} samples per cell; total {total_samples:,})."
    )

    # Serial mode preserves the old per-cell tqdm behavior and avoids multiprocessing
    # overhead when N_CACHE_WORKERS=1.
    if n_workers == 1:
        pools: Dict = {}
        for tp, k in tqdm(grid, desc="Cache cells", colour="blue"):
            seed = int(
                rng_seed
                + 1_000_003 * abs(hash((round(tp, 6), round(k, 4)))) % 2**31
            )
            cell_rng = np.random.default_rng(seed)
            x, w = build_or_load_cell(
                sf=sf,
                rng=cell_rng,
                tau_pivot=tp,
                kappa=k,
                n_pool=N_POOL_PER_CELL,
                cache_dir=cache_dir,
                chunk=CACHE_CHUNK_SIZE,
                progress_queue=None,
                verbose=True,
            )
            pools[(round(tp, 8), round(k, 6))] = (x, w)
        return pools, tau_pivot_grid, kappa_grid

    manager = mp.Manager()
    progress_queue = manager.Queue()
    worker_args = [
        (tp, k, cache_dir, int(rng_seed), progress_queue)
        for tp, k in grid
    ]

    active_cells = {}
    completed_meta = []

    # Use fork on Linux clusters when available; fall back to the default context.
    try:
        mp_context = mp.get_context("fork")
    except ValueError:
        mp_context = None

    executor_kwargs = {"max_workers": n_workers}
    if mp_context is not None:
        executor_kwargs["mp_context"] = mp_context

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        futures = [executor.submit(_cache_worker, a) for a in worker_args]

        with tqdm(total=total_cells, desc="Cache cells complete", colour="blue", position=0) as cell_pbar, \
             tqdm(total=total_samples, desc="Accepted pool samples", colour="green", position=1) as sample_pbar:

            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)

                _drain_cache_progress_queue(
                    progress_queue,
                    sample_pbar=sample_pbar,
                    cell_pbar=cell_pbar,
                    active_cells=active_cells,
                    completed_meta=completed_meta,
                )

                for fut in done:
                    # This raises worker exceptions immediately instead of hiding them.
                    meta = fut.result()
                    key = meta["key"]
                    cell_pbar.set_postfix_str(
                        f"done: tp={key[0]*1e3:.1f} ms, k={key[1]:+.3f}; "
                        f"mean_w={meta['mean_w']:.3g}"
                    )

            # Drain any final messages emitted just before worker shutdown.
            _drain_cache_progress_queue(
                progress_queue,
                sample_pbar=sample_pbar,
                cell_pbar=cell_pbar,
                active_cells=active_cells,
                completed_meta=completed_meta,
            )

    # Load arrays from disk after workers finish. This avoids pickling very large
    # arrays from worker processes back to the parent.
    pools = load_cache(cache_dir)
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
            ) -> Tuple[np.ndarray, dict, object]:
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


    return samples_np, result, posterior


# ────────────────────────────────────────────────────────────
# Section 7  –  Diagnostics
# ────────────────────────────────────────────────────────────

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


def diagnose_pools(pools: Dict, outdir: str) -> List[dict]:
    """Compute effective sample size diagnostics for every cache cell."""
    print("\n" + "=" * 60)
    print("Diagnostic 1 – Weighted-pool effective sample size")
    print("=" * 60)

    rows: List[dict] = []
    for (tp, k), (x, w) in sorted(pools.items()):
        n_raw = int(len(w))
        n_eff = effective_n(w)
        eff_frac = n_eff / max(n_raw, 1)
        nonzero_frac = float(np.mean(np.asarray(w) > 0)) if n_raw > 0 else 0.0
        mean_w = float(np.nanmean(w)) if n_raw > 0 else np.nan

        row = dict(
            tau_pivot_s=float(tp),
            tau_pivot_ms=float(tp) * 1e3,
            kappa=float(k),
            n_raw=n_raw,
            n_eff=float(n_eff),
            eff_frac=float(eff_frac),
            nonzero_weight_frac=nonzero_frac,
            mean_weight=mean_w,
        )
        rows.append(row)

        print(
            f"tp={tp*1e3:6.2f} ms  k={k:+6.3f}  "
            f"N_eff={n_eff:10.0f}  ({100*eff_frac:6.2f}%)  "
            f"nonzero={100*nonzero_frac:6.2f}%  mean_w={mean_w:.3g}"
        )

    if rows:
        n_eff_all = np.array([r["n_eff"] for r in rows], dtype=float)
        eff_frac_all = np.array([r["eff_frac"] for r in rows], dtype=float)
        print("\nESS summary:")
        print(f"  min N_eff:       {np.nanmin(n_eff_all):.0f}")
        print(f"  median N_eff:    {np.nanmedian(n_eff_all):.0f}")
        print(f"  min ESS frac:    {100*np.nanmin(eff_frac_all):.2f}%")
        print(f"  median ESS frac: {100*np.nanmedian(eff_frac_all):.2f}%")

        low = [r for r in rows if r["eff_frac"] < 0.05]
        if low:
            print(
                f"\nWARNING: {len(low)} cache cells have N_eff/N_raw < 5%. "
                "Those cells are candidates for increasing N_POOL_PER_CELL."
            )
        else:
            print("\nNo cache cells have N_eff/N_raw < 5%.")

    os.makedirs(outdir, exist_ok=True)
    out_json = os.path.join(outdir, "diagnostic_pool_ess.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote: {out_json}")

    return rows


def diagnose_grid_density(pools: Dict, outdir: str, rng_seed: int, n_draws: int = N_GRID_DIAG_DRAWS) -> dict:
    """Check nearest-cache-cell distance for random prior draws."""
    print("\n" + "=" * 60)
    print("Diagnostic 2 – Nearest-neighbor cache-grid distance")
    print("=" * 60)

    keys = np.array(list(pools.keys()), dtype=float)
    if keys.ndim != 2 or keys.shape[0] == 0:
        raise RuntimeError("No cache keys available for grid-density diagnostic.")

    ranges = keys.max(axis=0) - keys.min(axis=0)
    ranges[ranges == 0] = 1.0

    tp_lo = max(TAU_PIVOT_MIN_S, float(keys[:, 0].min()))
    tp_hi = min(TAU_PIVOT_MAX_S, float(keys[:, 0].max()))
    k_lo  = max(KAPPA_MIN,       float(keys[:, 1].min()))
    k_hi  = min(KAPPA_MAX,       float(keys[:, 1].max()))

    rng = np.random.default_rng(rng_seed)
    dists = np.empty(int(n_draws), dtype=float)
    for i in range(int(n_draws)):
        tp = rng.uniform(tp_lo, tp_hi)
        k  = rng.uniform(k_lo, k_hi)
        q  = np.array([tp, k], dtype=float)
        dists[i] = np.sqrt(np.sum(((keys - q) / ranges) ** 2, axis=1)).min()

    result = dict(
        n_draws=int(n_draws),
        tau_pivot_prior_s=[float(tp_lo), float(tp_hi)],
        kappa_prior=[float(k_lo), float(k_hi)],
        median_nn_dist=float(np.median(dists)),
        p95_nn_dist=float(np.percentile(dists, 95.0)),
        max_nn_dist=float(np.max(dists)),
    )

    print(f"Median NN dist: {result['median_nn_dist']:.4f}")
    print(f"95th pct:       {result['p95_nn_dist']:.4f}")
    print(f"Max:            {result['max_nn_dist']:.4f}")

    if result["p95_nn_dist"] > 0.05:
        print(
            "WARNING: 95th-percentile normalized NN distance is > 0.05. "
            "If SBC or PPC also show bias, increase the grid density."
        )
    else:
        print("Grid spacing looks reasonable by the 95th-percentile < 0.05 heuristic.")

    out_json = os.path.join(outdir, "diagnostic_grid_density.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote: {out_json}")

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    ax.hist(dists, bins=50, histtype="step", linewidth=2)
    ax.axvline(result["median_nn_dist"], linestyle="--", linewidth=1.5, label="median")
    ax.axvline(result["p95_nn_dist"], linestyle=":", linewidth=2.0, label="95th pct")
    ax.set_xlabel("Nearest-cache distance, normalized parameter units", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(fontsize=10)
    out_pdf = os.path.join(outdir, "diagnostic_grid_density.pdf")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_pdf}")

    return result


def posterior_predictive_check(
    samples_np: np.ndarray,
    pools: Dict,
    x_cat_ms: np.ndarray,
    outdir: str,
    rng_seed: int,
    n_draws: int = N_PPC_POSTERIOR_DRAWS,
) -> dict:
    """Posterior predictive check using posterior samples, cached pools, and catalog data."""
    print("\n" + "=" * 60)
    print("Diagnostic 3 – Posterior predictive check")
    print("=" * 60)

    rng = np.random.default_rng(rng_seed)
    x_cat_ms = np.asarray(x_cat_ms, dtype=float)
    samples_np = np.asarray(samples_np, dtype=float)

    if samples_np.ndim != 2 or samples_np.shape[1] < 2:
        raise RuntimeError("Posterior samples must have shape (N, 2).")

    obs_summary = compute_summary(x_cat_ms)
    n_use = min(int(n_draws), samples_np.shape[0])
    sample_idx = rng.choice(samples_np.shape[0], size=n_use, replace=False)

    ppc_summaries: List[np.ndarray] = []
    n_fail = 0
    for row in tqdm(samples_np[sample_idx], desc="PPC draws", colour="magenta"):
        tp, k = float(row[0]), float(row[1])
        try:
            x_p, w_p = nearest_pool(tp, k, pools)
            x_mock = weighted_resample(rng, x_p, w_p, size=len(x_cat_ms))
            ppc_summaries.append(compute_summary(x_mock))
        except Exception:
            n_fail += 1

    if not ppc_summaries:
        raise RuntimeError("All PPC draws failed. Check cache weights and posterior samples.")

    ppc = np.asarray(ppc_summaries, dtype=float)
    lo = np.percentile(ppc, 2.5, axis=0)
    hi = np.percentile(ppc, 97.5, axis=0)
    med = np.percentile(ppc, 50.0, axis=0)
    inside_95 = (obs_summary >= lo) & (obs_summary <= hi)

    result = dict(
        n_requested=int(n_draws),
        n_success=int(ppc.shape[0]),
        n_failed=int(n_fail),
        n_summary_dim=int(obs_summary.size),
        obs_summary=obs_summary.astype(float).tolist(),
        ppc_median=med.astype(float).tolist(),
        ppc_2p5=lo.astype(float).tolist(),
        ppc_97p5=hi.astype(float).tolist(),
        obs_inside_95=inside_95.astype(bool).tolist(),
        n_inside_95=int(np.sum(inside_95)),
    )

    print(f"Successful PPC draws: {result['n_success']} / {result['n_requested']}")
    print(f"Failed PPC draws:     {result['n_failed']}")
    print(f"Observed summary dims inside 95% PPC interval: {result['n_inside_95']} / {obs_summary.size}")
    bad = np.where(~inside_95)[0]
    if bad.size:
        print(f"WARNING: observed summary is outside the 95% PPC interval for dimensions: {bad.tolist()}")
    else:
        print("Observed summary lies inside the 95% PPC interval for all dimensions.")

    out_json = os.path.join(outdir, "diagnostic_ppc.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote: {out_json}")

    n_dim = ppc.shape[1]
    n_cols = 7
    n_rows = int(np.ceil(n_dim / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.1 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i >= n_dim:
            ax.set_visible(False)
            continue
        ax.hist(ppc[:, i], bins=30, density=True, histtype="stepfilled", alpha=0.45)
        ax.axvline(obs_summary[i], color="k", linewidth=2.0)
        ax.axvline(lo[i], color="k", linestyle=":", linewidth=1.0)
        ax.axvline(hi[i], color="k", linestyle=":", linewidth=1.0)
        ax.set_title(f"summary dim {i}", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    out_pdf = os.path.join(outdir, "diagnostic_ppc.pdf")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_pdf}")

    return result


def run_sbc_diagnostic(
    posterior,
    pools: Dict,
    x_cat_ms: np.ndarray,
    outdir: str,
    rng_seed: int,
    n_sbc: int = N_SBC_DRAWS,
    n_posterior_samples: int = N_SBC_POSTERIOR_SAMPLES,
) -> dict:
    """Optional SBC diagnostic. Requires the trained posterior object in memory."""
    print("\n" + "=" * 60)
    print("Diagnostic 4 – Simulation-based calibration")
    print("=" * 60)

    try:
        from sbi.analysis import run_sbc, check_sbc
    except Exception as exc:
        print(f"Skipping SBC: could not import sbi.analysis.run_sbc/check_sbc ({exc}).")
        return {"skipped": True, "reason": str(exc)}

    rng = np.random.default_rng(rng_seed)
    keys = np.array(list(pools.keys()), dtype=float)
    tp_lo = max(TAU_PIVOT_MIN_S, float(keys[:, 0].min()))
    tp_hi = min(TAU_PIVOT_MAX_S, float(keys[:, 0].max()))
    k_lo  = max(KAPPA_MIN,       float(keys[:, 1].min()))
    k_hi  = min(KAPPA_MAX,       float(keys[:, 1].max()))

    n_cat = int(len(x_cat_ms))
    thetas_sbc: List[List[float]] = []
    xs_sbc: List[np.ndarray] = []
    for _ in tqdm(range(int(n_sbc)), desc="SBC mock catalogs", colour="cyan"):
        tp = rng.uniform(tp_lo, tp_hi)
        k = rng.uniform(k_lo, k_hi)
        x_p, w_p = nearest_pool(tp, k, pools)
        x_mock = weighted_resample(rng, x_p, w_p, size=n_cat)
        thetas_sbc.append([tp, k])
        xs_sbc.append(compute_summary(x_mock))

    thetas_t = torch.tensor(np.array(thetas_sbc, dtype=np.float32))
    xs_t = torch.tensor(np.array(xs_sbc, dtype=np.float32))

    ranks, dap = run_sbc(
        thetas_t,
        xs_t,
        posterior,
        num_posterior_samples=int(n_posterior_samples),
    )
    checks = check_sbc(
        ranks,
        thetas_t,
        dap,
        num_posterior_samples=int(n_posterior_samples),
    )

    ranks_np = np.asarray(ranks)
    result = dict(
        skipped=False,
        n_sbc=int(n_sbc),
        n_posterior_samples=int(n_posterior_samples),
        rank_shape=list(ranks_np.shape),
        rank_mean=np.mean(ranks_np, axis=0).astype(float).tolist(),
        rank_std=np.std(ranks_np, axis=0).astype(float).tolist(),
        check_sbc=str(checks),
    )

    out_json = os.path.join(outdir, "diagnostic_sbc.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote: {out_json}")

    labels = [r"$\tau_{\rm pivot}$", r"$\kappa$"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    for j, ax in enumerate(np.atleast_1d(axes)):
        ax.hist(ranks_np[:, j], bins=30, histtype="step", linewidth=2)
        ax.set_title(labels[j], fontsize=12)
        ax.set_xlabel("SBC rank", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
    out_pdf = os.path.join(outdir, "diagnostic_sbc_ranks.pdf")
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_pdf}")

    return result


# ────────────────────────────────────────────────────────────
# Section 8  –  Plotting
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

    posterior = None
    x_cat_ms = None

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

    # --- Diagnostics that need only the cache ---
    # These run regardless of RUN_CACHE because `pools` exists either way:
    # freshly built above, or loaded from CACHE_DIR.
    if RUN_DIAGNOSTICS:
        if RUN_ESS_DIAGNOSTIC:
            diagnose_pools(pools, OUTDIR)
        if RUN_GRID_DIAGNOSTIC:
            diagnose_grid_density(pools, OUTDIR, RNG_SEED)

    # --- Stage 2: SBI ---
    if RUN_SBI:
        print("=" * 60)
        print("Stage 2 – Neural Posterior Estimation")
        print("=" * 60)
        x_cat_ms = load_catalog_scattering(CAT2_JSON)
        print(f"Catalog scattering times: n={len(x_cat_ms)}")
        samples_np, meta, posterior = run_sbi(pools, x_cat_ms, OUTDIR, RNG_SEED)
    else:
        print("Stage 2 skipped (RUN_SBI=False); loading existing posterior...")
        samples_np = np.load(os.path.join(OUTDIR, "posterior_2param_samples.npy"))
        with open(os.path.join(OUTDIR, "posterior_2param.json")) as f:
            meta = json.load(f)

    # --- Diagnostics that need posterior samples and the catalog ---
    # PPC still works when RUN_SBI=False, because it uses the saved posterior
    # samples and the loaded cache. SBC only runs when RUN_SBI=True because it
    # needs the live posterior object, not just posterior samples.
    if RUN_DIAGNOSTICS:
        need_catalog = RUN_PPC_DIAGNOSTIC or RUN_SBC_DIAGNOSTIC
        if need_catalog and x_cat_ms is None:
            x_cat_ms = load_catalog_scattering(CAT2_JSON)
            print(f"Catalog scattering times: n={len(x_cat_ms)}")

        if RUN_PPC_DIAGNOSTIC:
            posterior_predictive_check(
                samples_np=samples_np,
                pools=pools,
                x_cat_ms=x_cat_ms,
                outdir=OUTDIR,
                rng_seed=RNG_SEED,
                n_draws=N_PPC_POSTERIOR_DRAWS,
            )

        if RUN_SBC_DIAGNOSTIC:
            if posterior is None:
                print(
                    "Skipping SBC diagnostic: RUN_SBI=False, so the trained posterior "
                    "object is not in memory. Set RUN_SBI=True to run SBC."
                )
            else:
                run_sbc_diagnostic(
                    posterior=posterior,
                    pools=pools,
                    x_cat_ms=x_cat_ms,
                    outdir=OUTDIR,
                    rng_seed=RNG_SEED,
                    n_sbc=N_SBC_DRAWS,
                    n_posterior_samples=N_SBC_POSTERIOR_SAMPLES,
                )

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
