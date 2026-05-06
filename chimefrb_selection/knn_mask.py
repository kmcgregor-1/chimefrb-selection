import os
import click
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from .utils import read_json_file, get_injections_detected, cut_detections_nondetections, DEFAULT_LOCAL_DATA_PATH, DEFAULT_LOCAL_INJ_FILE, PROP_ORDER
import pickle

SNR_CUT = 12.0      # SNR cut for detections (CHANGE AS NEEDED)

# --------------------------- KNN Envelope ---------------------------

class KNNEnvelope:
    """
    Shape-agnostic support estimator in log-space using kNN radii.

    Threshold options:
      - quantile: self._thr = quantile(kth_radius, q)
      - sigma:    self._thr = mean(kth_radius) + sigma * std(kth_radius)
    """
    def __init__(self, k=50, quantile=0.99, sigma=None, metric='euclidean'):
        self.k = int(k)
        self.q = None if quantile is None else float(quantile)
        self.sigma = None if sigma is None else float(sigma)
        self.metric = metric

        self._nbrs = None         # sklearn or SciPy backend
        self._thr = None
        self._radii_train = None  # diagnostics
        self._Xtrain_log = None   # <-- store training data for portable pickling
        self._backend = None      # "sklearn" or "ckdtree"

    def fit(self, Xtrain):
        """
        Xtrain: (n_samples, d) in log10-space
        """
        import numpy as np
        Xtrain = np.asarray(Xtrain, dtype=float)
        self._Xtrain_log = Xtrain

        # Prefer sklearn if available, else fallback to SciPy's cKDTree
        try:
            from sklearn.neighbors import NearestNeighbors
            self._nbrs = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
            self._nbrs.fit(Xtrain)
            dists, _ = self._nbrs.kneighbors(Xtrain, n_neighbors=self.k)
            self._backend = "sklearn"
        except Exception:
            from scipy.spatial import cKDTree
            self._nbrs = cKDTree(Xtrain)
            dists, _ = self._nbrs.query(Xtrain, k=self.k)
            if self.k == 1:
                dists = dists.reshape(-1, 1)
            self._backend = "ckdtree"

        kth = dists[:, -1]
        self._radii_train = kth

        if self.sigma is not None:
            mu = float(np.mean(kth))
            sd = float(np.std(kth, ddof=1)) if len(kth) > 1 else 0.0
            self._thr = mu + self.sigma * sd
        else:
            q = 0.99 if self.q is None else self.q
            self._thr = float(np.quantile(kth, q))
        return self

    def kth_distance(self, X):
        import numpy as np
        X = np.asarray(X, dtype=float)
        if self._backend == "sklearn":
            dists, _ = self._nbrs.kneighbors(X, n_neighbors=self.k)
            return dists[:, -1]
        else:
            dists, _ = self._nbrs.query(X, k=self.k)
            if self.k == 1:
                return np.atleast_1d(dists)
            return dists[:, -1]

    def contains(self, X):
        kth = self.kth_distance(X)
        return kth <= self._thr

    # -------- Portable pickling --------
    def __getstate__(self):
        """
        Store only portable pieces: params, threshold, and Xtrain_log.
        (No sklearn objects / private modules.)
        """
        return {
            "k": self.k,
            "q": self.q,
            "sigma": self.sigma,
            "metric": self.metric,
            "_thr": self._thr,
            "_Xtrain_log": self._Xtrain_log,
        }

    def __setstate__(self, state):
        """
        Rebuild a neighbor backend from the stored training data
        with the current environment (sklearn if available, else SciPy).
        """
        self.k = int(state["k"])
        self.q = state["q"]
        self.sigma = state["sigma"]
        self.metric = state["metric"]
        self._thr = state["_thr"]
        self._Xtrain_log = state["_Xtrain_log"]
        self._radii_train = None

        # Recreate backend
        try:
            from sklearn.neighbors import NearestNeighbors
            self._nbrs = NearestNeighbors(n_neighbors=self.k, metric=self.metric)
            self._nbrs.fit(self._Xtrain_log)
            self._backend = "sklearn"
        except Exception:
            from scipy.spatial import cKDTree
            self._nbrs = cKDTree(self._Xtrain_log)
            self._backend = "ckdtree"


# --------------------------- Helpers ---------------------------

def _geom_grid(vmin, vmax, n=200):
    """
    Geometric grid between vmin and vmax (positive).
    Falls back to linear spacing if bounds are <= 0.
    """
    if vmin <= 0 or vmax <= 0:
        return np.linspace(max(vmin, 1e-12), vmax, n)
    return np.geomspace(vmin, vmax, n)

def _intervals_from_mask(x, mask):
    """
    Given x (1D increasing) and boolean mask (True=in support),
    return a list of (x0, x1) intervals where mask is False (=out-of-support).
    """
    intervals = []
    if len(x) == 0:
        return intervals
    current = None
    for i in range(len(x)):
        out = (not mask[i])
        if out and current is None:
            current = x[i]
        if (not out) and (current is not None):
            x1 = x[i-1] if i > 0 else x[i]
            intervals.append((current, x1))
            current = None
    if current is not None:
        intervals.append((current, x[-1]))
    return intervals

def _label(name):
    mapping = {
        'fluence': 'Fluence (Jy ms)',
        'scattering_time': 'Scattering time τ (ms) @ 1 GHz',
        'width': 'Pulse width (ms)',
        'dm': 'DM (pc cm⁻³)',
    }
    return mapping.get(name, name)

def _pairwise(iterable):
    L = list(iterable)
    for i in range(len(L)):
        for j in range(i+1, len(L)):
            yield L[i], L[j]

# --- Train on injections and write to pickle ---
if __name__ == "__main__":

    SNR_CUT = SNR_CUT   # SNR cut for detections (CHANGE AS NEEDED)

    SIGMA = 12.0        # threshold for KNNEnvelope
    SIDELOBE_CUT = 5.0  # beam_x cut for sidelobes (float or None)

    # ---------------- Read data ----------------
    print("Reading injections and detections...")
    injections, detections, nondetections = get_injections_detected(
        DEFAULT_LOCAL_INJ_FILE, return_nondets=True
    )
    print("Done reading injections and detections.")

    # ---------------- Cuts ----------------
    (
        detected_fluence, detected_scattering_time, detected_width, detected_dm, detected_status,
        nondetected_fluence, nondetected_scattering_time, nondetected_width, nondetected_dm, nondetected_status,
        all_fluence, all_scattering_time, all_width, all_dm, all_status
    ) = cut_detections_nondetections(
        detections, nondetections, snr_cut=SNR_CUT, sidelobe_cut=SIDELOBE_CUT, cut_badtimes=True
    )

    print("Done cutting detections and nondetections.")
    print(f"Total injections: {len(all_fluence)}")
    print(f"Detected injections: {np.sum(all_status)}")
    print(f"Nondetected injections: {np.sum(~all_status)}")

    # ---------------- Build 4D log-space training set ----------------
    params_log = {
        'fluence':         np.log10(all_fluence),
        #'scattering_time': np.log10(all_scattering_time),
        #'width':           np.log10(all_width),
        'dm':              np.log10(all_dm),
    }
    names = ('fluence','dm')  # PROP_ORDER  ('fluence','scattering_time','width','dm')

    Xtrain = np.column_stack([params_log[name] for name in names])
    knn_env = KNNEnvelope(k=50, quantile=None, sigma=SIGMA)
    knn_env.fit(Xtrain)

    # ---------------- Write to package-local data/masks ----------------
    # Build filename to match SelectionFunction: knn-mask-<preds>_sl<5.0|none>_sigma<...>.pkl
    sl_str = "none" if (SIDELOBE_CUT is None) else f"{float(SIDELOBE_CUT):.1f}"
    tag = f"{'-'.join(names)}_sl{sl_str}_sigma{float(SIGMA):.1f}"

    mask_dir = os.path.join(DEFAULT_LOCAL_DATA_PATH, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    out_path = os.path.join(mask_dir, f"knn-mask-{tag}.pkl")

    with open(out_path, "wb") as f:
        pickle.dump(knn_env, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"KNNEnvelope model saved to {out_path}")

