from dataclasses import dataclass, field
from typing import Mapping, Sequence, Union, Optional
import os
import numpy as np
import pickle
import warnings
from pathlib import Path

from .funcs import logistic_selection, logistic_selection_batch
from .utils import PROP_ORDER as _PROP_ORDER, DEFAULT_MODELS_DIR, DEFAULT_KNN_DIR, get_prop_vector, canonicalize_props as _canonicalize_props
from .knn_mask import KNNEnvelope

ArrayLike = Union[Sequence[float], np.ndarray]

import numpy as np
from typing import Mapping, Sequence, Union

ArrayLike = Union[Sequence[float], np.ndarray]

def _ci_get(props: Mapping[str, ArrayLike], name: str):
    """
    Case-insensitive, underscore-tolerant lookup:
      exact -> lower -> upper -> no-underscore variants.
    """
    if name in props: return props[name]
    lo, up = name.lower(), name.upper()
    if lo in props: return props[lo]
    if up in props: return props[up]
    nu = name.replace("_", "")
    if nu in props: return props[nu]
    if nu.lower() in props: return props[nu.lower()]
    if nu.upper() in props: return props[nu.upper()]
    raise KeyError(f"Missing predictor '{name}' in props.")

def _normalize_props_to_2d(
    props: Union[Mapping[str, ArrayLike], ArrayLike],
    predictor_names: Sequence[str]
) -> np.ndarray:
    """
    Returns (n_samples, n_predictors) float array in model order.
    - Mapping[str, scalar/array] → canonicalize aliases, broadcast columns, stack.
    - 1D array of length d → (1,d)
    - 2D array (n,d) → as-is (validates d).
    """
    if isinstance(props, Mapping):
        # Canonicalize keys to handle aliases (e.g., 'fluence_jy_ms' -> 'fluence').
        canon = _canonicalize_props(dict(props))
        cols = []
        for nm in predictor_names:
            if nm in canon:
                cols.append(np.asarray(canon[nm], dtype=float))
            else:
                # Fall back to case-insensitive lookup for non-alias mismatches
                cols.append(np.asarray(_ci_get(props, nm), dtype=float))
        # Broadcast columns to a common shape (NumPy handles scalars vs arrays)
        bcols = np.broadcast_arrays(*cols)
        flat_cols = [c.reshape(-1) for c in bcols]
        return np.column_stack(flat_cols)  # (n, d)
    arr = np.asarray(props, dtype=float)
    if arr.ndim == 1:
        if arr.size != len(predictor_names):
            raise ValueError(f"Expected {len(predictor_names)} predictors, got {arr.size}.")
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        if arr.shape[1] != len(predictor_names):
            raise ValueError(f"Expected {len(predictor_names)} predictors, got {arr.shape[1]}.")
        return arr
    raise ValueError("props must be a Mapping or a 1D/2D array-like.")

def _as_vec(props: Union[Mapping[str, float], ArrayLike], names: Sequence[str]) -> np.ndarray:
    """
    Return a 1D float array in the given `names` order.
    If `props` is a dict, aliases are accepted and canonicalized.
    If array-like, assume it already matches `names` order.
    """
    if isinstance(props, Mapping):
        # Pull exactly the required names (aliases OK)
        return get_prop_vector(dict(props), names=names)
    arr = np.asarray(props, dtype=float).ravel()
    if len(arr) != len(names):
        raise ValueError(f"Expected {len(names)} values in order {names}, got {len(arr)}.")
    return arr

# ------------------------------------------------------------------

@dataclass
class SelectionFunction:
    """
    Wrapper around logistic_selection() + KNNEnvelope

    Examples
    --------
    # 4D default
    sf4 = SelectionFunction()  # ['fluence','scattering_time','width','dm'], degree=3, snr_cut=8

    # 2D DM–scattering
    sf2 = SelectionFunction(predictor_names=['dm','scattering_time'], degree=2, snr_cut=12)

    # 1D scattering only
    sf1 = SelectionFunction(predictor_names=['scattering_time'], degree=3, snr_cut=8)

    # Explicitly use non-reweighted model
    sf_unweighted = SelectionFunction(reweighted=False)
    """
    # Defaults so constructor can be empty
    predictor_names: Sequence[str] = field(default_factory=lambda: list(_PROP_ORDER))
    degree: int = 3
    snr_cut: float = 12.0
    exclude_sidelobes: bool = True
    sidelobe_cut: Optional[float] = 5.0
    reweighted: bool = True 

    # Base directories (env-overridable)
    models_base_dir: str = DEFAULT_MODELS_DIR
    knn_dir: str = DEFAULT_KNN_DIR

    # Loaded artifacts
    beta: Optional[np.ndarray] = None
    cov_beta: Optional[np.ndarray] = None
    knn_envelope: Optional["KNNEnvelope"] = None

    # Internals (computed)
    _dim_folder: Optional[str] = None
    _pred_folder: Optional[str] = None
    _model_dir: Optional[Path] = None
    _tag: Optional[str] = None
    _npz_path: Optional[Path] = None
    _knn_path: Optional[Path] = None
    _knn_tag: Optional[str] = None

    def __post_init__(self):
        # only marginalize for less than 4D models
        if len(self.predictor_names) == 4:
            if self.reweighted:
                warnings.warn(
                    "4D selection functions are not reweighted; forcing reweighted=False",
                    RuntimeWarning,
                )
            self.reweighted = False

        # folder names
        self._dim_folder  = f"{len(self.predictor_names)}d_selection_function"
        self._pred_folder = "_".join(self.predictor_names)

        sl_str = "none" if (not self.exclude_sidelobes or self.sidelobe_cut is None) else f"{float(self.sidelobe_cut):.1f}"
        self._tag     = f"{'-'.join(self.predictor_names)}_order{int(self.degree)}_snr{int(self.snr_cut)}_sl{sl_str}"

        if self.reweighted:
            self._tag += "_reweighted"

        self._knn_tag = f"{'-'.join(self.predictor_names)}_sl{sl_str}_sigma{int(self.snr_cut)}"

        # Model dir: <models_base_dir>/<Nd>_selection_function/<pred_folder>
        self._model_dir = Path(self.models_base_dir) / self._dim_folder / self._pred_folder
        self._npz_path  = self._model_dir / f"IRLS_output_{self._tag}.npz"

        # Mask path (kept flat under masks/)
        self._knn_path  = Path(self.knn_dir) / f"knn-mask-{self._knn_tag}.pkl"

        # Auto-load
        if self.beta is None or self.cov_beta is None:
            self._load_npz(str(self._npz_path))
        if self.knn_envelope is None:
            self._load_knn(str(self._knn_path))

    # ---------- Path helpers ----------

    def npz_path(self) -> str:
        return str(self._npz_path)

    def knn_path(self) -> str:
        return str(self._knn_path)

    def model_dir(self) -> str:
        return str(self._model_dir)

    # ---------- Loaders ----------

    def _load_npz(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                "Model NPZ not found at:\n"
                f"  {path}\n"
                f"(Looked under models_base_dir='{self.models_base_dir}', model_dir='{self._model_dir}')\n"
                "If you don't have this file, you may need to fit your own model manually for your specified params.\n"
                "Use the CLI specificied in fitting/logistic_regression_cli.py to do so."
            )
        data = np.load(path)
        # Flexible key handling
        for key in ("beta_est", "beta", "coef_", "coefficients", "b"):
            if key in data:
                self.beta = np.asarray(data[key], dtype=float)
                break
        else:
            raise KeyError(f"No beta-like key found in {path}. Keys available: {list(data.keys())}")

        # cov_beta is optional
        for key in ("cov_matrix", "cov", "cov_beta", "covariance"):
            if key in data:
                self.cov_beta = np.asarray(data[key], dtype=float)
                break

    def _load_knn(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(
                "KNN envelope PKL not found at:\n"
                f"  {path}\n"
                f"(Looked under knn_dir='{self.knn_dir}')\n"
                "If you don't have this file, you may need to fit your own KNN envelope manually.\n"
                "Customize the script for your params and run python -m chimefrb_selection.knn_mask to do so"
            )

        # Tolerate legacy pickles saved from __main__
        class _KNNRedirectingUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "KNNEnvelope" and module in ("__main__", "knn_mask", "chimefrb_selection.knn_mask"):
                    from .knn_mask import KNNEnvelope
                    return KNNEnvelope
                return super().find_class(module, name)

        with open(path, "rb") as f:
            self.knn_envelope = _KNNRedirectingUnpickler(f).load()

    # ---------- Internals ----------

    def _vector_in_model_order(self, props: Union[Mapping[str, float], ArrayLike]) -> np.ndarray:
        """
        Build a vector in THIS model's order (self.predictor_names).
        No remapping to the global 4D order; supports arbitrary subsets.
        """
        return _as_vec(props, self.predictor_names)

    def _require_mask_contains(self, raw_vec_model_order: np.ndarray) -> None:
        """
        Check that linear-valued predictors are inside the KNN envelope
        (which was trained in log10-space) for THIS predictor set.
        """
        if self.knn_envelope is None:
            raise RuntimeError(
                "KNN envelope not loaded. Expected at:\n"
                f"  {self._knn_path}\n"
                "Load it or provide knn_envelope=... at construction."
            )

        if np.any(raw_vec_model_order <= 0):
            raise ValueError("All predictors must be > 0 to take log10.")
        xlog = np.log10(raw_vec_model_order).reshape(1, -1)

        if not bool(self.knn_envelope.contains(xlog)[0]):
            raise ValueError("Properties are outside the selection function domain!")

    def contains(self, props: Union[Mapping[str, ArrayLike], ArrayLike]) -> np.ndarray:
        """
        Alias-aware KNN containment. Supports dict of scalars/arrays (any broadcastable shape)
        or array-like with last axis == d (len(self.predictor_names)).
        Returns a boolean array with the same broadcast shape as the inputs (or shape[:-1] for arrays).
        """
        if self.knn_envelope is None:
            raise RuntimeError("KNN envelope not loaded.")

        d = len(self.predictor_names)

        # --- Normalize to (N, d) RAW-units array using your alias logic ---
        if isinstance(props, Mapping):
            # broadcast all provided keys to a common shape S
            keys = list(props.keys())
            vals = [np.asarray(props[k], dtype=float) for k in keys]
            if len(vals) == 0:
                raise ValueError("props dict is empty.")
            bvals = np.broadcast_arrays(*vals)
            S = bvals[0].shape   # broadcasted sample shape ((), (n,), (n,m), ...)
            N = int(np.prod(S)) if S else 1

            # build N scalar samples in original keys, then canonicalize via _vector_in_model_order
            raveled = {k: bv.ravel() for k, bv in zip(keys, bvals)}
            rows = []
            for i in range(N):
                sample = {k: float(raveled[k][i]) for k in keys}
                v = self._vector_in_model_order(sample)  # alias-aware, orders by self.predictor_names
                rows.append(v)
            X = np.vstack(rows)     # (N, d)
            out_shape = S
        else:
            arr = np.asarray(props, dtype=float)
            if arr.ndim == 1:
                if arr.size != d:
                    raise ValueError(f"Expected {d} predictors along last axis, got {arr.size}.")
                X = arr.reshape(1, d)
                out_shape = ()
            else:
                if arr.shape[-1] != d:
                    raise ValueError(f"Last axis must have length {d} predictors, got {arr.shape[-1]}.")
                out_shape = arr.shape[:-1]
                X = arr.reshape(-1, d)  # positional, already in model order

        if np.any(X <= 0.0):
            raise ValueError("All predictors must be > 0 to take log10.")
        xlog = np.log10(X)

        mask_flat = self.knn_envelope.contains(xlog)  # (N,)
        return mask_flat.reshape(out_shape) if out_shape != () else bool(mask_flat[0])


    # ---------- Public API ----------
    def calculate_selection_probability(
        self,
        props: Union[Mapping[str, ArrayLike], ArrayLike],
        return_std: bool = False
    ):
        """
        Calculate selection probability for single or multiple sets of burst properties.

        OUT-OF-DOMAIN BEHAVIOR
        ----------------------
        If any sample is outside the selection-function domain (<=0 for any predictor
        or outside the KNN envelope), then that sample's selection probability is NaN
        (and its std is NaN if return_std=True).
        """
        # ---- Build a (n_samples, d) matrix of canonical, ordered vectors ----
        arr = _normalize_props_to_2d(props, self.predictor_names)  # (n, d)

        n, d = arr.shape

        # ---- Determine in-domain samples ----
        # Domain requires predictors > 0 (for log10), AND KNN mask containment.
        positive = np.all(arr > 0.0, axis=1)  # (n,)
        contains = np.zeros(n, dtype=bool)

        # Only call contains() on rows that are strictly positive to avoid log10 issues
        if np.any(positive):
            contains[positive] = self.knn_envelope.contains(np.log10(arr[positive]))

        in_domain = positive & contains  # (n,)

        # ---- Allocate outputs filled with NaN ----
        p = np.full(n, np.nan, dtype=float)

        if not return_std:
            # Vectorized batch evaluation for all in-domain rows at once
            if np.any(in_domain):
                p[in_domain] = logistic_selection_batch(
                    arr[in_domain],
                    beta=self.beta,
                    order=self.degree,
                    cov_beta=None,
                    predictor_names=self.predictor_names,
                )

            return float(p[0]) if n == 1 else p

        # return_std=True
        if self.cov_beta is None:
            raise ValueError("cov_beta not available (missing in NPZ). Needed for return_std=True.")

        p_std = np.full(n, np.nan, dtype=float)
        if np.any(in_domain):
            pv, sv = logistic_selection_batch(
                arr[in_domain],
                beta=self.beta,
                order=self.degree,
                cov_beta=self.cov_beta,
                predictor_names=self.predictor_names,
            )
            p[in_domain] = pv
            p_std[in_domain] = sv

        if n == 1:
            return float(p[0]), float(p_std[0])
        return p, p_std


    def calculate_ht_weight(
        self,
        props: Union[Mapping[str, ArrayLike], ArrayLike],
        return_std: bool = False
    ):
        """
        Calculate the HT weight (1/p) and optionally its error using propagation.

        OUT-OF-DOMAIN BEHAVIOR
        ----------------------
        If p is NaN (out-of-domain) or p <= 0, then w is NaN (and w_std is NaN).
        """
        if return_std:
            p, p_std = self.calculate_selection_probability(props, return_std=True)
            p = np.asarray(p, dtype=float)
            p_std = np.asarray(p_std, dtype=float)

            w = np.full_like(p, np.nan, dtype=float)
            w_std = np.full_like(p, np.nan, dtype=float)

            good = np.isfinite(p) & (p > 0.0)
            if np.any(good):
                w[good] = 1.0 / p[good]
                w_std[good] = p_std[good] / (p[good] ** 2)

            if w.ndim == 0:
                return float(w), float(w_std)
            return w, w_std

        p = self.calculate_selection_probability(props, return_std=False)
        p = np.asarray(p, dtype=float)

        w = np.full_like(p, np.nan, dtype=float)
        good = np.isfinite(p) & (p > 0.0)
        if np.any(good):
            w[good] = 1.0 / p[good]

        return float(w) if w.ndim == 0 else w




if __name__ == "__main__":

    # Example: 2D DM–scattering model
    sf = SelectionFunction(
        predictor_names=['scattering_time','dm'],
        degree=2,
        snr_cut=12.0,
        exclude_sidelobes=True,
        sidelobe_cut=5.0,
    )

    p = sf.calculate_selection_probability({'scattering_time':0.01,'dm':500})
    w = sf.calculate_ht_weight({'scattering_time':0.01,'dm':500})
    print("Selection Probability:", p)
    print("HT Weight:", w)

    # 1D scattering-only (if you have a 1D fit + mask for it)
    # sf1 = SelectionFunction(predictor_names=['scattering_time'])
    # print(sf1.calculate_selection_probability({'scattering_time': 1.2}))