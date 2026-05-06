"""
Tests for vectorized evaluation of logistic_selection_batch and
SelectionFunction.calculate_selection_probability / calculate_ht_weight.

These tests are self-contained and do not require the packaged .npz / .pkl
model artifacts.  A tiny synthetic model (low-order polynomial, trivial KNN
envelope built from the in-domain test points) is assembled from scratch.
"""
import numpy as np
import pytest

from chimefrb_selection.funcs import (
    build_design_matrix,
    logistic_selection,
    logistic_selection_batch,
)
from chimefrb_selection.knn_mask import KNNEnvelope
from chimefrb_selection.selection import SelectionFunction


# ---------------------------------------------------------------------------
# Helpers to build a tiny synthetic SelectionFunction without touching disk
# ---------------------------------------------------------------------------

def _make_beta(predictor_names, degree):
    """Return a deterministic beta vector of the correct length."""
    from itertools import combinations_with_replacement

    d = len(predictor_names)
    n_terms = 1  # intercept
    for o in range(1, degree + 1):
        n_terms += len(list(combinations_with_replacement(range(d), o)))
    rng = np.random.default_rng(42)
    return rng.standard_normal(n_terms)


def _make_sf(predictor_names=("fluence", "dm"), degree=2):
    """
    Build a SelectionFunction with a synthetic beta / cov_beta and a KNNEnvelope
    trained on a small grid of log10-space points.
    """
    beta = _make_beta(predictor_names, degree)
    cov_beta = np.eye(len(beta)) * 1e-4

    # Train KNN envelope on a small 5×5 log-space grid
    g = np.linspace(0.0, 2.0, 5)  # log10 values in [0, 2]
    from itertools import product as iproduct
    d = len(predictor_names)
    Xtrain_log = np.array(list(iproduct(*([g] * d))))  # (25, d) or (125, d) etc.
    knn = KNNEnvelope(k=3, quantile=0.99)
    knn.fit(Xtrain_log)

    sf = SelectionFunction.__new__(SelectionFunction)
    # Set the dataclass fields directly
    sf.predictor_names = list(predictor_names)
    sf.degree = degree
    sf.snr_cut = 12.0
    sf.exclude_sidelobes = True
    sf.sidelobe_cut = 5.0
    sf.reweighted = False
    sf.models_base_dir = "/dev/null"
    sf.knn_dir = "/dev/null"
    sf.beta = beta
    sf.cov_beta = cov_beta
    sf.knn_envelope = knn
    sf._dim_folder = None
    sf._pred_folder = None
    sf._model_dir = None
    sf._tag = None
    sf._npz_path = None
    sf._knn_path = None
    sf._knn_tag = None
    return sf


# ---------------------------------------------------------------------------
# 1. logistic_selection_batch: batch vs scalar consistency
# ---------------------------------------------------------------------------

class TestLogisticSelectionBatch:
    """Batch evaluator must agree with scalar evaluator on every row."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.predictor_names = ["fluence", "scattering_time", "width", "dm"]
        self.degree = 2
        self.beta = _make_beta(self.predictor_names, self.degree)
        self.cov_beta = np.eye(len(self.beta)) * 5e-4
        rng = np.random.default_rng(0)
        # Linear-space values all > 0  (will be log10-transformed inside)
        self.X_raw = 10.0 ** rng.uniform(0.0, 2.0, size=(20, 4))

    def test_batch_p_matches_scalar(self):
        p_batch = logistic_selection_batch(
            self.X_raw, self.beta, self.degree,
            predictor_names=self.predictor_names,
        )
        p_scalar = np.array([
            logistic_selection(
                props=row,
                beta=self.beta,
                order=self.degree,
                predictor_names=self.predictor_names,
            )
            for row in self.X_raw
        ])
        np.testing.assert_allclose(p_batch, p_scalar, rtol=1e-10,
                                   err_msg="Batch probabilities differ from scalar loop")

    def test_batch_std_matches_scalar(self):
        p_batch, std_batch = logistic_selection_batch(
            self.X_raw, self.beta, self.degree,
            cov_beta=self.cov_beta,
            predictor_names=self.predictor_names,
        )
        for i, row in enumerate(self.X_raw):
            p_s, std_s = logistic_selection(
                props=row, beta=self.beta, order=self.degree,
                cov_beta=self.cov_beta,
                predictor_names=self.predictor_names,
            )
            np.testing.assert_allclose(p_batch[i], p_s, rtol=1e-10)
            np.testing.assert_allclose(std_batch[i], std_s, rtol=1e-10)

    def test_single_row_1d_input(self):
        row = self.X_raw[0]  # shape (4,)
        p_batch = logistic_selection_batch(
            row, self.beta, self.degree,
            predictor_names=self.predictor_names,
        )
        p_scalar = logistic_selection(
            props=row, beta=self.beta, order=self.degree,
            predictor_names=self.predictor_names,
        )
        assert p_batch.shape == (1,)
        np.testing.assert_allclose(p_batch[0], p_scalar, rtol=1e-10)


# ---------------------------------------------------------------------------
# 2. SelectionFunction: alias dict-of-arrays
# ---------------------------------------------------------------------------

class TestSelectionFunctionAlias:
    """Dict keys using recognised aliases must map to the same result as
    canonical names."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.sf = _make_sf(predictor_names=("fluence", "scattering_time", "width", "dm"), degree=2)
        rng = np.random.default_rng(7)
        n = 15
        # Pick values well inside the KNN training grid (log10 in [0.5, 1.5])
        self.flu   = 10.0 ** rng.uniform(0.5, 1.5, n)
        self.tau   = 10.0 ** rng.uniform(0.5, 1.5, n)
        self.wid   = 10.0 ** rng.uniform(0.5, 1.5, n)
        self.dm    = 10.0 ** rng.uniform(0.5, 1.5, n)

    def test_alias_keys_match_canonical(self):
        props_alias = {
            "fluence_jy_ms":  self.flu,
            "tau_1_ghz_ms":   self.tau,
            "pulse_width_ms": self.wid,
            "dm":             self.dm,
        }
        props_canon = {
            "fluence":        self.flu,
            "scattering_time": self.tau,
            "width":          self.wid,
            "dm":             self.dm,
        }
        p_alias = self.sf.calculate_selection_probability(props_alias)
        p_canon = self.sf.calculate_selection_probability(props_canon)
        np.testing.assert_array_equal(p_alias, p_canon,
                                      err_msg="Alias keys should give identical result to canonical keys")

    def test_alias_keys_std(self):
        props_alias = {
            "fluence_jy_ms":  self.flu,
            "tau_1_ghz_ms":   self.tau,
            "pulse_width_ms": self.wid,
            "dm":             self.dm,
        }
        p, p_std = self.sf.calculate_selection_probability(props_alias, return_std=True)
        assert p.shape == (len(self.flu),)
        assert p_std.shape == (len(self.flu),)
        assert np.all(np.isfinite(p) | np.isnan(p))

    def test_2d_array_input(self):
        arr = np.column_stack([self.flu, self.tau, self.wid, self.dm])
        p_arr = self.sf.calculate_selection_probability(arr)
        assert p_arr.shape == (len(self.flu),)

    def test_scalar_dict_returns_float(self):
        props = {
            "fluence": float(self.flu[0]),
            "scattering_time": float(self.tau[0]),
            "width": float(self.wid[0]),
            "dm": float(self.dm[0]),
        }
        p = self.sf.calculate_selection_probability(props)
        assert isinstance(p, float), f"Expected float, got {type(p)}"


# ---------------------------------------------------------------------------
# 3. Out-of-domain → NaN
# ---------------------------------------------------------------------------

class TestOutOfDomain:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.sf = _make_sf(predictor_names=("fluence", "dm"), degree=2)

    def test_negative_value_is_nan(self):
        props = {"fluence": np.array([5.0, -1.0, 10.0]),
                 "dm": np.array([100.0, 100.0, 100.0])}
        p = self.sf.calculate_selection_probability(props)
        assert np.isnan(p[1]), "Negative predictor value should give NaN"

    def test_zero_value_is_nan(self):
        props = {"fluence": np.array([5.0, 0.0]),
                 "dm": np.array([100.0, 100.0])}
        p = self.sf.calculate_selection_probability(props)
        assert np.isnan(p[1]), "Zero predictor value should give NaN"

    def test_out_of_knn_envelope_is_nan(self):
        # Extremely large values far outside training grid
        props = {"fluence": np.array([1e12]), "dm": np.array([1e12])}
        p = self.sf.calculate_selection_probability(props)
        # n=1 returns a scalar float
        assert np.isnan(p), "Out-of-envelope point should give NaN"

    def test_out_of_domain_std_is_nan(self):
        props = {"fluence": np.array([5.0, -1.0]), "dm": np.array([100.0, 100.0])}
        p, p_std = self.sf.calculate_selection_probability(props, return_std=True)
        assert np.isnan(p[1])
        assert np.isnan(p_std[1])


# ---------------------------------------------------------------------------
# 4. HT weight
# ---------------------------------------------------------------------------

class TestHTWeight:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.sf = _make_sf(predictor_names=("fluence", "dm"), degree=2)
        rng = np.random.default_rng(99)
        n = 10
        self.props = {
            "fluence": 10.0 ** rng.uniform(0.5, 1.5, n),
            "dm":      10.0 ** rng.uniform(0.5, 1.5, n),
        }

    def test_ht_weight_is_inverse_p(self):
        p = self.sf.calculate_selection_probability(self.props)
        w = self.sf.calculate_ht_weight(self.props)
        p = np.asarray(p, float)
        w = np.asarray(w, float)
        good = np.isfinite(p) & (p > 0)
        np.testing.assert_allclose(w[good], 1.0 / p[good], rtol=1e-10)

    def test_ht_weight_with_std(self):
        p, p_std = self.sf.calculate_selection_probability(self.props, return_std=True)
        w, w_std = self.sf.calculate_ht_weight(self.props, return_std=True)
        p = np.asarray(p, float)
        p_std = np.asarray(p_std, float)
        w = np.asarray(w, float)
        w_std = np.asarray(w_std, float)
        good = np.isfinite(p) & (p > 0)
        np.testing.assert_allclose(w[good], 1.0 / p[good], rtol=1e-10)
        np.testing.assert_allclose(w_std[good], p_std[good] / p[good] ** 2, rtol=1e-10)
