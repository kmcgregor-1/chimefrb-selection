from itertools import combinations_with_replacement
from collections import Counter
import numpy as np
from .utils import PROP_ORDER as _PROP_ORDER, get_prop_vector

def build_design_matrix(anchor, predictors, order, predictor_names=None, return_names=False):
    """
    Construct the polynomial design matrix used in both fitting and prediction.

    Parameters
    ----------
    anchor : array-like
        A 1-D array whose length sets the number of rows (any one of the predictors).
        This defines the number of samples (rows).
    predictors : list[array-like]
        Each element is a 1-D array of the same length as `anchor`. The predictors
        are combined into polynomial terms.
    order : int
        Maximum polynomial degree to include. Terms up to this order are generated
        using combinations_with_replacement.
    predictor_names : list[str], optional
        Human-readable names for each predictor, used for labeling. Must match
        the number of predictors. Defaults to x0, x1, …
    return_names : bool, default=False
        If True, also return a list of human-readable column names.

    Returns
    -------
    X : ndarray of shape (n_samples, n_terms)
        Polynomial design matrix, with intercept as the first column.
    names : list[str], optional
        Names of the terms, if return_names=True.

    Notes
    -----
    - The exact column ordering (intercept, then increasing polynomial degree,
      with predictors combined via combinations_with_replacement) matches what is
      used during coefficient fitting.
    - This function must be used consistently in both model fitting and evaluation
      to ensure that the beta coefficients line up with the same columns.
    """
    n_pred = len(predictors)
    if predictor_names is None:
        predictor_names = [f'x{i}' for i in range(n_pred)]
    elif len(predictor_names) != n_pred:
        raise ValueError("predictor_names must match the number of predictors")

    anchor = np.asarray(anchor, dtype=float)
    X_terms  = [np.ones_like(anchor, dtype=float)]
    name_terms = ['Intercept']

    for o in range(1, order + 1):
        for combo in combinations_with_replacement(range(n_pred), o):
            term = np.ones_like(anchor, dtype=float)
            for idx in combo:
                term *= np.asarray(predictors[idx], dtype=float)
            X_terms.append(term)

            counts = Counter(combo)
            pieces = []
            for idx in sorted(counts):
                base = predictor_names[idx]
                power = counts[idx]
                pieces.append(base if power == 1 else f'{base}^{power}')
            name_terms.append(' * '.join(pieces))

    X = np.column_stack(X_terms)
    return (X, name_terms) if return_names else X


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def logistic_selection(props, beta, order, cov_beta=None,
                       predictor_names=None, clip_eta=500.0, log_transform=True):
    """
    Evaluate a fitted logistic selection function at a single point.
    Takes linear-valued inputs and internally applies log10 transform
    (to match the fitting procedure).

    Parameters
    ----------
    props : dict or array-like
        Predictor values in *linear units*. If dict, must contain keys matching
        `predictor_names`. If array-like, must be ordered consistently with
        `predictor_names`.
    beta : array-like of shape (n_terms,)
        Logistic regression coefficients, including intercept as beta[0].
    order : int
        Maximum polynomial order. Must match the order used in training.
    cov_beta : array-like of shape (n_terms, n_terms), optional
        Covariance matrix of the fitted coefficients. If provided, the function
        also returns the delta-method standard deviation of the predicted
        selection probability.
    predictor_names : list[str]
        Names of the predictors in the same order as used during fitting.
        Must match the saved model metadata.
    clip_eta : float, default=500.0
        Bound for linear predictor η = Xβ to avoid numerical overflow.
    log_transform : bool, default=True
        If True, automatically apply np.log10 to each predictor value before
        building the design matrix (to match training pipeline).

    Returns
    -------
    p : float
        Predicted selection probability.
    p_std : float, optional
        Standard deviation of the predicted probability (if cov_beta provided).
    """
    if predictor_names is None:
        predictor_names = list(_PROP_ORDER)

    # Prepare predictors in the exact trained order.
    # If dict: accept aliases and canonicalize. If array-like: assume same order as predictor_names.
    if isinstance(props, dict):
        x_vals = get_prop_vector(props, names=predictor_names).tolist()
    else:
        x_vals = np.asarray(props, dtype=float).tolist()

    # Apply log10 transform (to match training)
    if log_transform:
        x_vals = [np.log10(v) for v in x_vals]

    # Build 1-row design matrix
    anchor = np.array([0.0], dtype=float)
    predictors = [np.array([xi], dtype=float) for xi in x_vals]
    X = build_design_matrix(anchor, predictors, order,
                            predictor_names=predictor_names,
                            return_names=False).astype(float)

    if X.shape[1] != len(beta):
        raise ValueError(f"Design matrix has {X.shape[1]} columns but beta has {len(beta)}. "
                         "Check predictor_names/order against the trained model.")

    # Linear predictor
    eta = float(np.squeeze(X @ np.asarray(beta, dtype=float)))
    eta = np.clip(eta, -clip_eta, clip_eta)
    p = 1.0 / (1.0 + np.exp(-eta))

    if cov_beta is None:
        return p
    else:
        cov_beta = np.asarray(cov_beta, dtype=float)
        var_eta = float(np.squeeze(X @ cov_beta @ X.T))  # 1x1 -> scalar
        p_std = np.sqrt(max(var_eta, 0.0)) * p * (1.0 - p)
        return p, p_std

def logistic_selection_batch(X_raw, beta, order, cov_beta=None,
                              predictor_names=None, clip_eta=500.0,
                              log_transform=True):
    """
    Vectorized batch evaluation of the logistic selection function.

    Evaluates the same logistic model as :func:`logistic_selection` but for an
    entire array of samples in a single NumPy call, avoiding Python-level loops.

    Parameters
    ----------
    X_raw : ndarray of shape (N, d)
        Predictor values in *linear units*. All values must be > 0 when
        ``log_transform=True``.
    beta : array-like of shape (n_terms,)
        Logistic regression coefficients, including intercept as ``beta[0]``.
    order : int
        Maximum polynomial order. Must match the order used in training.
    cov_beta : array-like of shape (n_terms, n_terms), optional
        Covariance matrix of the fitted coefficients. If provided, the function
        also returns the delta-method standard deviation for each sample.
    predictor_names : list[str], optional
        Names of the predictors in the same order as ``X_raw`` columns. Defaults
        to ``PROP_ORDER[:d]``.
    clip_eta : float, default=500.0
        Bound for linear predictor η = X β to avoid numerical overflow.
    log_transform : bool, default=True
        If True, apply ``np.log10`` to all predictor values before building the
        design matrix (to match the training pipeline).

    Returns
    -------
    p : ndarray of shape (N,)
        Predicted selection probabilities.
    p_std : ndarray of shape (N,), optional
        Delta-method standard deviations of the predicted probabilities.
        Only returned when ``cov_beta`` is provided.
    """
    X_raw = np.asarray(X_raw, dtype=float)
    if X_raw.ndim == 1:
        X_raw = X_raw.reshape(1, -1)
    N, d = X_raw.shape

    if predictor_names is None:
        predictor_names = list(_PROP_ORDER[:d])

    if log_transform:
        X_log = np.log10(X_raw)   # (N, d) — vectorized
    else:
        X_log = X_raw.copy()

    # Build the design matrix once for all N samples.
    # build_design_matrix accepts length-N arrays as predictors natively.
    anchor = np.zeros(N, dtype=float)
    predictors = [X_log[:, j] for j in range(d)]
    DM = build_design_matrix(anchor, predictors, order,
                             predictor_names=predictor_names,
                             return_names=False).astype(float)  # (N, n_terms)

    beta = np.asarray(beta, dtype=float)
    if DM.shape[1] != len(beta):
        raise ValueError(
            f"Design matrix has {DM.shape[1]} columns but beta has {len(beta)}. "
            "Check predictor_names/order against the trained model."
        )

    eta = DM @ beta                          # (N,)
    eta = np.clip(eta, -clip_eta, clip_eta)
    p = 1.0 / (1.0 + np.exp(-eta))          # (N,)

    if cov_beta is None:
        return p

    cov_beta = np.asarray(cov_beta, dtype=float)
    # Delta-method variance per row:
    # var_eta[i] = DM[i] @ cov_beta @ DM[i]^T  ≡  sum((DM @ cov_beta) * DM, axis=1)
    var_eta = np.sum((DM @ cov_beta) * DM, axis=1)   # (N,)
    var_eta = np.maximum(var_eta, 0.0)
    p_std = np.sqrt(var_eta) * p * (1.0 - p)         # (N,)
    return p, p_std


if __name__ == "__main__":

    # Example usage
    anchor = np.array([1, 2, 3])
    predictors = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    order = 2
    X, names = build_design_matrix(anchor, predictors, order, return_names=True)
    print("Design Matrix:\n", X)
    print("Column Names:\n", names)

    props = {'fluence': 2, 'scattering_time': 5, 'width': 3, 'dm': 7}
    beta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # Example coefficients
    p = logistic_selection(props, beta)
    print("Selection Probability:", p)