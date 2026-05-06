#!/usr/bin/env python3
import os
import glob
import json
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.colors as colors
from datetime import datetime
from collections import Counter
from itertools import combinations_with_replacement

from scipy.stats import chi2
from scipy.special import expit
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold

from chimefrb_selection.utils import cut_detections_nondetections, get_injections_detected
from chimefrb_selection.funcs import build_design_matrix, sigmoid
from chimefrb_selection.reweighting import (
    compute_marginal_weights,
    load_fiducial_params,
    plot_reweighting_diagnostics,
    DEFAULT_FIDUCIAL_PARAMS,
    ALL_PREDICTORS,
)

DEFAULT_DATA_PATH = "../chimefrb_selection/data/fits"
DEFAULT_INJ_FILE  = "/data/user-data/kmcgregor/09-2025_injections/output.json"

# ------------- Utilities --------------

def log_likelihood_logistic(X, Y, beta, eps=np.finfo(float).eps, weights=None):
    """
    Bernoulli log-likelihood for logistic regression.
    """
    eta = X @ beta
    p = expit(eta)
    p = np.clip(p, eps, 1 - eps)
    if weights is None:
        ll = np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p), dtype=np.float64)
    else:
        ll = np.sum(weights * (Y * np.log(p) + (1 - Y) * np.log(1 - p)),
                    dtype=np.float64)
    return ll

def compute_information_criteria(log_likelihood, k, n):
    """Compute AIC and BIC."""
    aic = 2 * k - 2 * log_likelihood
    bic = k * np.log(n) - 2 * log_likelihood
    return aic, bic

def likelihood_ratio_test(ll_null, ll_alt, k_null, k_alt):
    """
    LRT between nested models.
    """
    if k_alt <= k_null:
        raise ValueError("Alternative model must have more parameters than null model.")
    lr_stat = 2.0 * (ll_alt - ll_null)
    df = k_alt - k_null
    p_value = 1.0 - chi2.cdf(lr_stat, df)
    return lr_stat, p_value

def pearson_residuals(Y, p_pred, clip=5e-3, verbose=True):
    """
    Pearson residuals with clipping to avoid 0/1 division explosions.
    """
    Y = np.asarray(Y)
    p_pred = np.asarray(p_pred)
    p_pred = np.clip(p_pred, clip, 1 - clip)
    num_clipped = np.sum((p_pred <= clip) | (p_pred >= 1 - clip))
    pct = (num_clipped / len(p_pred)) * 100
    if verbose:
        print(f"Percentage of injections clipped: {pct:.4f}%")
    return (Y - p_pred) / np.sqrt(p_pred * (1 - p_pred))

def reweighted_least_squares_sparse(X, Y, beta_init=None, tol=1e-16, max_iter=500,
                                    return_errors=False, plot=True, verbose=False,
                                    plot_path=None, sample_weights=None):
    """
    Memory-efficient IRLS for logistic regression (with tiny ridge).
    """
    n_samples, n_features = X.shape
    beta = np.zeros(n_features) if beta_init is None else beta_init

    dists = []
    for s in range(max_iter):
        eta = X @ beta
        eta = np.clip(eta, -500, 500)
        p = sigmoid(eta)
        p = np.clip(p, tol, 1 - tol)
        
        v = p * (1 - p)
        
        if sample_weights is not None:
            W_diag = sample_weights * v
        else:
            W_diag = v

        Z = eta + (Y - p) / v

        sqrt_W = np.sqrt(W_diag)
        Xw = X * sqrt_W[:, np.newaxis]
        Zw = Z * sqrt_W

        lambda_ = 1e-4
        penalty = lambda_ * np.eye(n_features)
        penalty[0, 0] = 0
        XtX = Xw.T @ Xw + penalty
        XtZ = Xw.T @ Zw

        beta_new = np.linalg.solve(XtX, XtZ)

        alpha = 0.1
        beta = beta + alpha * (beta_new - beta)

        dist = np.linalg.norm(beta_new - beta)
        dists.append(dist)

        if verbose:
            print("Iteration", s+1, " logDelta:", np.log10(dist))

        if dist < tol:
            beta = beta_new
            break

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(range(len(dists)), np.log10(dists), marker='o', label='Distance to previous beta')
        plt.hlines(tol, 0, len(dists), colors='red', linestyles='dashed', label='Tolerance')
        plt.grid(True)
        plt.title('Convergence of IRLS')
        plt.xlabel('Iteration')
        plt.ylabel('log10 distance')
        plt.legend()
        if plot_path:
            os.makedirs(os.path.dirname(plot_path), exist_ok=True)
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    if return_errors:
        try:
            cov_matrix = np.linalg.inv(XtX)
            standard_errors = np.sqrt(np.diag(cov_matrix))
        except np.linalg.LinAlgError:
            standard_errors = np.full(beta.shape, np.nan)
            cov_matrix = np.full((n_features, n_features), np.nan)
        return beta, standard_errors, cov_matrix

    return beta


def cross_validate_logistic(X, Y, n_folds=5, sample_weights=None, 
                            tol=1e-16, max_iter=500, random_state=42):
    """
    K-fold cross-validation for logistic regression.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_ll = []
    fold_brier = []
    fold_auc = []
    fold_results = []
    
    print(f"\nRunning {n_folds}-fold cross-validation...")
    print("-" * 50)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        
        if sample_weights is not None:
            w_train = sample_weights[train_idx]
            w_val = sample_weights[val_idx]
            # Renormalize weights within each fold
            w_train = w_train / w_train.sum() * len(w_train)
            w_val = w_val / w_val.sum() * len(w_val)
        else:
            w_train, w_val = None, None
        
        # Fit on training fold
        beta = reweighted_least_squares_sparse(
            X_train, Y_train, tol=tol, max_iter=max_iter,
            plot=False, verbose=False, sample_weights=w_train
        )
        
        # Evaluate on validation fold
        eta_val = X_val @ beta
        p_val = expit(np.clip(eta_val, -500, 500))
        
        ll = log_likelihood_logistic(X_val, Y_val, beta, weights=w_val)
        brier = brier_score_loss(Y_val, p_val, sample_weight=w_val)
        
        try:
            auc = roc_auc_score(Y_val, p_val, sample_weight=w_val)
        except ValueError:
            auc = np.nan
        
        fold_ll.append(ll)
        fold_brier.append(brier)
        fold_auc.append(auc)
        
        fold_results.append({
            'fold': fold + 1,
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'log_likelihood': ll,
            'brier_score': brier,
            'auc': auc,
        })
        
        print(f"  Fold {fold+1}/{n_folds}: LL={ll:.1f}, Brier={brier:.4f}, AUC={auc:.4f}")
    
    print("-" * 50)
    
    cv_results = {
        'cv_log_likelihood_mean': np.mean(fold_ll),
        'cv_log_likelihood_std': np.std(fold_ll),
        'cv_brier_mean': np.mean(fold_brier),
        'cv_brier_std': np.std(fold_brier),
        'cv_auc_mean': np.nanmean(fold_auc),
        'cv_auc_std': np.nanstd(fold_auc),
        'n_folds': n_folds,
    }
    
    print(f"  CV Results: Brier={cv_results['cv_brier_mean']:.4f} ± {cv_results['cv_brier_std']:.4f}, "
          f"AUC={cv_results['cv_auc_mean']:.4f} ± {cv_results['cv_auc_std']:.4f}")
    print()
    
    return cv_results, fold_results


def plot_pred_histograms(p_pred_all, all_status, out_png, weights=None):
    """
    Histogram of predicted detection probabilities for dets vs non-dets.
    """
    plt.figure(figsize=(8,5))
    bins = np.linspace(0, 1, 50)
    
    if weights is None:
        plt.hist(p_pred_all[all_status == 1], bins=bins, alpha=0.5, label='Detected')
        plt.hist(p_pred_all[all_status == 0], bins=bins, alpha=0.5, label='Nondetected')
    else:
        plt.hist(p_pred_all[all_status == 1], bins=bins, weights=weights[all_status == 1],
                alpha=0.5, label='Detected')
        plt.hist(p_pred_all[all_status == 0], bins=bins, weights=weights[all_status == 0],
                alpha=0.5, label='Nondetected')
    
    plt.xlabel('Predicted detection probability')
    plt.ylabel('Count')
    plt.title('Histogram of predicted detection probabilities')
    plt.grid(True)
    plt.legend()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")

# =============== CLI ===============

PRED_CHOICES = ['fluence', 'scattering_time', 'width', 'dm']

@click.command(context_settings=dict(show_default=True))
@click.option('--predictor', 'predictors', multiple=True, type=click.Choice(PRED_CHOICES),
              default=['fluence','scattering_time','width','dm'],
              help="Predictor(s) to include. Repeat to select multiple.")
@click.option('--order', 'poly_order', default=3, type=int,
              help='Max polynomial order for the selection function.')
@click.option('--snr-cut', default=12.0, type=float,
              help='SNR cut for detected injections.')
@click.option('--sidelobe-cut', default=5.0, type=float,
              help='Absolute |beam_x| threshold; points beyond are removed.')
@click.option('--no-sidelobe-cut', is_flag=True, default=False,
              help='Disable sidelobe cut even if a threshold is specified.')
@click.option('--inj-file', default="/data/user-data/kmcgregor/09-2025_injections/output.json", type=str,
              help='Path to injections JSON (output.json).')
@click.option('--base-path', default="../chimefrb_selection/data/fits", type=str,
              help='Base directory for outputs. Subfolders created by dimension and predictors.')
@click.option('--rescaled/--no-rescaled', default=False,
              help='Also fit a rescaled version (0–1 scaled logs).')
@click.option('--compare-orders/--no-compare-orders', default=True,
              help='Scan *.npz in the target folder and compare orders (same predictors & cuts).')
@click.option('--cut-badtimes/--no-cut-badtimes', default=True,
              help='Apply time window cut where L2/L3 missed injections (Sept 11, 2025).')
@click.option('--plot/--no-plot', default=True, help='Save convergence and histogram plots.')
@click.option('--marginalize/--no-marginalize', default=True,
              help='For <4D fits, reweight to marginalize unobserved params over fiducial distribution.')
@click.option('--fiducial-params', default=None, type=str,
              help='Path to fiducial parameter file. If not provided, uses default parameters.')
@click.option('--weight-clip-min', default=1e-3, type=float,
              help='Minimum weight clipping value for marginalization.')
@click.option('--weight-clip-max', default=1e3, type=float,
              help='Maximum weight clipping value for marginalization.')
@click.option('--cross-validate/--no-cross-validate', 'do_cv', default=False,
              help='Run k-fold cross-validation to assess out-of-sample performance.')
@click.option('--cv-folds', default=5, type=int,
              help='Number of folds for cross-validation.')
def main(predictors, poly_order, snr_cut, sidelobe_cut, no_sidelobe_cut,
         inj_file, base_path, rescaled, compare_orders, cut_badtimes, plot,
         marginalize, fiducial_params, weight_clip_min, weight_clip_max,
         do_cv, cv_folds):
    """
    Fit logistic selection function with configurable predictors/order/cuts.
    
    For lower-dimensional fits (<4D), use --marginalize to reweight injections
    so that the distribution over unobserved parameters matches the fiducial
    astrophysical distribution. This ensures the fitted selection function
    represents the proper marginal over unobserved parameters.
    
    Saves to:
      <base_path>/<Nd>_selection_function/<predictor1>_<predictor2>_.../
    """
    # ---------------- Folder structure based on dimension & predictors ---------------
    if len(predictors) == 0:
        raise click.UsageError("You must specify at least one --predictor.")
    if len(predictors) == 4 and marginalize:
        print("Warning: All 4 predictors selected; ignoring --marginalize flag.")
        marginalize = False
    dim = len(predictors)
    dim_folder = f"{dim}d_selection_function"
    pred_folder = "_".join(predictors)
    data_path = os.path.join(base_path, dim_folder, pred_folder)
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(os.path.join(data_path, "plots"), exist_ok=True)
    print(f"Output directory: {data_path}")

    # ---------------- Read data ----------------
    print("Reading injections and detections...")
    injections, detections, nondetections = get_injections_detected(inj_file, return_nondets=True)
    print("Done reading injections and detections.")

    # ---------------- Cuts ----------------
    sl_cut = None if no_sidelobe_cut else sidelobe_cut
    (detected_fluence, detected_scattering_time, detected_width, detected_dm, detected_status,
     nondetected_fluence, nondetected_scattering_time, nondetected_width, nondetected_dm, nondetected_status,
     all_fluence, all_scattering_time, all_width, all_dm, all_status) = cut_detections_nondetections(
         detections, nondetections, snr_cut=snr_cut, sidelobe_cut=sl_cut, cut_badtimes=cut_badtimes
    )

    print("Done cutting detections and nondetections.")
    print(f"Total injections: {len(all_fluence)}")
    print(f"Detected injections: {np.sum(all_status)}")
    print(f"Nondetected injections: {np.sum(~all_status)}")

    # Bundle injection data for reweighting
    injection_data = {
        'fluence': all_fluence,
        'dm': all_dm,
        'width': all_width,
        'scattering_time': all_scattering_time,
    }

    # ---------------- Marginalization reweighting (for <4D fits) ----------------
    sample_weights = None
    reweighting_metadata = {}
    is_marginal_fit = (dim < 4) and marginalize
    
    if is_marginal_fit:
        print("\n" + "="*60)
        print(f"Computing marginal weights for {dim}D fit...")
        print(f"Marginalizing over unobserved parameters using fiducial distribution")
        print("="*60)
        
        # Load fiducial parameters
        if fiducial_params is not None:
            print(f"Loading fiducial parameters from: {fiducial_params}")
            fid_params = load_fiducial_params(fiducial_params)
        else:
            print("Using default fiducial parameters")
            fid_params = DEFAULT_FIDUCIAL_PARAMS
        
        # Compute marginal weights
        sample_weights, reweighting_metadata = compute_marginal_weights(
            injection_data,
            observed_predictors=list(predictors),
            fiducial_params=fid_params,
            normalize=True,
            clip_weights=(weight_clip_min, weight_clip_max),
        )
        
        print(f"Effective sample size: {reweighting_metadata['effective_sample_size']:.1f} "
              f"({reweighting_metadata['ess_fraction']*100:.1f}% of {len(all_fluence)})")
        print(f"Weight range: [{sample_weights.min():.3e}, {sample_weights.max():.3e}]")
        print("="*60 + "\n")
        
        # Generate diagnostic plots
        if plot:
            plot_dir = os.path.join(data_path, "plots")
            plot_reweighting_diagnostics(
                injection_data, sample_weights, reweighting_metadata, plot_dir
            )
    
    elif dim < 4 and not marginalize:
        print("\nWARNING: Fitting <4D model without marginalization.")
        print("The resulting selection function will depend on the injection distribution,")
        print("not the astrophysical fiducial distribution.\n")

    # ---------------- Build design matrix for selected predictors ----------------
    logs_all = {
        'fluence':          np.log10(all_fluence),
        'scattering_time':  np.log10(all_scattering_time),
        'width':            np.log10(all_width),
        'dm':               np.log10(all_dm),
    }
    selected_logs = [logs_all[p] for p in predictors]
    predictor_names = list(predictors)

    anchor = selected_logs[0]
    X, term_names = build_design_matrix(anchor, selected_logs, poly_order,
                                        predictor_names=predictor_names, return_names=True)
    Y = all_status.astype(int)
    print(X.shape, "shape of design matrix for", predictor_names)
    print("Condition number of design matrix:", np.linalg.cond(X))

    # ---------------- Cross-validation (if enabled) ----------------
    cv_results = None
    if do_cv:
        cv_results, fold_results = cross_validate_logistic(
            X, Y, n_folds=cv_folds, sample_weights=sample_weights
        )
        
        # Save fold-level results
        cv_tag = f"{'-'.join(predictor_names)}_order{poly_order}_snr{int(snr_cut)}_sl{('none' if sl_cut is None else str(sl_cut))}"
        if is_marginal_fit:
            cv_tag += "_reweighted"
        cv_folds_csv = os.path.join(data_path, f"cv_folds_{cv_tag}.csv")
        pd.DataFrame(fold_results).to_csv(cv_folds_csv, index=False)
        print(f"Saved CV fold results: {cv_folds_csv}")

    # ---------------- Filenames / tags ----------------
    tag = f"{'-'.join(predictor_names)}_order{poly_order}_snr{int(snr_cut)}_sl{('none' if sl_cut is None else str(sl_cut))}"
    if is_marginal_fit:
        tag += "_reweighted"
    
    out_npz   = os.path.join(data_path, f"IRLS_output_{tag}.npz")
    conv_plot = os.path.join(data_path, "plots", f"irls_convergence_{tag}.png")
    hist_plot = os.path.join(data_path, "plots", f"p_hist_{tag}.png")
    summary_csv = os.path.join(data_path, f"summary_{tag}.csv")

    # ---------------- Fit (cache-aware) ----------------
    if not os.path.exists(out_npz):
        print(f"Fitting model; cache not found: {out_npz}")
        beta_est, standard_errors, cov_matrix = reweighted_least_squares_sparse(
            X, Y, return_errors=True, plot=plot, verbose=True,
            plot_path=conv_plot if plot else None,
            sample_weights=sample_weights
        )
        
        save_dict = {
            'beta_est': beta_est,
            'standard_errors': standard_errors,
            'cov_matrix': cov_matrix,
            'term_names': np.array(term_names),
            'predictors': np.array(predictor_names),
            'marginalized': is_marginal_fit,
        }
        if is_marginal_fit:
            save_dict.update({
                'effective_sample_size': reweighting_metadata['effective_sample_size'],
                'ess_fraction': reweighting_metadata['ess_fraction'],
                'marginalized_predictors': np.array(reweighting_metadata['marginalized_predictors']),
            })
        
        np.savez(out_npz, **save_dict)
        print(f"Saved: {out_npz}")
    else:
        print(f"Loading cached fit: {out_npz}")
        with np.load(out_npz, allow_pickle=True) as data:
            beta_est = data['beta_est']
            standard_errors = data['standard_errors']
            cov_matrix = data['cov_matrix']
            if 'term_names' in data:
                term_names = list(data['term_names'])
            if 'predictors' in data:
                predictor_names = list(data['predictors'])

    # ---------------- Diagnostics ----------------
    ll_model = log_likelihood_logistic(X, Y, beta_est, weights=sample_weights)
    X_null = np.ones((len(Y), 1))
    beta_null = reweighted_least_squares_sparse(X_null, Y, plot=False, sample_weights=sample_weights)
    ll_null = log_likelihood_logistic(X_null, Y, beta_null, weights=sample_weights)

    r2_dev = 1 - (ll_model / ll_null)
    cox_snell_r2 = 1 - np.exp((ll_null - ll_model) / len(Y))
    nagelkerke_r2 = cox_snell_r2 / (1 - np.exp(ll_null / len(Y)))
    p_pred_all = 1.0 / (1.0 + np.exp(-(X @ beta_est)))
    brier_score = brier_score_loss(Y, p_pred_all, sample_weight=sample_weights)
    pearson_resids = pearson_residuals(Y, p_pred_all)
    ss_resid = np.sum(pearson_resids ** 2)

    print(f"McFadden R^2:     {r2_dev:.4f}")
    print(f"Cox-Snell R^2:    {cox_snell_r2:.4f}")
    print(f"Nagelkerke R^2:   {nagelkerke_r2:.4f}")
    print(f"Brier Score:      {brier_score:.4f}")
    print(f"Sum Sq Pearson:   {ss_resid:.4f}")

    summary_dict = {
        "predictors": "-".join(predictor_names),
        "order": poly_order,
        "snr_cut": snr_cut,
        "sidelobe_cut": (None if sl_cut is None else sl_cut),
        "n": len(Y),
        "n_detected": int(np.sum(Y)),
        "loglik": ll_model,
        "McFadden_R2": r2_dev,
        "CoxSnell_R2": cox_snell_r2,
        "Nagelkerke_R2": nagelkerke_r2,
        "Brier": brier_score,
        "SumSqPearson": ss_resid,
        "marginalized": is_marginal_fit,
    }
    
    if is_marginal_fit:
        summary_dict.update({
            "effective_sample_size": reweighting_metadata['effective_sample_size'],
            "ess_fraction": reweighting_metadata['ess_fraction'],
            "marginalized_over": "-".join(reweighting_metadata['marginalized_predictors']),
        })
    
    # Add CV results to summary if available
    if cv_results is not None:
        summary_dict.update({
            "cv_brier_mean": cv_results['cv_brier_mean'],
            "cv_brier_std": cv_results['cv_brier_std'],
            "cv_auc_mean": cv_results['cv_auc_mean'],
            "cv_auc_std": cv_results['cv_auc_std'],
            "cv_folds": cv_results['n_folds'],
        })
    
    pd.DataFrame([summary_dict]).to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    if plot:
        plot_pred_histograms(p_pred_all, Y, hist_plot, weights=sample_weights)

    # ---------------- Optional: rescaled fit ----------------
    if rescaled:
        print("Running rescaled fit (0–1 per selected log predictor)...")
        minmax = [(arr.min(), arr.max()) for arr in selected_logs]
        scaled = [(arr - mn) / (mx - mn) if (mx > mn) else np.zeros_like(arr)
                  for (arr, (mn, mx)) in zip(selected_logs, minmax)]
        Xs, names_scaled = build_design_matrix(scaled[0], scaled, poly_order,
                                               predictor_names=[f"{n}_scaled" for n in predictor_names],
                                               return_names=True)
        beta_s, se_s, cov_s = reweighted_least_squares_sparse(
            Xs, Y, return_errors=True, plot=plot,
            plot_path=os.path.join(data_path, "plots", f"irls_convergence_{tag}_rescaled.png") if plot else None,
            sample_weights=sample_weights
        )
        out_csv = os.path.join(data_path, f"IRLS_output_{tag}_rescaled.csv")
        pd.DataFrame({
            "Coefficient Name": names_scaled,
            "Beta Estimate": beta_s,
            "Standard Error": se_s
        }).to_csv(out_csv, index=False)
        print("Rescaled logistic regression completed and saved to", out_csv)

    # ---------------- Optional: compare orders ----------------
    if compare_orders:
        print("Comparing orders found on disk with matching tag stem...")
        pattern = os.path.join(data_path, f"IRLS_output_{'-'.join(predictor_names)}_order*.npz")
        paths = sorted(glob.glob(pattern))
        snr_str = f"_snr{int(snr_cut)}_"
        sl_str = f"_sl{('none' if sl_cut is None else str(sl_cut))}"
        marginal_str = "_reweighted" if is_marginal_fit else ""
        
        paths = [p for p in paths if snr_str in p and sl_str in p]
        
        # Filter by marginalization status
        if is_marginal_fit:
            paths = [p for p in paths if "_reweighted" in p]
        else:
            paths = [p for p in paths if "_reweighted" not in p]

        results = []
        for pth in paths:
            try:
                order = int(os.path.basename(pth).split("order")[-1].split("_")[0])
            except Exception:
                continue
            with np.load(pth, allow_pickle=True) as dat:
                beta = dat['beta_est']
            Xo = build_design_matrix(anchor, selected_logs, order,
                                     predictor_names=predictor_names, return_names=False)
            ll = log_likelihood_logistic(Xo, Y, beta, weights=sample_weights)
            k = len(beta)
            n = len(Y)
            aic, bic = compute_information_criteria(ll, k, n)

            r2_dev_o = 1 - (ll / ll_null)
            cox_snell_o = 1 - np.exp((ll_null - ll) / n)
            nagelkerke_o = cox_snell_o / (1 - np.exp(ll_null / n))
            p_pred = 1.0 / (1.0 + np.exp(-(Xo @ beta)))
            ssr = np.sum(pearson_residuals(Y, p_pred, verbose=False) ** 2)
            
            # Load CV results if available
            cv_summary_path = os.path.join(data_path, f"summary_{'-'.join(predictor_names)}_order{order}_snr{int(snr_cut)}_sl{('none' if sl_cut is None else str(sl_cut))}{marginal_str}.csv")
            cv_brier = np.nan
            cv_auc = np.nan
            if os.path.exists(cv_summary_path):
                try:
                    cv_df = pd.read_csv(cv_summary_path)
                    if 'cv_brier_mean' in cv_df.columns:
                        cv_brier = cv_df['cv_brier_mean'].iloc[0]
                    if 'cv_auc_mean' in cv_df.columns:
                        cv_auc = cv_df['cv_auc_mean'].iloc[0]
                except Exception:
                    pass

            results.append({
                "Order": order,
                "Number of Parameters": k,
                "AIC": aic,
                "BIC": bic,
                "Log-Likelihood": ll,
                "Deviance R^2": r2_dev_o,
                "Cox-Snell R^2": cox_snell_o,
                "Nagelkerke R^2": nagelkerke_o,
                "Sum of Squared Residuals": ssr,
                "CV Brier": cv_brier,
                "CV AUC": cv_auc,
            })

        if results:
            df = pd.DataFrame(results).sort_values(by="Number of Parameters").reset_index(drop=True)
            cmp_base = f"model_comparison_{'-'.join(predictor_names)}_snr{int(snr_cut)}_sl{('none' if sl_cut is None else str(sl_cut))}"
            if is_marginal_fit:
                cmp_base += "_reweighted"
            cmp_csv = os.path.join(data_path, f"{cmp_base}.csv")
            df.to_csv(cmp_csv, index=False)
            print(f"Saved order comparison: {cmp_csv}")

            # Pairwise LRT for consecutive orders
            lrt_rows = []
            for i in range(len(df)-1):
                ll_null_i = df.iloc[i]["Log-Likelihood"]
                ll_alt_i  = df.iloc[i+1]["Log-Likelihood"]
                k_null_i  = df.iloc[i]["Number of Parameters"]
                k_alt_i   = df.iloc[i+1]["Number of Parameters"]
                lr_stat, p_value = likelihood_ratio_test(ll_null_i, ll_alt_i, k_null_i, k_alt_i)
                label = f"{int(df.iloc[i]['Order'])} -> {int(df.iloc[i+1]['Order'])}"
                lrt_rows.append({
                    "Label": label,
                    "Model Order (Null)": int(df.iloc[i]['Order']),
                    "Model Order (Alt)": int(df.iloc[i+1]['Order']),
                    "LR Statistic": lr_stat,
                    "P-value": p_value,
                })
            if lrt_rows:
                lrt_base = f"likelihood_ratio_test_{'-'.join(predictor_names)}_snr{int(snr_cut)}_sl{('none' if sl_cut is None else str(sl_cut))}"
                if is_marginal_fit:
                    lrt_base += "_reweighted"
                lrt_csv = os.path.join(data_path, f"{lrt_base}.csv")
                pd.DataFrame(lrt_rows).to_csv(lrt_csv, index=False)
                print(f"Saved LRT table: {lrt_csv}")
        else:
            print("No matching order files found to compare.")



if __name__ == "__main__":
    main()
