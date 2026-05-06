"""
Reweighting utilities for selection function fitting.

Provides:
1. Fiducial model sampling for marginalization over unobserved dimensions
2. Importance weights for lower-dimensional selection function fits
"""

import numpy as np
from scipy.stats import lognorm

# ============================================================
# Default fiducial parameters (fit to observed FRB population)
# Units: fluence in Jy ms, DM in pc/cm³, width/scattering in SECONDS
# ============================================================

DEFAULT_FIDUCIAL_PARAMS = {
    'alpha': -1.203108427072943,          # Fluence power-law index (negative)
    'F_min': 0.1,                          # Jy ms, minimum fluence
    'F_max': 1e4,                          # Jy ms, maximum fluence
    'dm_shape': 0.6076516890745998,        # Log-normal shape (sigma) for DM
    'dm_scale': 534.4727066208081,         # Log-normal scale (median) for DM in pc/cm³
    'width_shape': 1.1012853240184415,     # Log-normal shape for width
    'width_scale': 0.0007389903164298552,  # Log-normal scale for width in SECONDS
    'scat_shape': 1.9262242015344677,      # Log-normal shape for scattering
    'scat_scale': 0.012965230586403698,    # Log-normal scale for scattering in SECONDS @ 600 MHz
    'scat_ref_freq_mhz': 600.0,            # Reference frequency for scattering
    'target_freq_mhz': 600.0,              # Target frequency (set equal to avoid scaling)
}

# Canonical predictor names
ALL_PREDICTORS = ['fluence', 'scattering_time', 'width', 'dm']


def load_fiducial_params(filepath):
    """
    Load fiducial parameters from a file.
    
    Supports:
    - .json files
    - .dat/.txt files with 'key = value' or 'key: value' format
    
    Parameters
    ----------
    filepath : str
        Path to parameter file
    
    Returns
    -------
    params : dict
        Fiducial parameters
    """
    import os
    
    params = DEFAULT_FIDUCIAL_PARAMS.copy()
    
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.json':
        import json
        with open(filepath, 'r') as f:
            loaded = json.load(f)
        params.update(loaded)
    else:
        # Parse key=value or key: value format
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, val = line.split('=', 1)
                elif ':' in line:
                    key, val = line.split(':', 1)
                else:
                    continue
                key = key.strip()
                val = val.strip()
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val
    
    return params


def sample_fiducial_parameter(param_name, n_samples, fiducial_params=None, rng=None):
    """
    Draw samples from the fiducial distribution for a single parameter.
    
    Parameters
    ----------
    param_name : str
        One of 'fluence', 'dm', 'width', 'scattering_time'
    n_samples : int
        Number of samples to draw
    fiducial_params : dict, optional
        Fiducial parameters. Uses defaults if None.
    rng : np.random.Generator, optional
        Random number generator for reproducibility
    
    Returns
    -------
    samples : ndarray
        Samples from the fiducial distribution (in standard units: Jy ms, pc/cm³, ms)
    """
    if fiducial_params is None:
        fiducial_params = DEFAULT_FIDUCIAL_PARAMS.copy()
    else:
        fiducial_params = fiducial_params.copy()
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Convert width and scattering scales from seconds to ms
    width_scale_ms = fiducial_params['width_scale'] * 1000.0
    scat_scale_ms = fiducial_params['scat_scale'] * 1000.0
    
    # Scale scattering to target frequency
    freq_ratio = fiducial_params['scat_ref_freq_mhz'] / fiducial_params['target_freq_mhz']
    scat_scale_ms_target = scat_scale_ms * (freq_ratio ** 4)
    
    if param_name == 'fluence':
        # Power-law distribution: p(F) ∝ F^alpha
        alpha = fiducial_params['alpha']
        F_min = fiducial_params['F_min']
        F_max = fiducial_params.get('F_max', 1e4)
        
        # Inverse CDF sampling for power-law
        u = rng.uniform(0, 1, n_samples)
        if alpha == -1:
            samples = F_min * (F_max / F_min) ** u
        else:
            ap1 = alpha + 1
            samples = (F_min**ap1 + u * (F_max**ap1 - F_min**ap1)) ** (1/ap1)
    
    elif param_name == 'dm':
        # Log-normal distribution
        shape = fiducial_params['dm_shape']
        scale = fiducial_params['dm_scale']
        samples = rng.lognormal(mean=np.log(scale), sigma=shape, size=n_samples)
    
    elif param_name == 'width':
        # Log-normal distribution (output in ms)
        shape = fiducial_params['width_shape']
        samples = rng.lognormal(mean=np.log(width_scale_ms), sigma=shape, size=n_samples)
    
    elif param_name == 'scattering_time':
        # Log-normal distribution (output in ms at target frequency)
        shape = fiducial_params['scat_shape']
        samples = rng.lognormal(mean=np.log(scat_scale_ms_target), sigma=shape, size=n_samples)
    
    else:
        raise ValueError(f"Unknown parameter: {param_name}. "
                        f"Must be one of {ALL_PREDICTORS}")
    
    return samples


def sample_fiducial_distribution(param_names, n_samples, fiducial_params=None, seed=None):
    """
    Draw samples from the joint fiducial distribution for multiple parameters.
    Assumes parameters are independent.
    
    Parameters
    ----------
    param_names : list of str
        Parameters to sample, e.g., ['dm', 'width', 'scattering_time']
    n_samples : int
        Number of samples to draw
    fiducial_params : dict, optional
        Fiducial parameters. Uses defaults if None.
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    samples : dict
        Dictionary mapping parameter names to sample arrays
    """
    rng = np.random.default_rng(seed)
    
    samples = {}
    for param in param_names:
        samples[param] = sample_fiducial_parameter(
            param, n_samples, fiducial_params, rng
        )
    
    return samples

def compute_marginal_weights(injection_data, observed_predictors, 
                             fiducial_params=None, n_mc_samples=1000,
                             normalize=True, clip_weights=(1e-3, 1e3),
                             seed=42):
    """
    Compute importance weights for fitting a lower-dimensional selection function
    by marginalizing over unobserved parameters using the fiducial distribution.
    
    Weight for injection i:
        w_i = p_fid(x_marg_i) / p_inj(x_marg_i)
    
    where x_marg_i are the marginalized (unobserved) parameters for injection i.
    
    Parameters
    ----------
    injection_data : dict
        Dictionary with keys 'fluence', 'dm', 'width', 'scattering_time'
        containing arrays of injection parameters
    observed_predictors : list of str
        Predictors that WILL be in the lower-dimensional model
    fiducial_params : dict, optional
        Fiducial model parameters
    n_mc_samples : int
        Not used currently, kept for API compatibility
    normalize : bool
        If True, normalize weights to sum to n_samples
    clip_weights : tuple
        (min, max) for weight clipping
    seed : int
        Random seed
    
    Returns
    -------
    weights : ndarray
        Importance weights for each injection
    metadata : dict
        Diagnostic information
    """
    if fiducial_params is None:
        fiducial_params = DEFAULT_FIDUCIAL_PARAMS.copy()
    else:
        fiducial_params = fiducial_params.copy()
    
    # Convert fiducial width/scattering from seconds to ms for comparison
    fiducial_params_ms = fiducial_params.copy()
    fiducial_params_ms['width_scale'] = fiducial_params['width_scale'] * 1000.0
    fiducial_params_ms['scat_scale'] = fiducial_params['scat_scale'] * 1000.0
    
    # Scale scattering to target frequency
    freq_ratio = fiducial_params['scat_ref_freq_mhz'] / fiducial_params['target_freq_mhz']
    fiducial_params_ms['scat_scale'] *= (freq_ratio ** 4)
    
    # Determine which parameters to marginalize over
    marginalized_predictors = [p for p in ALL_PREDICTORS if p not in observed_predictors]
    
    if len(marginalized_predictors) == 0:
        # No marginalization needed (4D case)
        n_samples = len(injection_data['fluence'])
        weights = np.ones(n_samples)
        metadata = {
            'effective_sample_size': float(n_samples),
            'ess_fraction': 1.0,
            'marginalized_predictors': [],
            'observed_predictors': observed_predictors,
        }
        return weights, metadata
    
    print(f"Marginalizing over: {marginalized_predictors}")
    print(f"Observed predictors: {observed_predictors}")
    
    n_samples = len(injection_data['fluence'])
    log_weights = np.zeros(n_samples)
    
    # For each marginalized parameter, compute log(p_fid / p_inj)
    for param in marginalized_predictors:
        values = np.asarray(injection_data[param]).flatten()
        
        print(f"\n  Processing {param}...")
        print(f"    Value range: [{values.min():.3e}, {values.max():.3e}]")
        
        # Estimate injection PDF via histogram in log-space
        log_values = np.log10(values)
        hist, bin_edges = np.histogram(log_values, bins=100, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Evaluate injection PDF at each point (in log10 space, then transform)
        p_inj_log10 = np.interp(log_values, bin_centers, hist, left=1e-10, right=1e-10)
        # Transform: p(x) = p(log10(x)) / (x * ln(10))
        p_inj = p_inj_log10 / (values * np.log(10))
        p_inj = np.maximum(p_inj, 1e-30)
        
        print(f"    p_inj range: [{p_inj.min():.3e}, {p_inj.max():.3e}]")
        
        # Evaluate fiducial PDF at each point
        if param == 'fluence':
            alpha = fiducial_params['alpha']
            F_min = fiducial_params['F_min']
            F_max = fiducial_params.get('F_max', 1e4)
            
            print(f"    Fiducial power-law: alpha={alpha}, F_min={F_min}, F_max={F_max}")
            
            # Power-law: p(F) ∝ F^alpha, normalized over [F_min, F_max]
            ap1 = alpha + 1
            
            if np.abs(ap1) < 1e-10:  # α ≈ -1, special case
                log_norm = -np.log(np.log(F_max / F_min))
                log_p_fid = log_norm - np.log(values)
            else:
                # For α < -1: F_min^(α+1) > F_max^(α+1), so denominator is positive
                # For α > -1: F_max^(α+1) > F_min^(α+1), so denominator is positive
                # Either way, we need |ap1| / |F_max^ap1 - F_min^ap1|
                denom = F_max**ap1 - F_min**ap1
                norm = ap1 / denom  # This can be negative if ap1 < 0 and denom > 0
                # Take absolute value to ensure positive PDF
                norm = np.abs(norm)
                log_p_fid = np.log(norm) + alpha * np.log(values)
            
            p_fid = np.exp(log_p_fid)
            
            # Zero outside valid range
            outside_range = (values < F_min) | (values > F_max)
            p_fid = np.where(outside_range, 1e-30, p_fid)
            
            print(f"    Injections outside [{F_min}, {F_max}]: {outside_range.sum()}")
        
        elif param == 'dm':
            shape = fiducial_params['dm_shape']
            scale = fiducial_params['dm_scale']
            print(f"    Fiducial log-normal: shape={shape}, scale={scale}")
            p_fid = lognorm.pdf(values, s=shape, scale=scale)
        
        elif param == 'width':
            shape = fiducial_params_ms['width_shape']
            scale = fiducial_params_ms['width_scale']
            print(f"    Fiducial log-normal: shape={shape}, scale={scale} ms")
            p_fid = lognorm.pdf(values, s=shape, scale=scale)
        
        elif param == 'scattering_time':
            shape = fiducial_params_ms['scat_shape']
            scale = fiducial_params_ms['scat_scale']
            print(f"    Fiducial log-normal: shape={shape}, scale={scale} ms")
            p_fid = lognorm.pdf(values, s=shape, scale=scale)
        
        p_fid = np.maximum(p_fid, 1e-30)
        
        print(f"    p_fid range: [{p_fid.min():.3e}, {p_fid.max():.3e}]")
        
        # Compute weight contribution from this parameter
        weight_contribution = p_fid / p_inj
        print(f"    Weight contribution range: [{weight_contribution.min():.3e}, {weight_contribution.max():.3e}]")
        
        # Accumulate log-weight
        log_weights += np.log(p_fid) - np.log(p_inj)
    
    # Convert to weights
    # Subtract median for numerical stability (not max, to be more robust to outliers)
    log_weights -= np.median(log_weights)
    weights = np.exp(log_weights)
    
    print(f"\n  Raw weights range: [{weights.min():.3e}, {weights.max():.3e}]")
    
    # Clip extreme weights
    n_clipped_low = 0
    n_clipped_high = 0
    if clip_weights is not None:
        n_clipped_low = int(np.sum(weights < clip_weights[0]))
        n_clipped_high = int(np.sum(weights > clip_weights[1]))
        weights = np.clip(weights, clip_weights[0], clip_weights[1])
        print(f"  Clipped: {n_clipped_low} low, {n_clipped_high} high")
    
    # Normalize
    if normalize:
        weights = weights / weights.sum() * n_samples
    
    # Compute effective sample size
    ess = np.sum(weights)**2 / np.sum(weights**2)
    ess_fraction = ess / n_samples
    
    print(f"\n  Final weights range: [{weights.min():.3e}, {weights.max():.3e}]")
    print(f"  Effective sample size: {ess:.1f} ({ess_fraction*100:.1f}%)")
    
    metadata = {
        'effective_sample_size': ess,
        'ess_fraction': ess_fraction,
        'weight_mean': np.mean(weights),
        'weight_std': np.std(weights),
        'weight_min': np.min(weights),
        'weight_max': np.max(weights),
        'n_clipped_low': n_clipped_low,
        'n_clipped_high': n_clipped_high,
        'marginalized_predictors': marginalized_predictors,
        'observed_predictors': observed_predictors,
    }
    
    return weights, metadata


def plot_reweighting_diagnostics(injection_data, weights, metadata, output_dir,
                                 marginalized_predictors=None):
    """
    Plot diagnostics for marginalization reweighting.
    
    Parameters
    ----------
    injection_data : dict
        Dictionary with injection parameter arrays
    weights : ndarray
        Importance weights
    metadata : dict
        Metadata from compute_marginal_weights
    output_dir : str
        Directory to save plots
    marginalized_predictors : list, optional
        Which predictors were marginalized. If None, read from metadata.
    """
    import matplotlib.pyplot as plt
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    if marginalized_predictors is None:
        marginalized_predictors = metadata.get('marginalized_predictors', [])
    
    if len(marginalized_predictors) == 0:
        print("No marginalized predictors - skipping reweighting diagnostics.")
        return
    
    n_params = len(marginalized_predictors)
    fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 4))
    if n_params == 1:
        axes = [axes]
    
    for ax, param in zip(axes, marginalized_predictors):
        values = injection_data[param]
        
        # Original distribution
        ax.hist(np.log10(values), bins=80, alpha=0.5, density=False,
                label='Injection', color='blue')
        
        # Reweighted distribution
        ax.hist(np.log10(values), bins=80, weights=weights, alpha=0.5, density=False,
                label='Reweighted', color='orange')
        
        ax.set_xlabel(f'log10({param})')
        ax.set_ylabel('Counts')
        ax.set_yscale("log")
        ax.set_title(f'{param} (marginalized)')
        ax.legend()
    
    plt.suptitle(f"ESS: {metadata['effective_sample_size']:.1f} "
                 f"({metadata['ess_fraction']*100:.1f}%)")
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'marginal_reweighting_diagnostics.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")
    
    # Weight distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log10(weights + 1e-10), bins=80, edgecolor='black', alpha=0.7)
    ax.axvline(np.log10(np.mean(weights)), color='red', linestyle='--', 
               label=f'Mean: {np.mean(weights):.2f}')
    ax.set_xlabel('log10(weight)')
    ax.set_ylabel('Count')
    ax.set_title(f'Weight Distribution (ESS={metadata["effective_sample_size"]:.1f})')
    ax.legend()
    
    out_path = os.path.join(output_dir, 'marginal_weight_distribution.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")