import os
from pathlib import Path
import numpy as np
import h5py
import json

PROP_ORDER = ('fluence', 'scattering_time', 'width', 'dm')

def data_dir() -> str:
    """
    Return the on-disk path of the installed package's data directory,
    without using importlib.resources (works on older Python).
    """
    return str(Path(__file__).resolve().parent / "data")

# Base dirs (dimension-agnostic). Can be overridden with env vars.
DEFAULT_MODELS_DIR = os.environ.get(
    "CHIMEFRB_SELECTION_MODELS_DIR",
    str(Path(data_dir()) / "fits")
)
DEFAULT_KNN_DIR = os.environ.get(
    "CHIMEFRB_SELECTION_KNN_DIR",
    str(Path(data_dir()) / "masks")
)

DEFAULT_LOCAL_DATA_PATH = "/data/user-data/kmcgregor/selection_func_models/4d_selection_function/"
DEFAULT_LOCAL_INJ_FILE  = "/data/user-data/kmcgregor/09-2025_injections/output.json"

# --- Accepted aliases for each canonical predictor name ---
PREDICTOR_ALIASES = {
    'fluence': {'fluence', 'fluence_jy_ms', 'f', 'F', 'F_jy_ms'},
    'scattering_time': {'scattering_time', 'tau_1_ghz_ms', 'tau_ms', 'tau', 'tau_1GHz_ms'},
    'width': {'width', 'pulse_width_ms', 'w_ms', 'w'},
    'dm': {'dm', 'dispersion_measure', 'DM'},
}

# Build reverse map: alias -> canonical
_ALIAS_TO_CANON = {alias: canon
                   for canon, aliases in PREDICTOR_ALIASES.items()
                   for alias in aliases}

def canonicalize_props(props: dict) -> dict:
    """
    Return a new dict with keys converted to canonical predictor names
    when they match a known alias. Unknown keys are preserved as-is.
    If both an alias and canonical appear, canonical wins.
    """
    out = {}
    # First copy through any already-canonical keys
    for k, v in props.items():
        if k in PROP_ORDER:
            out[k] = v
    # Then map aliases that are not already present canonically
    for k, v in props.items():
        if k in PROP_ORDER:
            continue
        canon = _ALIAS_TO_CANON.get(k)
        if canon is not None and canon not in out:
            out[canon] = v
    return out

def get_prop_vector(props: dict, names=PROP_ORDER) -> np.ndarray:
    """
    Build a vector in the given canonical `names` order from a props dict
    that may contain aliases. Raises if any required value is missing.
    """
    props_canon = canonicalize_props(props)
    missing = [n for n in names if n not in props_canon]
    if missing:
        raise KeyError(f"Missing required predictors: {missing}. "
                       f"Accepted aliases: {PREDICTOR_ALIASES}")
    return np.array([props_canon[n] for n in names], dtype=float)

def read_h5_file(file_path,key="frb"):
    h5_file = h5py.File(file_path, "r")
    data = h5_file[key]
    return data

def read_json_file(file_path):
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data

def get_injections_detected(file_path, return_nondets=False):
    data = read_json_file(file_path)
    if not data:
        return
    
    # Extract timestamps
    injections = data.get("injections", [])
    detections = data.get("detections", [])

    # Create a dictionary for detections with "det_id" as the key and the whole dictionary as the value
    detections_dict = {det["det_id"]: det for det in detections}

    # Filter injections with injection_time after 2:20 UTC on 03-06-2025
    #injections = [inj for inj in injections if datetime.strptime(inj["injection_time"], "%Y-%m-%dT%H:%M:%S.%fZ") > datetime(2025, 3, 6, 1, 20)]

    #inj_ids = [inj["id"] for inj in injections]
    detected_ids = set([det["det_id"] for det in detections])

    detected_mask = np.array([inj["id"] in detected_ids for inj in injections])

    injections = np.array(injections)
    nondetections = injections[~detected_mask]
    detections = injections[detected_mask]

    for inj in detections:
        inj_id = inj["id"]
        if inj_id in detected_ids:
            inj["extra_detection_parameters"] = detections_dict[inj_id]

    if return_nondets:
        return injections, detections, nondetections

    else:
        return injections, detections

def cut_detections_nondetections(detections, nondetections, snr_cut=None, sidelobe_cut=None, cut_badtimes=True):
    """
    Your existing cut function (unchanged except it's here for a self-contained script).
    """
    print(len(detections)+len(nondetections), "total injections before cuts")
    print(len(detections), "detections before cuts")
    print(len(nondetections), "nondetections before cuts")

    if cut_badtimes:
        window_start = np.datetime64("2025-09-11T05:29:05")
        window_end   = np.datetime64("2025-09-11T14:50:12")

        def in_bad_window(rec):
            ts = np.datetime64(rec["injection_time"].rstrip("Z"))
            return (ts >= window_start) and (ts <= window_end)

        detections = [d for d in detections if not in_bad_window(d)]
        nondetections = [n for n in nondetections if not in_bad_window(n)]

    detected_fluence = np.array([det['fluence_jy_ms'] for det in detections])
    detected_scattering_time = np.array([det["extra_injection_parameters"]['tau_1_ghz_ms'] for det in detections])
    detected_width = np.array([det['pulse_width_ms'] for det in detections])
    detected_dm = np.array([det['dm'] for det in detections])
    detected_snr = np.array([det["extra_detection_parameters"]['combined_snr'] for det in detections])
    detected_status = np.ones(len(detected_fluence), dtype=bool)
    detected_beam_x = np.array([det["extra_injection_parameters"]["beam_x"] for det in detections])

    nondetected_fluence = np.array([inj['fluence_jy_ms'] for inj in nondetections])
    nondetected_scattering_time = np.array([inj["extra_injection_parameters"]['tau_1_ghz_ms'] for inj in nondetections])
    nondetected_width = np.array([inj['pulse_width_ms'] for inj in nondetections])
    nondetected_dm = np.array([inj['dm'] for inj in nondetections])
    nondetected_status = np.zeros(len(nondetected_fluence), dtype=bool)
    nondetected_beam_x = np.array([inj["extra_injection_parameters"]["beam_x"] for inj in nondetections])
    nondetected_beam_id = np.array([inj["beams"] for inj in nondetections])

    bad_beams = {
        1248,1249,1250,1251,1252,1253,1254,1255,
        2024,2025,2026,2027,2028,2029,2030,2031,
        2080,2081,2082,2083,2084,2085,2086,2087,
        2120,2121,2122,2123,2124,2125,2126,2127,
        3240,3241,3242,3243,3244,3245,3246,3247,
        3248,3249,3250,3251,3252,3253,3254,3255,
        72,73,74,75
    }

    if snr_cut is not None:
        detected_status[detected_snr < snr_cut] = False

    mask_good_beams = np.array([beam_id[0] not in bad_beams for beam_id in nondetected_beam_id])
    nondetected_fluence = nondetected_fluence[mask_good_beams]
    nondetected_scattering_time = nondetected_scattering_time[mask_good_beams]
    nondetected_width = nondetected_width[mask_good_beams]
    nondetected_dm = nondetected_dm[mask_good_beams]
    nondetected_status = nondetected_status[mask_good_beams]
    nondetected_beam_x = nondetected_beam_x[mask_good_beams]
    nondetected_beam_id = nondetected_beam_id[mask_good_beams]

    all_fluence = np.concatenate((detected_fluence, nondetected_fluence))
    all_scattering_time = np.concatenate((detected_scattering_time, nondetected_scattering_time))
    all_width = np.concatenate((detected_width, nondetected_width))
    all_dm = np.concatenate((detected_dm, nondetected_dm))
    all_status = np.concatenate((detected_status, nondetected_status))
    all_beam_x = np.concatenate((detected_beam_x, nondetected_beam_x))

    if sidelobe_cut is not None:
        mask = ~np.logical_or(detected_beam_x > sidelobe_cut, detected_beam_x < -sidelobe_cut)
        detected_fluence = detected_fluence[mask]
        detected_scattering_time = detected_scattering_time[mask]
        detected_width = detected_width[mask]
        detected_dm = detected_dm[mask]
        detected_status = detected_status[mask]

        mask_nondet = ~np.logical_or(nondetected_beam_x > sidelobe_cut, nondetected_beam_x < -sidelobe_cut)
        nondetected_fluence = nondetected_fluence[mask_nondet]
        nondetected_scattering_time = nondetected_scattering_time[mask_nondet]
        nondetected_width = nondetected_width[mask_nondet]
        nondetected_dm = nondetected_dm[mask_nondet]
        nondetected_status = nondetected_status[mask_nondet]

        mask_all = ~np.logical_or(all_beam_x > sidelobe_cut, all_beam_x < -sidelobe_cut)
        all_fluence = all_fluence[mask_all]
        all_scattering_time = all_scattering_time[mask_all]
        all_width = all_width[mask_all]
        all_dm = all_dm[mask_all]
        all_status = all_status[mask_all]

    print(f"After cuts: {np.sum(all_status)} detected out of {len(all_status)} injections.")

    return (detected_fluence, detected_scattering_time, detected_width, detected_dm, detected_status,
            nondetected_fluence, nondetected_scattering_time, nondetected_width, nondetected_dm, nondetected_status,
            all_fluence, all_scattering_time, all_width, all_dm, all_status)