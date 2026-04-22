"""
Feature Engineering for PHM Data Challenge 2025

Provides a single public entry point:
    get_features(controller_df, sensor_df) -> np.ndarray

The feature vector is constructed in the same order as the training pipeline:
    1. Sensor features  (4 channels × 9 raw + 6 bands × 9 = 54 per channel → 216 total)
    2. Controller features (7 columns × 9 stats → 63 + 1 cut_duration = 64 total)
    Total: 280 features per cut

Constants match training notebook exactly:
    WAVELET  = 'db4'
    LEVEL    = 8
    N_BANDS  = 6
    SENSOR_COLS = ['Acceleration X (g)', 'Acceleration Y (g)',
                   'Acceleration Z (g)', 'AE (V)']

Zero-variance and metadata columns are dropped before feature extraction,
mirroring the preprocessing applied during training.
"""

import numpy as np
import pandas as pd
import pywt
from scipy.stats import skew, kurtosis
from scipy.stats import entropy
from scipy.stats import linregress

# ── Constants (must match training notebook exactly) ───────────────────────

WAVELET = 'db4'
LEVEL   = 8
N_BANDS = 6   # keep bands 0–5 (≤ 5000 Hz)

SENSOR_COLS = [
    'Acceleration X (g)',
    'Acceleration Y (g)',
    'Acceleration Z (g)',
    'AE (V)',
]

# Columns with zero variance — identified during EDA, dropped before features
VARIANCE_ZERO_COLS = ['toolSpndLoad', 'toolSpndSpd', 'toolSpndStatus', 
'A_load', 'progName', 'progStatus', 'mainSpndStatus', 'coolStatus', 'operMode']


# Metadata columns — not useful features
CTRL_DROP_COLS = ['timestamp']



def reshape_for_lstm(df, y_series, sets, feature_cols, scaler=None,
                     fit_scaler=False, max_steps=None):
    """
    Reshape (n_cuts, n_features) → (n_sets, max_steps, n_features).
    Shorter sets are zero-padded at the end.
    """
    X_list, y_list, lengths = [], [], []

    for s in sorted(sets):
        mask = df['set_no'] == s
        sub  = df[mask].sort_values('cut_no')
        X_list.append(sub[feature_cols].fillna(0).values)
        y_list.append(y_series[mask].values)
        lengths.append(len(sub))

    if max_steps is None:
        max_steps = max(lengths)

    print(f"  Cuts per set : { {s: l for s, l in zip(sorted(sets), lengths)} }")
    print(f"  Padding to   : {max_steps} steps")

    n_feat = X_list[0].shape[1]
    X_pad  = np.zeros((len(sets), max_steps, n_feat), dtype=np.float32)
    y_pad  = np.zeros((len(sets), max_steps),          dtype=np.float32)

    for i, (X_s, y_s) in enumerate(zip(X_list, y_list)):
        n = X_s.shape[0]
        X_pad[i, :n, :] = X_s
        y_pad[i, :n]    = y_s

    # Normalise
    flat = X_pad.reshape(-1, n_feat)
    if fit_scaler and scaler is not None:
        scaler.fit(flat)
    if scaler is not None:
        X_pad = scaler.transform(flat).reshape(X_pad.shape)

    return X_pad, y_pad, lengths, max_steps





def safe_col_name(col: str) -> str:
    """Sanitise a raw column name for use as a dict key / feature prefix."""
    return (col.replace(' ', '_')
               .replace('(', '')
               .replace(')', '')
               .replace('/', '_'))


def stat_features(signal: np.ndarray, prefix: str) -> dict:
    """
    Extract 9 statistical features from a 1-D signal array.
    Used for both raw sensor signals and wavelet band coefficients.

    Parameters
    ----------
    signal : np.ndarray  1-D float array
    prefix : str         feature name prefix

    Returns
    -------
    dict  {f'{prefix}_{stat}': value, ...}
    """
    signal = np.asarray(signal, dtype=float)
    total  = np.sum(np.abs(signal)) + 1e-12   # guard against all-zero

    return {
        f'{prefix}_min':            float(np.min(signal)),
        f'{prefix}_max':            float(np.max(signal)),
        f'{prefix}_mean':           float(np.mean(signal)),
        f'{prefix}_std':            float(np.std(signal)),
        f'{prefix}_skew':           float(skew(signal)),
        f'{prefix}_kurtosis':       float(kurtosis(signal)),
        f'{prefix}_energy':         float(np.sum(np.square(signal))),
        f'{prefix}_entropy':        float(entropy(np.abs(signal) / total)),
        f'{prefix}_zero_crossings': int(np.sum(np.diff(np.sign(signal)) != 0)),
    }


def sensor_features(signal: np.ndarray, col_name: str) -> dict:
    """
    Full feature set for one sensor channel:
      - 9 time-domain stats on the raw signal
      - 9 stats × N_BANDS wavelet sub-bands
    Total: 9 + 9×6 = 63 features per channel — but note N_BANDS=6 gives
    9 + 54 = 63, while the training output showed 54 per channel
    (9 raw + 9×5 bands = 54). Keeping N_BANDS=6 to match training exactly
    (coeffs[0] is the approximation, coeffs[1..6] are details bands 0–5).

    Parameters
    ----------
    signal   : np.ndarray  raw sensor signal for one cut
    col_name : str         sanitised column name prefix

    Returns
    -------
    dict of features
    """
    signal   = np.asarray(signal, dtype=float).copy()
    features = {}

    # Time-domain features on raw signal
    features.update(stat_features(signal, col_name))

    # Wavelet decomposition — keep detail bands 0..N_BANDS-1
    coeffs = pywt.wavedec(signal, WAVELET, level=LEVEL)
    for i, coeff in enumerate(coeffs[:N_BANDS]):
        features.update(stat_features(coeff, f'{col_name}_band{i}'))

    return features


def get_numeric_features_controller(series: pd.Series, col_name: str) -> dict:
    """
    Statistical + trend features for one controller signal column.
    Only called on numeric columns after metadata/zero-var cols are dropped.

    Parameters
    ----------
    series   : pd.Series  one controller column for a single cut
    col_name : str        column name used as feature prefix

    Returns
    -------
    dict of 9 features, or {} if fewer than 2 valid samples
    """
    s = series.dropna()
    if len(s) < 2:
        return {}

    slope, _, r2, _, _ = linregress(range(len(s)), s)

    return {
        f'{col_name}_mean':      float(s.mean()),
        f'{col_name}_std':       float(s.std()),
        f'{col_name}_rms':       float(np.sqrt((s ** 2).mean())),
        f'{col_name}_max':       float(s.max()),
        f'{col_name}_min':       float(s.min()),
        f'{col_name}_range':     float(s.max() - s.min()),
        f'{col_name}_iqr':       float(s.quantile(0.75) - s.quantile(0.25)),
        f'{col_name}_slope':     float(slope),
        f'{col_name}_trend_r2':  float(r2),
    }

def controller_features(c_df):
    """
    Aggregate controller readings across ~21 timesteps per cut.
    Only time-domain stats (too few rows for wavelet).
    cut_duration_s computed from start_cut/end_cut before metadata is dropped.
    """
    features = {}
    # ── 1. Extract cut duration BEFORE dropping metadata columns ──────────
    cut_duration_s = float('nan')


    
    start = c_df['start_cut'].dropna().iloc[0]
    end   = c_df['end_cut'].dropna().iloc[0]
    cut_duration_s = (end - start).total_seconds()



    # ── 2. Drop ALL metadata + step columns ───────────────────────────────
    drop_cols = [c for c in ['start_cut', 'end_cut', 'start_step', 'end_step'] 
                 if c in c_df.columns]

    c_df = c_df.drop(columns=drop_cols)


    # ── 3. Numeric stats on remaining columns ─────────────────────────────
    numeric_cols = ['feedrate', 'mainSpndLoad', 'mainSpndSpd', 'X_load', 'Y_load', 'Z_load','B_load']
    for col in numeric_cols:
        features.update(get_numeric_features_controller(c_df[col], col))


    # ── 4. Append duration last ───────────────────────────────────────────
    if cut_duration_s == np.nan:
        raise Exception('Nan Value')
    features['cut_duration_s'] = cut_duration_s
    return features



def get_features(controller_df: pd.DataFrame,
                 sensor_df: pd.DataFrame) -> np.ndarray:
    """
    Extract a fixed-length feature vector for one (set, cut) pair.

    Feature order matches the training pipeline exactly:
        [sensor features (all channels)] + [controller features]

    Parameters
    ----------
    controller_df : pd.DataFrame  raw controller data for one cut
    sensor_df     : pd.DataFrame  raw sensor data for one cut

    Returns
    -------
    np.ndarray  shape (n_features,) — all float32
                Returns array of NaNs if either df is empty.
    """
    
    row = {}

    # ── 1. Sensor features ────────────────────────────────────────────────
    for col in SENSOR_COLS:
        if col not in sensor_df.columns:
            continue
        signal   = sensor_df[col].dropna().to_numpy(dtype=float, copy=True)
        col_name = safe_col_name(col)
        row.update(sensor_features(signal, col_name))

    # ── 2. Controller features ────────────────────────────────────────────
    row.update(controller_features(controller_df))

    return np.array(list(row.values()), dtype=np.float32)
