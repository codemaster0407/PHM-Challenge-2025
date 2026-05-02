import numpy as np
import pywt
from scipy.stats import skew, kurtosis, entropy
from scipy.stats import linregress


WAVELET  = 'db4'
LEVEL    = 8
N_BANDS  = 6   # keep bands 0–6 (≤ 5000 Hz)

SENSOR_COLS = [
    'Acceleration X (g)',
    'Acceleration Y (g)',
    'Acceleration Z (g)',
    'AE (V)'
]

CONTROLLER_COLS = [
    'feedrate', 'mainSpndLoad', 'mainSpndSpd',
    'X_load', 'Y_load', 'Z_load'
]


def safe_col_name(col):
    """Sanitise column name for use as a dict key."""
    return col.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')

def stat_features(signal, prefix):
    """
    Only for Sensor Dataframe
    Extract 9 statistical features from a 1D array.
    Works for both raw signal and wavelet coefficients.
    """
    signal = np.array(signal, dtype=float)
    total  = np.sum(np.abs(signal)) + 1e-12  # avoid div by zero in entropy

    return {
        f'{prefix}_min':            np.min(signal),
        f'{prefix}_max':            np.max(signal),
        f'{prefix}_mean':           np.mean(signal),
        f'{prefix}_std':            np.std(signal),
        f'{prefix}_skew':           skew(signal),
        f'{prefix}_kurtosis':       kurtosis(signal),
        f'{prefix}_energy':         np.sum(np.square(signal)),
        f'{prefix}_entropy':        entropy(np.abs(signal) / total),
        f'{prefix}_zero_crossings': int(np.sum(np.diff(np.sign(signal)) != 0)),
    }



def sensor_features(signal, col_name):
    """
    Full feature set for one sensor channel:
      - 9 time-domain stats on the raw signal
    Total: 54 features per channel
    """
    signal = np.asarray(signal, dtype = float).copy()
    features = {}

    # Time-domain features on raw signal
    features.update(stat_features(signal, col_name))

    # Wavelet decomposition → keep bands 0–6
    coeffs = pywt.wavedec(signal, WAVELET, level=LEVEL)
    for i, coeff in enumerate(coeffs[:N_BANDS]):
        features.update(stat_features(coeff, f'{col_name}_band{i}'))

    return features


def get_numeric_features_controller(series, col_name):
    """
    Only for Controller DF 
    Statistical + trend features for a continuous signal for the controller.
    """
    s = series.dropna()
    if len(s) < 2:
        return {}
    
    slope, _, r2, _, _ = linregress(range(len(s)), s)
    
    return {
        f'{col_name}_mean':    s.mean(),
        f'{col_name}_std':     s.std(),
        f'{col_name}_rms':     np.sqrt((s**2).mean()),
        f'{col_name}_max':     s.max(),
        f'{col_name}_min':     s.min(),
        f'{col_name}_range':   s.max() - s.min(),
        f'{col_name}_iqr':     s.quantile(0.75) - s.quantile(0.25),
        f'{col_name}_slope':   slope,       # intra-cut trend
        f'{col_name}_trend_r2': r2,
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
