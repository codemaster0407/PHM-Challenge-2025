import os
import pickle
import pandas as pd
import numpy as np
from tensorflow import keras
from lib.data_loader import DataLoader
from lib.feature_engineering import (
    VARIANCE_ZERO_COLS, CTRL_DROP_COLS, SENSOR_COLS,
    safe_col_name, sensor_features, controller_features,
)
print(os.getcwd())


def prepare_set_input(cut_features_list, scaler, max_steps):

    X = np.array(cut_features_list, dtype=np.float32)   # (n_cuts, n_features)

    # ── Sanitise: replace NaN/inf before scaling ──────────────────────────
    nan_count = np.isnan(X).sum()

    if nan_count > 0:
        print(f"{nan_count} NaN values in features")
        raise Exception('Nan values found')

    n_cuts, n_feat = X.shape
    X_pad = np.zeros((1, max_steps, n_feat), dtype=np.float32)

    if n_cuts < max_steps:
        # Forward-fill first cut into missing leading positions
        pad_len = max_steps - n_cuts
        X_pad[0, :pad_len, :] = X[0]    # repeat first available cut at start
        X_pad[0, pad_len:, :] = X
    else:
        X_pad[0, :n_cuts, :] = X

    flat  = X_pad.reshape(-1, n_feat)
    X_pad = scaler.transform(flat).reshape(X_pad.shape)

    return X_pad, n_cuts



def main():
    """
    Run evaluation on all controller and sensor datasets,
    extract features, apply trained LSTM model, and save predictions.

    Output
    ------
    result.csv saved to /work/result.csv
    """

    controller_path = 'tcdata/Controller_Data'
    sensor_path     = 'tcdata/Sensor_Data'
    output_path     = 'work/result.csv'
    model_path      = 'model/lstm_wear_model_2.keras'
    scaler_path     = 'model/scaler_2.pkl'
    meta_path       = 'model/meta_2.pkl'

    evalset_list = [1, 2, 3]
    cut_list     = list(range(2, 27))   # cuts 2–26

    # ── Load model, scaler, metadata ──────────────────────────────────────
    print("Loading model...")
    model  = keras.models.load_model(model_path)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)

    max_steps = meta['max_steps']
    print(f"Model ready  │  max_steps={max_steps}")

    # ── Feature extraction ────────────────────────────────────────────────
    loader  = DataLoader(controller_path, sensor_path)
    records = []
    skipped = []

    for set_no in evalset_list:
        for cut_no in cut_list:
            print(f"  evalset_{set_no:02d} | Cut {cut_no:02d} ...", end=' ')

            sensor_df = loader.get_sensor_data(set_no, cut_no)
            ctrl_df   = loader.get_controller_data(set_no, cut_no)

            if sensor_df.empty or ctrl_df.empty:
                print("⚠ skipped (no data)")
                skipped.append((set_no, cut_no))
                continue

            # Drop zero-variance and metadata columns before feature extraction
            ctrl_df.drop(columns=[c for c in VARIANCE_ZERO_COLS if c in ctrl_df.columns], inplace=True)
            ctrl_df.drop(columns=[c for c in CTRL_DROP_COLS     if c in ctrl_df.columns], inplace=True)

            row = {'set_no': set_no, 'cut_no': cut_no}

            # ── Sensor features ──────────────────────────────────────────
            for col in SENSOR_COLS:
                if col not in sensor_df.columns:
                    continue
                signal   = sensor_df[col].dropna().to_numpy(dtype=float, copy=True)
                col_name = safe_col_name(col)
                row.update(sensor_features(signal, col_name))

            # ── Controller features ──────────────────────────────────────
            row.update(controller_features(ctrl_df))

            records.append(row)
            if len(records) == 1:
                print(f"\n── Feature columns ({len(row)} total) ──")
            print(f"✅  ({len(sensor_df):,} sensor rows)")

    feature_df_eval = pd.DataFrame(records)
    print(f"\nFeature matrix shape (all cuts): {feature_df_eval.shape}")

    if skipped:
        print(f"Skipped {len(skipped)} cut(s): {skipped}")

    # Drop column that was absent during training
    feature_df_eval.drop(columns=['B_load_trend_r2'], inplace=True, errors='ignore')
    print(f"Feature matrix shape (after drop): {feature_df_eval.shape}")

    # ── Inference per set ─────────────────────────────────────────────────
    feature_cols = [c for c in feature_df_eval.columns if c not in ('set_no', 'cut_no')]
    result_records = []

    for set_no in evalset_list:
        print(f"\n── Set {set_no} ──────────────────────────")
        subset = feature_df_eval[feature_df_eval['set_no'] == set_no].sort_values('cut_no')

        if subset.empty:
            print(f"  No valid cuts found for set {set_no}, skipping.")
            continue

        valid_cuts        = subset['cut_no'].tolist()
        cut_features_list = subset[feature_cols].values.tolist()

        X_input, n_cuts = prepare_set_input(cut_features_list, scaler, max_steps)
        y_pred_raw      = model.predict(X_input, verbose=0)        # (1, max_steps)
        y_pred_cuts     = y_pred_raw[0, :n_cuts]                   # (n_cuts,)

        for cut_no, pred in zip(valid_cuts, y_pred_cuts):
            result_records.append({
                'set_num': f'evalset_{set_no:02d}',
                'cut_num': cut_no,
                'pred':    float(pred),
            })
            print(f"  Cut {cut_no:>2d}  →  {pred:.2f}")

    # ── Save results ──────────────────────────────────────────────────────
    result_df = pd.DataFrame(result_records, columns=['set_num', 'cut_num', 'pred'])
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}  ({len(result_df)} rows)")


if __name__ == '__main__':
    main()
