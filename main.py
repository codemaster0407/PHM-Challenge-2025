from lib.data_loader import DataLoader
from lib.feature_engineering import *
import pandas as pd
from lib.model_inference import model_inference


if __name__ == "__main__":

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

    variance_zero_cols = ['toolSpndLoad', 'toolSpndSpd', 'toolSpndStatus', 'A_load', 'progName', 'progStatus', 'mainSpndStatus', 'coolStatus', 'operMode']
    nan_cols = ['B_load_trend_r2']

    evalset_list = [1, 2, 3]
    cut_list = list(range(2, 27))
    CTRL_DROP_COLS = ['timestamp']

    output_path = 'work/result.csv'
    controller_path = 'tcdata/Controller_Data'
    sensor_path = 'tcdata/Sensor_Data'

    loader = DataLoader(controller_path, sensor_path)

    results = []
    skipped = []

    print('DATA LOADING START .....')
    for set_no in evalset_list:
        for cut_no in cut_list:
            sensor_df = loader.get_sensor_data(set_no, cut_no)
            ctrl_df   = loader.get_controller_data(set_no, cut_no)

            if sensor_df.empty or ctrl_df.empty:
                print("⚠ skipped (no data)")
                skipped.append((set_no, cut_no))
                continue

            ctrl_df.drop(columns=variance_zero_cols, inplace=True)
            ctrl_df.drop(columns=CTRL_DROP_COLS, inplace=True)

            row = {'cut_no': cut_no}

            # ── Sensor features ──────────────────────────────────────────
            for col in SENSOR_COLS:
                if col not in sensor_df.columns:
                    continue
                signal = sensor_df[col].dropna().to_numpy(dtype=float, copy=True)
                col_name = safe_col_name(col)
                row.update(sensor_features(signal, col_name))

            # ── Controller features ──────────────────────────────────────
            row.update(controller_features(ctrl_df))

            # ── Inference ────────────────────────────────────────────────
            feature_array = pd.Series(row)
            pred = model_inference(feature_array, cut_no)
            
            # print(f'LOG : Prediction for evalset_{int(set_no):02d} cut_no {cut_no} is {pred}')

            results.append({
                'set_num': f'evalset_{int(set_no):02d}',
                'cut_num': cut_no,
                'pred': pred
            })


    output_csv = pd.DataFrame(results)
    output_csv.to_csv(output_path, index=False)
    print(f'LOG : Inference output saved at {output_path}')

    if skipped:
        print(f"Skipped cuts: {skipped} (There may be error in data generation)")
