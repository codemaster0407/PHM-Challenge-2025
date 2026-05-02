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
    
    evalset_list = [1,2,3]
    cut_list = list(range(2, 27))
    CTRL_DROP_COLS = ['timestamp']


    # Feature extraction 
    records = []
    skipped = []

    output_path = '/work/result.csv'
    controller_path = '/tcdata/Controller_Data'
    sensor_path = '/tcdata/Sensor_Data'

    loader = DataLoader(controller_path, sensor_path)

    print(f'DATA LOADING START .....')
    for set_no in evalset_list:
        for cut_no in cut_list:
            print(f"  evalset_{set_no:02d} | Cut {cut_no:02d} ...", end=' ')

            sensor_df = loader.get_sensor_data(set_no, cut_no)
            ctrl_df   = loader.get_controller_data(set_no, cut_no)
            


            if sensor_df.empty or ctrl_df.empty:
                print("⚠ skipped (no data)")
                skipped.append((set_no, cut_no))
                continue
            ctrl_df.drop(columns = variance_zero_cols, inplace = True) 
            ctrl_df.drop(columns = CTRL_DROP_COLS, inplace = True )
            row = {'set_no': set_no, 'cut_no': cut_no}

            # ── Sensor features ──────────────────────────────────────────
            for col in SENSOR_COLS:
                if col not in sensor_df.columns:
                    continue
                signal = sensor_df[col].dropna().to_numpy(dtype=float, copy=True)  # ← safest
                col_name = safe_col_name(col)
                row.update(sensor_features(signal, col_name))

            # ── Controller features ──────────────────────────────────────
            row.update(controller_features(ctrl_df))

            records.append(row)
            if len(records) == 1:
                print(f"\n── Feature columns ({len(row)} total) ──")
                # for col in row.keys():
                #     print(f"  {col}")
            print(f"✅  ({len(sensor_df):,} sensor rows)")

    features_test = pd.DataFrame(records)
    
    
    nan_cols = ['B_load_trend_r2']
    features_test.drop(columns = nan_cols, inplace = True) 


    output_csv = features_test[['set_no', 'cut_no']].copy()


    output_csv.rename(columns={'set_no': 'set_num', 'cut_no': 'cut_num'}, inplace=True)


    output_csv['set_num'] = output_csv['set_num'].apply(lambda x: f'evalset_{int(x):02d}')




    features_test.drop(columns=['set_no'], inplace=True)
    


    print(features_test.columns)

    predictions = []
    for i, row in features_test.iterrows():
        pred = model_inference(row.to_numpy())
        predictions.append(pred)

    output_csv['pred'] = predictions
    print(f'LOG : Inference Output saved at {output_path}')
    output_csv.to_csv(output_path, index=False)

    
    


    