import joblib
import pandas as pd


# Load the trained ElasticNet model
enet = joblib.load('model/enet_model.pkl')
print("Model loaded from enet_model.pkl")

xgb = joblib.load('model/xgb_model.pkl')
print("Model loaded from xgb_model.pkl")



# Load the scaler
scaler = joblib.load('model/scaler.pkl')
print("Scaler loaded from scaler.pkl")

# Load the feature indices 
selected_features = joblib.load('model/selected_features.pkl')
print(f"Selected features loaded: {len(selected_features)} features")
selected_features.remove('set_no')
selected_features.remove('Wear_Estimate')

def model_inference(input_features: dict, cut_no):
    """
    Generate wear estimate from a dictionary of feature name → value.
    
    Parameters:
        input_features: dict or pandas Series with feature names as keys
    
    Returns:
        float: predicted wear value

        XGB_Weights : 0.467
        ElasticNet weight: 0.533

    """
    # 1. Convert dict/row to a Series so we can select by name
    
    feature_series = pd.Series(input_features)
    
    # 2. Select only the features the model was trained on (by name!)
    X_selected = feature_series[selected_features].values.reshape(1, -1)
    
    # 3. Scale using the same scaler fitted during training
    X_scaled = scaler.transform(X_selected)
    
    # 4. Predict
    if cut_no <= 11:
        print(f'Used ELastic Net')
        return enet.predict(X_scaled)[0]
    else:
        print(f'Used XGBoost')
        return xgb.predict(X_scaled)[0]
    # enet_pred = enet.predict(X_scaled)[0] 
    # xgb_pred = xgb.predict(X_scaled)[0]

    # pred = 0.467 * xgb_pred + 0.533 * enet_pred
    # return float(pred)
