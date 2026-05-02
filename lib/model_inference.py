import joblib
import numpy as np



# Load the trained ElasticNet model
enet = joblib.load('model/enet_model.pkl')
print("Model loaded from enet_model.pkl")

# Load the scaler
scaler = joblib.load('model/scaler.pkl')
print("Scaler loaded from scaler.pkl")

# Load the feature indices
top_indices = np.load('model/top_indices.npy')
print("Feature indices loaded from top_indices.npy")


def model_inference(input_features):
    """Generate wear estimate from preprocessed features.
    """
    # apply same feature selection
    X_scaled = scaler.transform(input_features.reshape(1,-1))
    X_scaled = X_scaled[:, top_indices]
    
    # predict
    pred = enet.predict(X_scaled)[0]
    return float(pred)
