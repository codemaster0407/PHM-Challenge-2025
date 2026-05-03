# PHM Challenge 2025 Solution

This repository contains the end-to-end machine learning pipeline for the **PHM Challenge 2025**. The solution focuses on robust feature engineering, advanced feature selection, and a cut-wise ensemble modeling strategy to predict tool wear.

## 🚀 Pipeline Overview

The pipeline integrates high-frequency sensor data and controller logs to extract meaningful features, followed by a multi-stage modeling approach.

![Pipeline Explanation](PHM%20Challenge%20A-Star.jpg)

## 🛠️ Feature Engineering

The feature extraction process is detailed in `feature_engineering.ipynb` and follows these four critical steps:

1.  **Data Loading**: Read sensor and controller data using a predefined `DataLoader`.
2.  **Aggregation**: Generate aggregated sensor features (Max, Min, Mean, Skew, Kurtosis) across different frequency ranges.
3.  **Frequency Analysis**: Utilize 8 distinct frequency bands for detailed frequency-wise feature extraction.
4.  **Iterative Extraction**: Iterate through all cuts in both `train_set` and `eval_set` to generate a comprehensive feature matrix for model training.

### Extracted Features
For each signal/band, the following 8 statistical and signal-processing features are computed:
1.  **Maximum Value**
2.  **Minimum Value**
3.  **Skewness**
4.  **Kurtosis**
5.  **Zero Crossings**
6.  **Standard Deviation**
7.  **Energy** (Sum of squares of values)
8.  **Entropy** (|val| / Σ|signal|)

Processed features are stored in the `features/` directory (`train_features.csv` and `evaluation_features.csv`).

## 🔍 Feature Selection

To reduce dimensionality and focus on the most predictive signals, **Mutual Information Regression** was employed. This method captures both linear and non-linear dependencies between features and the target tool wear, ensuring the models are trained on the most informative data points.

## 🤖 Modeling & Ensemble Strategy

Multiple architectures were explored and trained, as documented in `model_2.ipynb`. 

### Cut-wise Ensemble Modeling
A specialized ensemble strategy was implemented based on the tool's lifecycle (cuts):
*   **Cuts < 11**: **Elastic Net** regression is utilized, as it demonstrated superior stability and performance during the early stages of tool wear.
*   **Cuts ≥ 11**: **XGBoost** is deployed for later stages, effectively capturing the complex, non-linear wear patterns that emerge as the tool approaches its end-of-life.

## 📂 Repository Structure

- `main.py`: Main entry point for the inference pipeline.
- `feature_engineering.ipynb`: Contains the logic for feature extraction and preprocessing.
- `model_2.ipynb`: Contains the training code for the various model architectures.
- `features/`: Contains the extracted feature CSV files.
- `lib/`: Core library code for data loading and feature utilities.
- `model/`: Directory for saved model weights and configurations.
- `tcdata/`: Dataset directory.
- `work/`: Output directory for results (e.g., `result.csv`).
- `Dockerfile` & `run.sh`: Environment setup and execution scripts for containerization.

## ⚙️ Requirements

To set up the environment and install dependencies:

```bash
pip install -r requirements.txt
```

## 🏃 Running the Pipeline

To execute the solution:

```bash
./run.sh
```
