# Hospitalization_Prediction

Predicts whether a patient will be hospitalized next year, using an XGBoost model trained on patient features (demographics, chronic conditions, utilization history, labs, etc.).

## What's in this repo

- `train.py` - trains the model on `data/patients.csv`, selects the top 10 features, evaluates it, and saves the model + supporting files to `models/`.
- `score.py` - loads the saved model and scores new patients from `data/patients_to_score.csv`.
- `config_model.yaml` - column/feature configuration used by both scripts.
- `eda.py` - generates an exploratory data analysis report (`output/eda/patients_profile.html`).
- `helpers/functions.py` - shared preprocessing helpers (missing value treatment, train/test split).
- `data/` - input CSVs (`patients.csv` for training, `patients_to_score.csv` for scoring).
- `models/` - saved model artifacts (`xgb_model.json`, `feature_list.json`, `cat_categories.json`, `metadata.json`).
- `output/` - generated outputs, including the final scored predictions.
- `Summary.md` - write-up of the approach, findings, and model evaluation.

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Train the model:
   ```
   python train.py
   ```
   This reads `data/patients.csv`, trains the model, prints evaluation metrics, and saves the model artifacts to `models/`.
3. Score new patients:
   ```
   python score.py
   ```
   This reads `data/patients_to_score.csv`, loads the trained model from `models/`, and writes predicted risk scores to:
   ```
   output/patients_scored.csv
   ```
   Each row contains a `patient_id` and its predicted `risk_score` (probability of hospitalization next year).