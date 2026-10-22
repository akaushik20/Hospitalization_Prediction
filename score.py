import json
import yaml
import pandas as pd
from xgboost import XGBClassifier
from helpers.functions import treat_missing_values


if __name__ == "__main__":
    with open("config_model.yaml") as f:
        config = yaml.safe_load(f)

    # Step 1 : fetch file
    new_df = pd.read_csv("data/patients_to_score.csv")
    patient_ids = new_df["patient_id"]

    # Step 2: Load model
    model = XGBClassifier()
    model.load_model("models/xgb_model.json")

    with open("models/feature_list.json") as f:
        top10 = json.load(f)
    with open("models/cat_categories.json") as f:
        cat_categories = json.load(f)


    # Step 3: Preprocess 
    new_df = treat_missing_values(new_df, config["cat_cols"], config["num_cols"])
    new_df = new_df[top10]

    # Step 4: Score
    risk_score = model.predict_proba(new_df)[:, 1]

    # Step 5: Save results
    results = pd.DataFrame({
        "patient_id": patient_ids,
        "risk_score": risk_score
    })
    results.to_csv("output/patients_scored.csv", index=False)