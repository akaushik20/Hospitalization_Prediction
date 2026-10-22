import yaml
import pandas as pd
from datetime import datetime, timezone
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, average_precision_score, precision_score, recall_score

from helpers.functions import treat_missing_values, train_test_split_data
import json

def main():
    # Load config
    with open("config_model.yaml") as f:
        config = yaml.safe_load(f)

    # Step 1: Reading training data
    df = pd.read_csv(config["training_data_path"])
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    # Step 1.1 Dropping duplicate rows 
    df.drop_duplicates(inplace=True)


    # Step 2: Train Test split
    leaky_features = config.get("leaky-features", [])
    cat_cols = [c for c in config["cat_cols"] if c not in leaky_features]
    num_cols = [c for c in config["num_cols"] if c not in leaky_features]
    feature_cols = cat_cols + num_cols

    X = df[feature_cols]
    y = df[config["target_col"]]
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Step 3: Preprocessing
    X_train = treat_missing_values(X_train, cat_cols, num_cols)
    X_test = treat_missing_values(X_test, cat_cols, num_cols)

    # Step 5: Feature selection using feature importance
    model = XGBClassifier(enable_categorical=True, tree_method="hist",
                          importance_type="gain", random_state=10)
    model.fit(X_train[feature_cols], y_train)

    gain = model.get_booster().get_score(importance_type="gain")
    gain = dict(sorted(gain.items(), key=lambda item: item[1], reverse=True))
    print("Feature importance (gain):", gain)

    top10 = list(gain.keys())[:10]
    print("Top 10 features:", top10)

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    full = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    sel  = cross_val_score(model, X_train[top10], y_train, cv=cv, scoring="roc_auc")
    print(f"All features: {full.mean():.4f} ± {full.std():.4f}")
    print(f"Top 10:       {sel.mean():.4f} ± {sel.std():.4f}")

    # Step 6: Model training and evaluation
    # Calculate class ratio for handling class imbalance in XGBoost
    num_1 = sum(y_train == 1)
    num_0 = sum(y_train == 0)
    class_ratio = num_0 / num_1

    final_model = XGBClassifier(enable_categorical=True, tree_method="hist",
                                importance_type="gain", random_state=10, scale_pos_weight=class_ratio)        
    final_model.fit(X_train[top10], y_train)
    y_pred = final_model.predict(X_test[top10])
    y_proba = final_model.predict_proba(X_test[top10])[:, 1]
    print("Test F1 Score (class1):", f1_score(y_test, y_pred, pos_label=1))
    print("Test F1 Score (class0):", f1_score(y_test, y_pred, pos_label=0))
    print("Test Precision Score (class1):", precision_score(y_test, y_pred, pos_label=1))
    print("Test Recall Score (class1):", recall_score(y_test, y_pred, pos_label=1))
    print("Test ROC-AUC Score:", roc_auc_score(y_test, y_proba))
    print("Test Average Precision Score:", average_precision_score(y_test, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Step 7: Save the trained model to score it on new population
    final_model.save_model("models/xgb_model.json")

    with open("models/feature_list.json", "w") as f:
        json.dump(top10, f)

    cat_categories = { 
        col: X_train[col].unique().tolist() for col in top10 }

    with open("models/cat_categories.json", "w") as f:
        json.dump(cat_categories, f)

    metadata = {"trained_at": datetime.now().isoformat()}
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f)
    
    print("Model saved to:", "models/xgb_model.json")
    print("Feature list saved to:", "models/feature_list.json")



if __name__ == "__main__":
    main()