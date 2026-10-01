"""
Train multiple regression models, evaluate them, and save the best one.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from preprocess import load_raw_data, encode_and_scale, MODELS_DIR

RANDOM_STATE = 42


def evaluate(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def train():
    print("Loading data...")
    df = load_raw_data()
    print(f"Dataset shape: {df.shape}")

    X, y, encoders, scaler = encode_and_scale(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1
        ),
    }

    results = {}
    trained_models = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        trained_models[name] = model
        print(f"    {metrics}")

    # Pick best by R2
    best_name = max(results, key=lambda k: results[k]["R2"])
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name}  R2={results[best_name]['R2']}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(trained_models, os.path.join(MODELS_DIR, "all_models.pkl"))

    # Save feature names
    feature_names = list(X.columns)
    with open(os.path.join(MODELS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Save metrics and best model name
    summary = {"best_model": best_name, "metrics": results}
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Save test split for later use
    X_test_df = X_test.copy()
    X_test_df["Exam_Score"] = y_test.values
    X_test_df.to_csv(os.path.join(MODELS_DIR, "test_data.csv"), index=False)

    print("All artifacts saved to models/")
    return summary


if __name__ == "__main__":
    train()
