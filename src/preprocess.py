"""
Data preprocessing utilities for Student Performance Prediction.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "StudentPerformanceFactors.xlsx")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Categorical columns and their expected categories
CATEGORICAL_COLS = [
    "Parental_Involvement",    # Low / Medium / High
    "Access_to_Resources",     # Low / Medium / High
    "Extracurricular_Activities",  # Yes / No
    "Motivation_Level",        # Low / Medium / High
    "Internet_Access",         # Yes / No
    "Family_Income",           # Low / Medium / High
    "Teacher_Quality",         # Low / Medium / High
    "School_Type",             # Public / Private
    "Peer_Influence",          # Positive / Neutral / Negative
    "Learning_Disabilities",   # Yes / No
    "Parental_Education_Level",# High School / College / Postgraduate
    "Distance_from_Home",      # Near / Moderate / Far
    "Gender",                  # Male / Female
]

NUMERICAL_COLS = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]

TARGET_COL = "Exam_Score"


def load_raw_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name="StudentPerformanceFactors")
    df.columns = df.columns.str.strip()
    return df


def encode_and_scale(df: pd.DataFrame, fit: bool = True):
    """
    Encode categorical columns and scale numerical columns.
    Returns (X, y, encoders_dict, scaler).
    If fit=False, loads saved encoders/scaler from models/ dir.
    """
    df = df.copy()

    encoders = {}
    if fit:
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        scaler = StandardScaler()
        df[NUMERICAL_COLS] = scaler.fit_transform(df[NUMERICAL_COLS])
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    else:
        encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.pkl"))
        scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        for col in CATEGORICAL_COLS:
            df[col] = encoders[col].transform(df[col].astype(str))
        df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])

    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    X = df[feature_cols]
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y, encoders, scaler


def preprocess_single(input_dict: dict):
    """Preprocess a single input dict for inference."""
    df = pd.DataFrame([input_dict])
    X, _, _, _ = encode_and_scale(df, fit=False)
    return X
