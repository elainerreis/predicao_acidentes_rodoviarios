import joblib
import pandas as pd


artifacts = joblib.load(
    "C:/Users/Elaine/Desktop/TCC_acidentes/models/xgb_artifacts.pkl"
)

model = artifacts["model"]
trained_categories = artifacts["categories"]
feature_order = artifacts["feature_order"]

def preprocess_input(data: dict):

    df = pd.DataFrame([data])

    # Ajustar categóricas
    for col in trained_categories:
        df[col] = df[col].astype(str).astype("category")
        df[col] = df[col].cat.set_categories(trained_categories[col])

    # Garantir ordem correta
    df = df[feature_order]

    return df
