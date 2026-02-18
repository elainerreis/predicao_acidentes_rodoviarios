import joblib
import pandas as pd
import numpy as np
import shap

artifacts = joblib.load("models/xgb_artifacts.pkl")

model = artifacts["model"]
categories = artifacts["categories"]

cat_features = list(categories.keys())

explainer = shap.TreeExplainer(model)

def explain(input_df: pd.DataFrame):

    for col in cat_features:
        input_df[col] = (
            input_df[col]
            .astype(str)
            .astype("category")
            .cat.set_categories(categories[col])
        )

    shap_values = explainer(input_df)

    return shap_values.values[0].tolist()

