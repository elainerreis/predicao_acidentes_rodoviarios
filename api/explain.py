import joblib
import pandas as pd
import numpy as np
import shap
from pathlib import Path


#artifacts = joblib.load("C:/Users/Elaine/Desktop/TCC_acidentes/models/xgb_artifacts.pkl")

# 1. Pega o caminho absoluto da pasta onde o model.py está (pasta 'api')
BASE_DIR = Path(__file__).resolve().parent

# 2. Volta uma pasta (para a raiz do projeto) e entra na pasta 'models'
MODEL_PATH = BASE_DIR.parent / "models" / "xgb_artifacts.pkl"

# 3. Carrega o modelo usando o caminho dinâmico
artifacts = joblib.load(MODEL_PATH)

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

