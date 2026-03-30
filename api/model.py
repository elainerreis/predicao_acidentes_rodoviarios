import joblib
import pandas as pd
from pathlib import Path


#artifacts = joblib.load("C:/Users/Elaine/Desktop/TCC_acidentes/models/xgb_artifacts.pkl")

# 1. Pega o caminho absoluto da pasta onde o model.py está (pasta 'api')
BASE_DIR = Path(__file__).resolve().parent

# 2. Volta uma pasta (para a raiz do projeto) e entra na pasta 'models'
MODEL_PATH = BASE_DIR.parent / "models" / "xgb_artifacts.pkl"

# 3. Carrega o modelo usando o caminho dinâmico
artifacts = joblib.load(MODEL_PATH)

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
