from fastapi import FastAPI
from api.schemas import AccidentInput
from api.model import preprocess_input, model
from api.explain import explain

app = FastAPI(title="Risk Road API")

@app.post("/predict")
def predict(data: AccidentInput):

    df = preprocess_input(data.dict())

    prob = model.predict_proba(df)[0, 1]

    return {
        "probabilidade": round(float(prob), 4)
    }

@app.post("/predict_explain")
def predict_explain(data: AccidentInput):

    df = preprocess_input(data.dict())

    prob = model.predict_proba(df)[0, 1]

    shap_vals = explain(df)  # já é lista simples

    feature_names = model.get_booster().feature_names

    impacto = dict(zip(feature_names, shap_vals))

    return {
        "probabilidade": round(float(prob), 4),
        "impacto_variaveis": impacto
    }

