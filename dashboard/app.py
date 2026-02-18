import streamlit as st
import requests
import pandas as pd

st.title("Risco de Acidentes – Rodovias Federais")

payload = {
    "uf": st.selectbox("UF", ["SP","RJ","MG","GO","PR"]),
    "br": str(st.number_input("BR", min_value=1)),
    "km": st.number_input("KM", min_value=0.0),
    "dia_semana": st.selectbox("Dia", ["segunda-feira","terca-feira","quarta-feira", "sábado"]),
    "fase_dia": st.selectbox("Fase do dia", ["Pleno dia","Plena Noite"]),
    "sentido_via": st.selectbox("Sentido", ["Crescente","Decrescente"]),
    "condicao_metereologica": st.selectbox("Clima", ["Céu Claro","Nublado","Chuva"]),
    "tipo_pista": st.selectbox("Pista", ["Simples","Dupla","Múltipla"]),
    "tracado_via": st.selectbox("Traçado", ["Reta","Curva","Rotatória"]),
    "uso_solo": st.selectbox("Uso do solo", ["Sim","Não"]),
    "tipo_veiculo": st.selectbox("Veículo", ["Automóvel","Caminhão","Ônibus"]),
    "ano": st.number_input("Ano", 2015, 2025),
    "frota": st.number_input("Frota", min_value=0.0)
}

if st.button("Calcular risco"):
    r = requests.post("http://localhost:8000/predict_explain", json=payload)

    res = r.json()

    st.metric("Probabilidade de acidente", f"{res['probabilidade']*100:.2f}%")

    

    df_shap = (
        pd.DataFrame(res["impacto_variaveis"].items(),
                     columns=["Variável","Impacto"])
        .sort_values("Impacto", key=abs, ascending=False)
    )

    st.bar_chart(df_shap.set_index("Variável"))
