import streamlit as st
import pandas as pd
import plotly.express as px

# =========================
# ⚙️ Configuração
# =========================
st.set_page_config(
    page_title="TCC - Acidentes",
    layout="wide"
)

# =========================
# 📥 Carregar dados
# =========================
@st.cache_data
def load_data():
    df = pd.read_parquet(r"C:\Users\Elaine\Desktop\TCC_acidentes\data\tratados\acidentes.parquet")

    # 🔥 AJUSTES IMPORTANTES
    df["br"] = df["br"].astype(str)
    df["km"] = df["km"].astype(str)
    df["ano"] = df["ano"].astype(str)

    return df

df = load_data()

# =========================
# 🎯 Título
# =========================
st.title("🚧 Análise de Acidentes em Rodovias Federais")

st.markdown("""
Painel descritivo com foco na identificação de padrões e fatores associados à gravidade dos acidentes.
""")

# =========================
# 🎛️ Filtros
# =========================
st.sidebar.header("🔎 Filtros")

ufs = st.sidebar.multiselect(
    "UF",
    sorted(df["uf"].unique()),
    default=df["uf"].unique()
)

anos = st.sidebar.multiselect(
    "Ano",
    sorted(df["ano"].unique()),
    default=df["ano"].unique()
)

dias = st.sidebar.multiselect(
    "Dia da Semana",
    df["dia_semana"].unique(),
    default=df["dia_semana"].unique()
)

df = df[
    (df["uf"].isin(ufs)) &
    (df["ano"].isin(anos)) &
    (df["dia_semana"].isin(dias))
]

# =========================
# 📊 KPIs
# =========================
st.subheader("📊 Visão Geral")

col1, col2, col3 = st.columns(3)

total = len(df)
graves = df["gravidade"].sum()
taxa = df["gravidade"].mean()

col1.metric("Total de Acidentes", f"{total:,}")
col2.metric("Acidentes Graves", f"{graves:,}")
col3.metric("Taxa de Gravidade", f"{taxa:.2%}")

# =========================
# 📊 Distribuição Gravidade
# =========================
st.subheader("⚰️ Distribuição de Gravidade")

df_grav = df["gravidade"].value_counts().reset_index()
df_grav.columns = ["gravidade", "qtd"]

fig_grav = px.pie(
    df_grav,
    names="gravidade",
    values="qtd"
)

st.plotly_chart(fig_grav, use_container_width=True)

# =========================
# 📅 ANÁLISE TEMPORAL
# =========================
st.subheader("📅 Padrões Temporais")

col1, col2 = st.columns(2)

# Dia da semana
df_dia = df["dia_semana"].value_counts().reset_index()
df_dia.columns = ["dia", "qtd"]

fig_dia = px.bar(df_dia, x="dia", y="qtd", title="Acidentes por Dia")
col1.plotly_chart(fig_dia, use_container_width=True)

# Fase do dia
df_fase = df["fase_dia"].value_counts().reset_index()
df_fase.columns = ["fase", "qtd"]

fig_fase = px.bar(df_fase, x="fase", y="qtd", title="Fase do Dia")
col2.plotly_chart(fig_fase, use_container_width=True)

# =========================
# 🔥 HEATMAP
# =========================
st.subheader("🔥 Heatmap: Dia x Fase do Dia")

heatmap = pd.crosstab(df["dia_semana"], df["fase_dia"])

fig_heat = px.imshow(heatmap)

st.plotly_chart(fig_heat, use_container_width=True)

# =========================
# 📍 GEOGRÁFICO
# =========================
st.subheader("📍 Distribuição Geográfica")

col1, col2 = st.columns(2)

# UF
df_uf = df["uf"].value_counts().reset_index()
df_uf.columns = ["uf", "qtd"]

fig_uf = px.bar(df_uf, x="uf", y="qtd")
col1.plotly_chart(fig_uf, use_container_width=True)

# BR mais perigosas
df_br = df.groupby("br")["gravidade"].mean().reset_index()
df_br = df_br.sort_values(by="gravidade", ascending=False).head(10)

fig_br = px.bar(
    df_br,
    x="gravidade",
    y="br",
    orientation="h",
    title="BRs com Maior Proporção de Acidentes Graves"
)

col2.plotly_chart(fig_br, use_container_width=True)

# =========================
# ⚠️ FATORES DE RISCO
# =========================
st.subheader("⚠️ Fatores de Risco")

col1, col2 = st.columns(2)

# Clima
df_clima = pd.crosstab(
    df["condicao_metereologica"],
    df["gravidade"],
    normalize="index"
)

fig_clima = px.bar(df_clima, barmode="group", title="Gravidade x Clima")
col1.plotly_chart(fig_clima, use_container_width=True)

# Tipo de pista
df_pista = pd.crosstab(
    df["tipo_pista"],
    df["gravidade"],
    normalize="index"
)

fig_pista = px.bar(df_pista, barmode="group", title="Gravidade x Tipo de Pista")
col2.plotly_chart(fig_pista, use_container_width=True)

# =========================
# 🚗 VEÍCULOS
# =========================
st.subheader("🚗 Tipo de Veículo")

df_veic = pd.crosstab(
    df["tipo_veiculo"],
    df["gravidade"],
    normalize="index"
).sort_values(by=1, ascending=False).head(10)

fig_veic = px.bar(
    df_veic,
    barmode="group",
    title="Veículos com Maior Proporção de Gravidade"
)

st.plotly_chart(fig_veic, use_container_width=True)

# =========================
# 🧠 INSIGHTS
# =========================
st.subheader("🧠 Insights")

st.info("""
- A maioria dos acidentes não é grave, mas há padrões claros de risco.
- Determinadas BRs apresentam maior proporção de acidentes graves.
- Fatores como clima, tipo de pista e veículo influenciam diretamente a gravidade.
- Esses padrões são fundamentais para o modelo preditivo do TCC.
""")