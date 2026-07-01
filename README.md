# 🚦 Predição de Acidentes Rodoviários com Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-API-success)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)

Sistema de Apoio à Decisão (Decision Support System - DSS) desenvolvido para prever a gravidade de acidentes em rodovias federais brasileiras utilizando técnicas de Machine Learning e Inteligência Artificial Explicável (XAI).

O projeto foi desenvolvido como Trabalho de Conclusão de Curso (TCC) do curso de Ciência de Dados para Negócios da Universidade Federal da Paraíba (UFPB).

---

# Demonstração

> 📌 **Adicionar GIF da aplicação em funcionamento**

```markdown
![](docs/images/dashboard.gif)
```

---

# Dashboard

> 📌 **Adicionar imagem da tela principal**

```markdown
![](docs/images/dashboard.png)
```

---

# O Problema

Os acidentes em rodovias federais brasileiras representam um dos principais desafios da segurança viária nacional.

Grande parte das análises atualmente realizadas é descritiva, permitindo compreender apenas o que já aconteceu. Entretanto, gestores públicos necessitam de ferramentas capazes de identificar, de forma antecipada, os fatores associados à ocorrência de acidentes graves para apoiar ações preventivas.

Este projeto propõe um Sistema de Apoio à Decisão capaz de estimar a probabilidade de gravidade de um acidente a partir das características da ocorrência, fornecendo também explicações interpretáveis para cada previsão.

---

# Objetivos

- Desenvolver um modelo de classificação para prever acidentes graves;
- Comparar diferentes algoritmos de Machine Learning;
- Construir um pipeline completo de preparação dos dados;
- Aplicar técnicas de Explainable AI utilizando SHAP;
- Disponibilizar o modelo através de uma API;
- Desenvolver uma interface web para apoio à tomada de decisão.

---

# Arquitetura da Solução

> 📌 **Adicionar diagrama da arquitetura**

```markdown
![](docs/images/arquitetura.png)
```

Fluxo geral:

```
Dados PRF
      │
      ▼
ETL
      │
      ▼
Integração das Bases
      │
      ▼
Engenharia de Atributos
      │
      ▼
Treinamento
      │
      ▼
Modelo XGBoost
      │
      ▼
SHAP
      │
      ▼
FastAPI
      │
      ▼
Dashboard Streamlit
```

---

# Estrutura do Projeto

```text
predicao_acidentes_rodoviarios/

├── api/
├── app/
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── docs/
│   └── images/
│
├── models/
├── notebooks/
├── src/
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- SHAP
- FastAPI
- Streamlit
- Joblib
- Matplotlib
- Plotly

---

# Base de Dados

O projeto utiliza dados públicos provenientes de diferentes fontes governamentais.

### Fontes

- Polícia Rodoviária Federal (PRF)
- Ministério dos Transportes

### Informações utilizadas

- UF
- Município
- Rodovia
- Quilômetro
- Tipo de veículo
- Tipo de pista
- Traçado da via
- Condições meteorológicas
- Dia da semana
- Fase do dia
- Frota municipal
- Gravidade do acidente

---

# Pipeline de Machine Learning

> 📌 **Adicionar fluxograma do pipeline**

```markdown
![](docs/images/pipeline.png)
```

Etapas:

1. Coleta dos dados
2. ETL
3. Integração das bases
4. Limpeza
5. Engenharia de atributos
6. Divisão treino/teste
7. Treinamento
8. Otimização de hiperparâmetros
9. Avaliação
10. Interpretabilidade (SHAP)
11. Disponibilização via API
12. Dashboard

---

# Engenharia de Atributos

O pipeline realiza automaticamente:

- tratamento de valores ausentes;
- criação de variáveis derivadas;
- codificação de atributos categóricos;
- normalização de variáveis numéricas;
- integração entre diferentes bases de dados;
- tratamento do desbalanceamento das classes.

---

# Modelos Avaliados

| Modelo | Finalidade |
|---------|------------|
| Regressão Logística | Baseline |
| XGBoost | Modelo principal |

---

# Otimização dos Modelos

Foram utilizadas:

- Randomized Search
- Validação Cruzada (5-Fold Cross Validation)

Métrica principal:

- **F1-Score**

---

# Resultados

## Comparação dos Modelos

> 📌 **Adicionar tabela de métricas**

| Modelo | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| Logistic Regression | | | | |
| XGBoost | | | | |

---

## Matriz de Confusão

> 📌 **Adicionar imagem**

```markdown
![](docs/images/confusion_matrix.png)
```

---

## Curva ROC

> 📌 **Adicionar imagem**

```markdown
![](docs/images/roc_curve.png)
```

---

## Importância das Variáveis

> 📌 **Adicionar gráfico**

```markdown
![](docs/images/feature_importance.png)
```

---

# Explainable AI (SHAP)

O projeto utiliza SHAP para explicar individualmente cada previsão realizada pelo modelo.

## Summary Plot

> 📌 **Adicionar imagem**

```markdown
![](docs/images/shap_summary.png)
```

---

## Waterfall Plot

> 📌 **Adicionar imagem**

```markdown
![](docs/images/shap_waterfall.png)
```

---

## Force Plot

> 📌 **Adicionar imagem**

```markdown
![](docs/images/shap_force.png)
```

---

# Análise Exploratória

## Gravidade por Estado

> 📌 **Adicionar gráfico**

```markdown
![](docs/images/acidentes_estado.png)
```

---

## Gravidade por Tipo de Veículo

> 📌 **Adicionar gráfico**

```markdown
![](docs/images/tipo_veiculo.png)
```

---

## Condições Meteorológicas

> 📌 **Adicionar gráfico**

```markdown
![](docs/images/clima.png)
```

---

## Tipo de Pista

> 📌 **Adicionar gráfico**

```markdown
![](docs/images/tipo_pista.png)
```

---

# API

Exemplo de requisição

```http
POST /predict
```

```json
{
    "uf": "PB",
    "tipo_pista": "Simples",
    "fase_dia": "Plena Noite",
    "tipo_veiculo": "Automóvel",
    "frota": 35000
}
```

Resposta

```json
{
    "probabilidade": 0.84,
    "classe": "Alto Risco"
}
```

---

# Dashboard

Fluxo do usuário

```
Usuário

↓

Seleciona as características do acidente

↓

Modelo realiza a previsão

↓

SHAP explica os fatores

↓

Resultado exibido no dashboard
```

---

# Como Executar

## Clone o repositório

```bash
git clone https://github.com/elainerreis/predicao_acidentes_rodoviarios.git
```

## Acesse a pasta

```bash
cd predicao_acidentes_rodoviarios
```

## Crie um ambiente virtual

```bash
python -m venv .venv
```

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

## Instale as dependências

```bash
pip install -r requirements.txt
```

---

# Executando o Projeto

ETL

```bash
python src/etl.py
```

Treinamento - Regressão Logística

```bash
python src/train_logistic.py
```

Treinamento - XGBoost

```bash
python src/train_xgboost.py
```

API

```bash
uvicorn api.main:app --reload
```

Dashboard

```bash
streamlit run app/app.py
```

---

# Roadmap

- [x] ETL
- [x] Integração das bases
- [x] Engenharia de atributos
- [x] Regressão Logística
- [x] XGBoost
- [x] SHAP
- [x] API
- [x] Dashboard
- [ ] Deploy em nuvem
- [ ] Docker
- [ ] Monitoramento do modelo
- [ ] Atualização automática dos dados

---

# Trabalhos Futuros

- Inclusão de novos modelos de Machine Learning;
- Predição em tempo real;
- Atualização automática da base da PRF;
- Integração com mapas interativos;
- Geração automática de relatórios;
- Sistema de autenticação de usuários;
- Dashboard gerencial para órgãos públicos.

---

# Trabalho Acadêmico

**Título**

Dos Dados ao Alerta: Utilizando Machine Learning para Identificar e Prevenir Riscos em Trechos Críticos das Rodovias Nordestinas

**Autor**

Elaine Reis

**Instituição**

Universidade Federal da Paraíba (UFPB)

Curso de Ciência de Dados para Negócios

---

# Citação

```bibtex
@misc{reis2026,
  author = {Elaine Reis},
  title = {Dos Dados ao Alerta: Utilizando Machine Learning para Identificar e Prevenir Riscos em Trechos Críticos das Rodovias Nordestinas},
  year = {2026},
  note = {Trabalho de Conclusão de Curso},
  institution = {Universidade Federal da Paraíba}
}
```

---

# Licença

Este projeto foi desenvolvido para fins acadêmicos e de pesquisa.
 