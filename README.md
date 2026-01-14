# Predição da Gravidade de Acidentes de Trânsito
## Visão Geral

Este projeto tem como objetivo desenvolver e avaliar modelos de *machine learning* para **predição da gravidade de acidentes de trânsito** em rodovias federais brasileiras, utilizando dados da Polícia Rodoviária Federal (PRF).

O problema é formulado como uma **classificação binária**, onde:

* **Classe 0** → Acidente não grave
* **Classe 1** → Acidente grave (feridos graves ou óbitos)

Devido ao **forte desbalanceamento entre as classes**, foram adotadas estratégias específicas de pré-processamento, modelagem e avaliação.

---

## Base de Dados

* **Fontes:** Polícia Rodoviária Federal (PRF) e Ministerio do Transporte
* **Período:** 2019 a 2021
* **Variável alvo:** `gravidade`
* **Variáveis explicativas:**

  * Contexto do acidente (dia da semana, fase do dia, condição meteorológica)
  * Características da via (BR, km, tipo de pista, traçado, uso do solo)
  * Localização (UF, município)
  * Características do veículo
  * Frota veicular associada ao município

---

## Tecnologias Utilizadas
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost

## Estrutura do Projeto
- `data` 
        -`data/ac_por_pessoa`→ dados extraidos da PRF de acidentes por pessoa envolvida
        -`data/frota`→ dados de frotas por município extraidos do Ministerio do Transporte 
        -`data/tratados` → dados de acidentes e frotas tratados 
- `notebooks`
        -`notebooks/modelostest.ipynb` →  Modelagem
        -`notebooks/transform.ipynb` → Tratamento dos dados
- `README.md` → Documento descritivo do projeto




## Pré-processamento dos Dados

O pré-processamento foi realizado utilizando `Pipeline` e `ColumnTransformer`, garantindo reprodutibilidade e evitando vazamento de dados.

As principais etapas incluem:

* Separação entre variáveis **numéricas** e **categóricas**
* Imputação de valores ausentes:

  * Numéricas → mediana
  * Categóricas → valor mais frequente
* Padronização das variáveis numéricas (`StandardScaler`)
* Codificação das variáveis categóricas via `OneHotEncoder`
* Integração de todas as etapas aos pipelines dos modelos

---

## Modelos Utilizados

Foram avaliadas cinco abordagens principais de modelagem:

### 🔹 1. Regressão Logística (Baseline)

Modelo linear utilizado como referência inicial.

* Vantagens: simplicidade e interpretabilidade
* Estratégias adotadas:

  * `class_weight='balanced'`
  * Ajuste de hiperparâmetros via Grid Search

---

### 🔹 2. XGBoost

Modelo baseado em árvores de decisão e *gradient boosting*.

* Capaz de capturar relações não lineares
* Melhor desempenho em dados tabulares
* Estratégias adotadas:

  * Ajuste do parâmetro `scale_pos_weight` para lidar com desbalanceamento
  * Otimização de hiperparâmetros via Grid Search

---


### 🔹 3. StratifiedKFold + Regressão Logística

Aplicação de **validação cruzada estratificada**, garantindo a mesma proporção de classes em todos os folds.

* Uso de `RandomizedSearchCV`
* Avaliação mais robusta e menos dependente de uma única divisão treino-teste

---

### 🔹 4. StratifiedKFold + XGBoost

Validação cruzada estratificada aplicada ao XGBoost.

* Confirma a robustez dos resultados
* Reduz variância na avaliação
* Abordagem mais confiável para uso prático

---

## Métricas de Avaliação

Devido ao desbalanceamento dos dados, as métricas priorizadas foram:

* **Recall (classe grave)** → capacidade de identificar acidentes graves
* **Precision (classe grave)** → confiabilidade das previsões positivas
* **F1-score (classe grave)** → equilíbrio entre recall e precision
* **Accuracy** → utilizada apenas como métrica complementar

O **recall da classe grave** foi considerado a métrica mais importante, pois falsos negativos representam acidentes graves não identificados.

---

## Resultados Obtidos

### Comparação geral dos modelos

| Modelo                           | Recall (classe grave) | F1-score (classe grave) | Accuracy |
| -------------------------------- | --------------------- | ----------------------- | -------- |
| Regressão Logística              | ~0.71                 | ~0.42                   | ~0.70    |
| XGBoost                          | ~0.73                 | ~0.46                   | ~0.73    |
| StratifiedKFold + Regressão Log. | ~0.70                 | ~0.42                   | ~0.69    |
| StratifiedKFold + XGBoost        | ~0.73                 | ~0.46                   | ~0.73    |

---

## Análise dos Resultados

* O **XGBoost** apresentou melhor desempenho geral em relação à Regressão Logística
* A validação cruzada estratificada confirmou a **estabilidade dos modelos**

---

## Modelo Mais Adequado

* **Para segurança viária (priorizar detecção):**
  ➝ **XGBoost com StratifiedKFold**


---

## Conclusão

Os resultados indicam que o XGBoost é mais adequados para a predição da gravidade de acidentes de trânsito em bases desbalanceadas.


---

## Tecnologias Utilizadas

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost
* Matplotlib

---

## Observações

Projeto desenvolvido com fins acadêmicos, voltado à análise e modelagem de dados aplicados à segurança no trânsito.