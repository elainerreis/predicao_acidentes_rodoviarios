import logging
import joblib
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from scipy.stats import uniform, randint
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# CONFIGURAÇÕES DE AMBIENTE
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Caminhos de pastas
BASE_DIR = Path("C:/Users/Elaine/Desktop/TCC_acidentes")
DATA_DIR = BASE_DIR / "data" / "modelos"
REPORTS_DIR = BASE_DIR / "relatorios" / "figuras"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FILE_SPLITS = DATA_DIR / "data_splits.pkl"


# ==========================================
# FUNÇÕES DE PROCESSAMENTO E TREINO
# ==========================================

def preparar_dados_xgboost(X: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que as colunas do tipo 'object' (as que não viraram dummies) 
    sejam convertidas para 'category' para suporte nativo do XGBoost.
    """
    X_cat = X.copy()
    for col in X_cat.select_dtypes(include=['object']).columns:
        X_cat[col] = X_cat[col].astype('category')
    return X_cat

def preparar_categorias_teste(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Garante que as colunas categóricas no teste tenham as mesmas categorias do treino
    para evitar erros de 'mismatch' no XGBoost.
    """
    X_test_cat = X_test.copy()
    for col in X_train.select_dtypes(include=['category']).columns:
        X_test_cat[col] = X_test_cat[col].astype('category')
        X_test_cat[col] = X_test_cat[col].cat.set_categories(X_train[col].cat.categories)
    return X_test_cat

def treinar_modelo_otimizado(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Configura e executa a busca de hiperparâmetros para o XGBoost.
    """
    logger.info("A iniciar a pesquisa de hiperparâmetros (RandomizedSearchCV)...")
    
    xgb = XGBClassifier(
        tree_method='hist',
        enable_categorical=True,
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1
    )

    param_dist = {
        'n_estimators': randint(100, 1000),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'colsample_bytree': uniform(0.5, 0.5),
        'subsample': uniform(0.5, 0.5),
        'scale_pos_weight': uniform(1, 10) 
    }

    search = RandomizedSearchCV(
        xgb,
        param_distributions=param_dist,
        n_iter=100,
        scoring='f1',
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    logger.info(f"Melhores parâmetros encontrados: {search.best_params_}")
    return search.best_estimator_


# ==========================================
# INTERPRETABILIDADE (SHAP)
# ==========================================

def agrupar_shap_dummies(shap_values, X_test, prefixo: str, novo_nome: str):
    """
    Soma os valores SHAP de features dummies numa única feature 
    para manter o gráfico limpo sem perder a informação matemática do modelo.
    """
    logger.info(f"A agrupar valores SHAP para as colunas com prefixo '{prefixo}'...")
    
    shap_agrupado = copy.deepcopy(shap_values)
    
    # Encontra os índices de todas as colunas que têm o prefixo especificado
    indices_alvo = [i for i, col in enumerate(X_test.columns) if str(col).startswith(prefixo)]
    
    if len(indices_alvo) <= 1:
        logger.warning(f"Não foram encontradas colunas suficientes com o prefixo '{prefixo}' para agrupar.")
        return shap_values
        
    idx_mestre = indices_alvo[0]
    
    # Soma os valores SHAP na coluna mestre
    shap_agrupado.values[:, idx_mestre] = shap_agrupado.values[:, indices_alvo].sum(axis=1)
    
    if shap_agrupado.data is not None:
        shap_agrupado.data[:, idx_mestre] = shap_agrupado.data[:, indices_alvo].sum(axis=1)
        
    shap_agrupado.feature_names[idx_mestre] = novo_nome
    
    # Remove as colunas redundantes
    indices_remover = indices_alvo[1:]
    shap_agrupado.values = np.delete(shap_agrupado.values, indices_remover, axis=1)
    
    if shap_agrupado.data is not None:
        shap_agrupado.data = np.delete(shap_agrupado.data, indices_remover, axis=1)
        
    shap_agrupado.feature_names = [
        nome for i, nome in enumerate(shap_agrupado.feature_names) if i not in indices_remover
    ]
    
    return shap_agrupado

def salvar_interpretacao_shap(model, X_test: pd.DataFrame):
    """
    Gera e guarda as visualizações SHAP explicativas.
    """
    logger.info("A calcular valores SHAP (isto pode demorar um pouco)...")
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # ---> APLICAR O AGRUPAMENTO DOS DUMMIES AQUI <---
    # Agrupa todas as colunas que começam com "tracado_via_" numa única barra
    shap_values = agrupar_shap_dummies(
        shap_values, 
        X_test, 
        prefixo="tracado_via_", 
        novo_nome="Traçado da Via (Global)"
    )

    # 1. Gráfico de Importância Global (Barras)
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Importância Global das Variáveis (SHAP)")
    plt.savefig(REPORTS_DIR / "xgboost_shap_bar.png", bbox_inches='tight')
    plt.close()

    # 2. Gráfico de Dispersão (Beeswarm)
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("Impacto das Variáveis na Predição de Gravidade")
    plt.savefig(REPORTS_DIR / "xgboost_shap_beeswarm.png", bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráficos SHAP guardados com sucesso na pasta: {REPORTS_DIR}")


# ==========================================
# BLOCO PRINCIPAL DE EXECUÇÃO
# ==========================================

def main():
    logger.info("--- INÍCIO DO TREINO XGBOOST ---")

    # 1. Carregamento dos dados
    if not FILE_SPLITS.exists():
        logger.error(f"Ficheiro {FILE_SPLITS} não encontrado! Corra o pré-processamento primeiro.")
        return

    X_train, X_test, y_train, y_test = joblib.load(FILE_SPLITS)
    logger.info(f"Dados carregados: Treino {X_train.shape}, Teste {X_test.shape}")

    # 2. Preparação Categórica
    X_train = preparar_dados_xgboost(X_train)
    X_test = preparar_categorias_teste(X_train, X_test) 

    # 3. Treino do Modelo
    melhor_xgb = treinar_modelo_otimizado(X_train, y_train)

    # 4. Avaliação
    y_pred = melhor_xgb.predict(X_test)
    print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---")
    print(classification_report(y_test, y_pred))
    
    print("\n--- MATRIZ DE CONFUSÃO ---")
    print(confusion_matrix(y_test, y_pred))

    # 5. Interpretabilidade SHAP e Exportação
    salvar_interpretacao_shap(melhor_xgb, X_test)
    
    modelo_path = DATA_DIR / "modelo_xgboost_final.pkl"
    joblib.dump(melhor_xgb, modelo_path)
    logger.info(f"Modelo final guardado em: {modelo_path}")

    logger.info("--- FIM DO PROCESSAMENTO ---")

if __name__ == "__main__":
    main()