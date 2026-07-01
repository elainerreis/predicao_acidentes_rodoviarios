import logging
import joblib
import copy
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from pathlib import Path

# ==========================================
# CONFIGURAÇÕES DE AMBIENTE
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "modelos"
REPORTS_DIR = BASE_DIR / "relatorios" / "figuras"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FILE_SPLITS = DATA_DIR / "data_splitst.pkl"
MODELO_PATH = DATA_DIR / "modelo_xgboost.pkl"


# ==========================================
# FUNÇÕES AUXILIARES DE ALINHAMENTO E SHAP
# ==========================================

def preparar_dados_xgboost(X: pd.DataFrame) -> pd.DataFrame:
    X_cat = X.copy()
    for col in X_cat.select_dtypes(include=['object']).columns:
        X_cat[col] = X_cat[col].astype('category')
    return X_cat

def preparar_categorias_teste(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    X_test_cat = X_test.copy()
    for col in X_train.select_dtypes(include=['category']).columns:
        X_test_cat[col] = X_test_cat[col].astype('category')
        X_test_cat[col] = X_test_cat[col].cat.set_categories(X_train[col].cat.categories)
    return X_test_cat

def agrupar_shap_dummies(shap_values, X_test, prefixo: str, novo_nome: str):
    """
    Soma os valores SHAP de features dummies numa única feature.
    """
    logger.info(f"A agrupar valores SHAP para as colunas com prefixo '{prefixo}'...")
    shap_agrupado = copy.deepcopy(shap_values)
    indices_alvo = [i for i, col in enumerate(X_test.columns) if str(col).startswith(prefixo)]
    
    if len(indices_alvo) <= 1:
        logger.warning(f"Não foram encontradas colunas suficientes com o prefixo '{prefixo}' para agrupar.")
        return shap_values
        
    idx_mestre = indices_alvo[0]
    shap_agrupado.values[:, idx_mestre] = shap_agrupado.values[:, indices_alvo].sum(axis=1)
    
    if shap_agrupado.data is not None:
        shap_agrupado.data[:, idx_mestre] = shap_agrupado.data[:, indices_alvo].sum(axis=1)
        
    shap_agrupado.feature_names[idx_mestre] = novo_nome
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

    # Agrupa colunas dummies do traçado da via
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
    logger.info("--- INÍCIO DA INTERPRETAÇÃO SHAP ---")

    # Verificar existência dos arquivos necessários
    if not FILE_SPLITS.exists() or not MODELO_PATH.exists():
        logger.error("Ficheiros de dados ou modelo final não encontrados!")
        return

    # 1. Carregar dados e modelo
    X_train, X_test, _, _ = joblib.load(FILE_SPLITS)
    melhor_xgb = joblib.load(MODELO_PATH)
    logger.info("Dados e Modelo carregados com sucesso.")

    # 2. Alinhamento Categórico idêntico ao treino
    X_train = preparar_dados_xgboost(X_train)
    X_test = preparar_categorias_teste(X_train, X_test)

    # 💡 DICA CONTRA MEMORY ERROR: Se a memória RAM estourar com o SHAP, 
    # descomente a linha abaixo para calcular o SHAP usando uma amostra representativa.
    # X_test = X_test.sample(n=2000, random_state=42)

    # 3. Executar e salvar gráficos SHAP
    salvar_interpretacao_shap(melhor_xgb, X_test)
    
    logger.info("--- FIM PROCESSAMENTO SHAP ---")

if __name__ == "__main__":
    main()