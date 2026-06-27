import logging
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "modelos"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FILE_SPLITS = DATA_DIR / "data_splitst.pkl"
MODELO_PATH = DATA_DIR / "modelo_xgboost_final.pkl"


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
# BLOCO PRINCIPAL DE EXECUÇÃO
# ==========================================

def main():
    logger.info("--- INÍCIO DO TREINO XGBOOST ---")

    if not FILE_SPLITS.exists():
        logger.error(f"Ficheiro {FILE_SPLITS} não encontrado! Corra o pré-processamento primeiro.")
        return

    X_train, X_test, y_train, y_test = joblib.load(FILE_SPLITS)
    logger.info(f"Dados carregados: Treino {X_train.shape}, Teste {X_test.shape}")

    # Preparação Categórica
    X_train = preparar_dados_xgboost(X_train)
    X_test = preparar_categorias_teste(X_train, X_test) 

    # Treino do Modelo
    melhor_xgb = treinar_modelo_otimizado(X_train, y_train)

    # Avaliação
    y_pred = melhor_xgb.predict(X_test)
    print("\n--- RELATÓRIO DE CLASSIFICAÇÃO ---")
    print(classification_report(y_test, y_pred))
    
    print("\n--- MATRIZ DE CONFUSÃO ---")
    print(confusion_matrix(y_test, y_pred))

    # Salvamento do modelo treinado
    joblib.dump(melhor_xgb, MODELO_PATH)
    logger.info(f"Modelo final guardado com sucesso em: {MODELO_PATH}")
    logger.info("--- FIM DO TREINO E AVALIAÇÃO ---")

if __name__ == "__main__":
    main()