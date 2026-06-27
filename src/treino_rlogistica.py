import logging
from pathlib import Path
from multiprocessing import cpu_count

import joblib
import pandas as pd
from numpy import linspace
from scipy.stats import uniform, loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------
# Configurações Iniciais
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Defina o diretório base (ajuste se for executar de outra pasta)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "modelos"

# ---------------------------------------------------------
# Funções do Pipeline
# ---------------------------------------------------------
def carregar_dados_e_preprocessador(data_dir: Path):
    """
    Carrega o pré-processador e as partições de dados geradas anteriormente.
    """
    logger.info("A iniciar o carregamento dos dados e do pré-processador...")
    
    caminho_prep = data_dir / 'preprocesst.pkl'
    caminho_splits = data_dir / 'data_splitst.pkl'
    
    if not caminho_prep.exists() or not caminho_splits.exists():
        logger.error("Arquivos .pkl não encontrados. Verifique o diretório DATA_DIR.")
        raise FileNotFoundError
        
    preprocess = joblib.load(caminho_prep)
    X_train, X_test, y_train, y_test = joblib.load(caminho_splits)
    
    logger.info(f"Dados carregados com sucesso. Linhas no conjunto de treino: {X_train.shape[0]}")
    return preprocess, X_train, X_test, y_train, y_test


def otimizar_regressao_logistica(X_train: pd.DataFrame, y_train: pd.Series, preprocess) -> RandomizedSearchCV:
    """
    Constrói o pipeline e executa a busca de hiperparâmetros (RandomizedSearchCV)
    para o modelo de Regressão Logística.
    """
    logger.info("A configurar o Pipeline e o RandomizedSearchCV...")
    
    # Construção do Pipeline
    pipeline_lr = Pipeline(steps=[
        ('preprocess', preprocess),
        ('clf', LogisticRegression(max_iter=300))
    ])

    # Espaço de hiperparâmetros (Note o prefixo 'clf__' para o pipeline)
    param_grid_lr = {
        'clf__solver': ['saga'],                
        'clf__penalty': ['elasticnet'],         
        'clf__C': loguniform(1e-2, 1e2),
        'clf__l1_ratio': uniform(0, 1),        
        'clf__class_weight': [None, 'balanced']
    }

    # Configuração da busca
    random_lr = RandomizedSearchCV(
        estimator=pipeline_lr,
        param_distributions=param_grid_lr,  
        n_iter=100,                           
        scoring='f1',
        cv=5,
        n_jobs=cpu_count() // 2,
        random_state=42,
        verbose=1
    )
    
    logger.info("A treinar os modelos (isso pode demorar vários minutos)...")
    random_lr.fit(X_train, y_train)
    
    logger.info(f"Treinamento concluído. Melhor F1 na Validação Cruzada: {random_lr.best_score_:.4f}")
    return random_lr


def avaliar_modelo(modelo: RandomizedSearchCV, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Imprime as métricas básicas e avalia diferentes limiares (thresholds)
    para a classificação final.
    """
    logger.info("A gerar relatórios e métricas de diagnóstico...")
    
    # 1. Métricas Básicas (Threshold padrão de 0.5)
    y_pred = modelo.predict(X_test)
    
    print("\n" + "="*40)
    print("MÉTRICAS BÁSICAS (LIMIAR 0.5)")
    print("="*40)
    print(classification_report(y_test, y_pred))
    print("Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))

    # 2. Mudança de Limiar do Predict
    logger.info("A processar a mudança de limiares de decisão (Thresholds)...")
    y_proba = modelo.predict_proba(X_test)[:, 1]
    thresholds = linspace(0, 0.9, 25)

    print("\n" + "="*40)
    print("ANÁLISE DE DIFERENTES LIMIARES (THRESHOLDS)")
    print("="*40)
    print("Limiar | F1-Score | Recall | Precision")
    print("-" * 39)
    
    for th in thresholds:
        y_pred_th = (y_proba >= th).astype(int)
        
        # Prevenção de divisão por zero caso o modelo não preveja nenhum '1' no limiar atual
        total_positivos_previstos = sum(y_pred_th == 1)
        if total_positivos_previstos == 0:
            prec = 0.0
        else:
            prec = sum((y_pred_th == 1) & (y_test == 1)) / total_positivos_previstos
            
        rec = sum((y_pred_th == 1) & (y_test == 1)) / sum(y_test == 1)
        f1 = f1_score(y_test, y_pred_th)
        
        print(f" {th:.2f}  |  {f1:.4f}  |  {rec:.4f} |  {prec:.4f}")


# ---------------------------------------------------------
# Execução Principal
# ---------------------------------------------------------
def main():
    try:
        # Passo 1: Abertura
        preprocess, X_train, X_test, y_train, y_test = carregar_dados_e_preprocessador(DATA_DIR)
        
        # Passo 2: Otimização e Treino
        modelo_final = otimizar_regressao_logistica(X_train, y_train, preprocess)
        
        # Salvar o melhor modelo final em disco
        joblib.dump(modelo_final.best_estimator_, DATA_DIR / 'melhor_modelo_lrt.pkl')
        
        # Passo 3: Diagnóstico
        avaliar_modelo(modelo_final, X_test, y_test)
        
    except Exception as e:
        logger.exception(f"A execução falhou: {e}")

if __name__ == "__main__":
    main()
