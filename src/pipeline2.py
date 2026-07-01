import logging
import joblib
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ==========================================
# CONFIGURAÇÕES INICIAIS
# ==========================================

# Configuração do Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configurações de caminhos corrigidas
BASE_DIR = Path().resolve().parent 
DATA_DIR = BASE_DIR / "TCC_ACIDENTES" 
INPUT_FILE = DATA_DIR / "data" / "tratados" / "acidente_frota_17_25.csv"

# 👉 CORREÇÃO AQUI: Agora aponta corretamente para TCC_ACIDENTES/data/modelos
MODELS_DIR = DATA_DIR / "data" / "modelos"


# ==========================================
# FUNÇÕES DE PROCESSAMENTO
# ==========================================

def carregar_dados(caminho_ficheiro: Path) -> pd.DataFrame:
    """Lê o ficheiro CSV contendo os dados já tratados."""
    logger.info(f"A carregar dados do ficheiro: {caminho_ficheiro}")
    try:
        df = pd.read_csv(caminho_ficheiro)
        
        # Transformando colunas 'br' e 'km' em strings
        logger.info("Transformando colunas 'br' e 'km' em strings...")
        if 'br' in df.columns:
            df['br'] = df['br'].astype(str)
        if 'ano' in df.columns:
            df['ano'] = df['ano'].astype(str)
    
            
        logger.info(f"Dados carregados com sucesso. Dimensões: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Ficheiro não encontrado em: {caminho_ficheiro}")
        raise
    except Exception as e:
        logger.error(f"Erro inesperado ao carregar dados: {e}")
        raise


def dividir_dados(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa os dados em variáveis preditivas (X) e alvo (y), e em conjuntos de treino e teste."""
    logger.info("A dividir os dados em conjuntos de treino e teste (Holdout)...")
    
    if 'gravidade' not in df.columns:
        raise ValueError("A coluna alvo 'gravidade' não foi encontrada no DataFrame.")

    X = df.drop('gravidade', axis=1)
    y = df['gravidade']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    logger.info(f"Divisão concluída. Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras.")
    return X_train, X_test, y_train, y_test


def criar_pipeline_processamento() -> ColumnTransformer:
    """Cria e devolve o pipeline de transformação do Scikit-Learn."""
    logger.info("A configurar o pipeline de pré-processamento...")
    
    numeric_features = ['frota']

    binary_features = [
        'Reta', 'Curva', 'Declive', 'Aclive',
        'Interseção de Vias', 'Em Obras',
        'Retorno Regulamentado', 'Rotatória',
        'Ponte', 'Viaduto', 'Desvio Temporário',
        'Túnel'
    ]
    
    categorical_features = [
        'br', 'ano', 'uf', 'dia_semana', 'fase_dia', 
        'sentido_via', 'condicao_metereologica', 'tipo_pista',
        'uso_solo', 'tipo_veiculo', 'br_km'
    ]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=0.001))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('bin', 'passthrough', binary_features),
            ('cat', categorical_transformer, categorical_features)
        ], 
        n_jobs=2
    )
    
    return preprocess


def guardar_artefactos(
    preprocessador: ColumnTransformer, 
    splits: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], 
    diretorio_saida: Path
):
    """Guarda o pipeline treinado e os dados particionados (splits) em ficheiros .pkl."""
    logger.info(f"A guardar os artefactos no diretório: {diretorio_saida}")
    
    # Cria o diretório de modelos caso não exista
    diretorio_saida.mkdir(parents=True, exist_ok=True)
    
    caminho_pipeline = diretorio_saida / 'preprocesst.pkl'
    caminho_splits = diretorio_saida / 'data_splitst.pkl'

    # Guardar pipeline
    joblib.dump(preprocessador, caminho_pipeline)
    logger.info(f"Pipeline guardado em: {caminho_pipeline}")

    # Guardar splits
    joblib.dump(splits, caminho_splits)
    logger.info(f"Divisões de dados guardadas em: {caminho_splits}")


# ==========================================
# EXECUÇÃO PRINCIPAL
# ==========================================

def main():
    logger.info("--- INÍCIO DO PRÉ-PROCESSAMENTO ---")
    
    # 1. Carregamento dos dados
    df = carregar_dados(INPUT_FILE)
    
    # 2. Divisão (Train-Test Split)
    X_train, X_test, y_train, y_test = dividir_dados(df)
    splits = (X_train, X_test, y_train, y_test)
    
    # 3. Construção e treino do Pipeline
    preprocessador = criar_pipeline_processamento()
    
    # Filtrando as colunas no pipeline baseadas no que realmente sobrou em X_train
    for i, transformer in enumerate(preprocessador.transformers):
        nome_transf, obj_transf, cols_transf = transformer
        if obj_transf != 'drop':
             cols_validas = [c for c in cols_transf if c in X_train.columns]
             preprocessador.transformers[i] = (nome_transf, obj_transf, cols_validas)
    
    logger.info("A treinar (fit_transform) o pipeline com os dados de treino...")
    X_train_transformed = preprocessador.fit_transform(X_train)
    logger.info(f"Transformação concluída. Dimensões após pré-processamento: {X_train_transformed.shape}")
    
    # 4. Exportação dos ficheiros
    guardar_artefactos(preprocessador, splits, MODELS_DIR)
    
    logger.info("--- PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO ---")

if __name__ == "__main__":
    main()