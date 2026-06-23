import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# CONFIGURAÇÕES INICIAIS

# Configuração do Logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuração de caminhos
BASE_DATA_DIR = Path().resolve().parent 
BASE_DATA_DIR  = BASE_DATA_DIR  / "TCC_ACIDENTES" / "data"

# FUNÇÕES AUXILIARES

def extrair_ano_primeira_linha(arquivo: Path) -> int:
    """Extrai o ano da primeira linha de um arquivo Excel de frota."""
    try:
        df_tmp = pd.read_excel(arquivo, header=None, nrows=1, dtype=str)
        linha = str(df_tmp.iloc[0, 0])
        ano = re.findall(r'(\d{4})', linha)
        
        if ano:
            return int(ano[-1])
        raise ValueError(f"Ano não encontrado na primeira linha.")
    except Exception as e:
        logger.error(f"Erro ao extrair ano do arquivo {arquivo.name}: {e}")
        raise


def encontrar_header(arquivo: Path) -> int:
    """Procura a linha do cabeçalho que contém a coluna 'UF' no Excel."""
    try:
        df_tmp = pd.read_excel(arquivo, header=None, nrows=10)
        for i, row in df_tmp.iterrows():
            valores = row.astype(str).str.upper().tolist()
            if "UF" in valores:
                return i
        raise ValueError("Cabeçalho 'UF' não encontrado nas primeiras 10 linhas.")
    except Exception as e:
        logger.error(f"Erro ao buscar cabeçalho no arquivo {arquivo.name}: {e}")
        raise



# PIPELINE DE DADOS


def carregar_dados_prf(data_dir: Path) -> pd.DataFrame:
    """Lê e consolida os arquivos CSV de acidentes da PRF."""
    logger.info("Iniciando carregamento dos dados da PRF...")
    caminho_acidentes = data_dir / "ac_por_pessoa"
    arquivos = list(caminho_acidentes.glob("*.csv"))
    
    if not arquivos:
        logger.warning(f"Nenhum arquivo encontrado em {caminho_acidentes}")
        return pd.DataFrame()

    logger.info(f"Encontrados {len(arquivos)} arquivos da PRF.")
    lista_acidentes = []
    
    colunas_desejadas = [
        'uf', 'br', 'km', 'municipio', 'dia_semana', 'fase_dia', 'sentido_via',
        'condicao_metereologica', 'tipo_pista', 'tracado_via', 'uso_solo',
        'tipo_veiculo', 'data_inversa', 'estado_fisico'
    ]

    for arquivo in tqdm(arquivos, desc="Lendo arquivos PRF"):
        try:
            df = pd.read_csv(arquivo, encoding="ISO-8859-1", sep=";")
            
            # Filtra apenas colunas existentes para evitar erros caso falte alguma
            colunas_presentes = [c for c in colunas_desejadas if c in df.columns]
            df = df[colunas_presentes]
            
            # Tratamento de datas
            if "data_inversa" in df.columns:
                df["data_inversa"] = pd.to_datetime(df["data_inversa"], errors="coerce")
                df["ano"] = df["data_inversa"].dt.year
            
            lista_acidentes.append(df)
            
        except Exception as e:
            logger.error(f"Erro ao processar {arquivo.name}: {e}")

    df_final = pd.concat(lista_acidentes, ignore_index=True) if lista_acidentes else pd.DataFrame()
    logger.info(f"Dados da PRF carregados. Shape: {df_final.shape}")
    return df_final


def carregar_dados_frota(data_dir: Path) -> pd.DataFrame:
    """Lê e consolida os arquivos Excel com os dados de frota de veículos."""
    logger.info("Iniciando carregamento dos dados de Frota...")
    caminho_frota = data_dir / "frota"
    arquivos = list(caminho_frota.glob("*.xls")) + list(caminho_frota.glob("*.xlsx"))
    
    lista_frotas = []

    for arquivo in tqdm(arquivos, desc="Lendo arquivos de Frota"):
        if "munic" not in arquivo.name.lower():
            continue

        try:
            ano = extrair_ano_primeira_linha(arquivo)
            header_row = encontrar_header(arquivo)

            df = pd.read_excel(arquivo, skiprows=header_row)
            df.columns = df.columns.str.upper().str.strip()

            df = df.rename(columns={
                "UF": "uf",
                "MUNICIPIO": "municipio",
                "MUNICÍPIO": "municipio",
                "TOTAL": "frota"
            })

            # Limpeza e conversões
            df = df[~df["uf"].astype(str).str.upper().str.strip().eq("UF")]
            df["frota"] = pd.to_numeric(df["frota"], errors="coerce")
            df = df.dropna(subset=["frota"])
            df["ano"] = ano

            colunas_finais = [c for c in ["uf", "municipio", "frota", "ano"] if c in df.columns]
            lista_frotas.append(df[colunas_finais])

        except Exception as e:
            logger.error(f"Erro ao processar frota {arquivo.name}: {e}")

    df_final = pd.concat(lista_frotas, ignore_index=True) if lista_frotas else pd.DataFrame()
    logger.info(f"Dados de Frota carregados. Shape: {df_final.shape}")
    return df_final


def mesclar_e_limpar_dados(df_prf: pd.DataFrame, df_frota: pd.DataFrame) -> pd.DataFrame:
    """Realiza o merge e os tratamentos finais na base de acidentes e frota."""
    logger.info("Mesclando bases da PRF e Frota...")
    
    df_merged = df_prf.merge(
        df_frota,
        on=["municipio", "uf", "ano"],
        how="left"
    )

    # 1. Tratamento de Gravidade
    if 'estado_fisico' in df_merged.columns:
        df_merged['gravidade'] = df_merged['estado_fisico'].apply(
            lambda x: 1 if x in ['Lesões Graves', 'Óbito'] else 0
        )
    else:
        logger.warning("Coluna 'estado_fisico' não encontrada. Pulando criação de 'gravidade'.")

    # 2. Tratamento de KM
    if 'km' in df_merged.columns:
        df_merged['km'] = df_merged['km'].astype(str).str.replace(",", ".").astype(float).round(0)

    # 3. Junção das colunas BR e KM e remoção da coluna km
    if 'br' in df_merged.columns and 'km' in df_merged.columns:
        logger.info("Criando a coluna 'br_km' e removendo a coluna 'km'...")
        df_merged['br_km'] = df_merged['br'].astype(str) + df_merged['km'].astype(str)
        # 👇 Aqui está a exclusão da coluna km
        df_merged = df_merged.drop(columns=['km'])

    # 4. Tratamento Tracado Via (Dummies)
    if 'tracado_via' in df_merged.columns:
        logger.info("Transformando a coluna 'tracado_via' em colunas dummies...")
        df_dummies = df_merged['tracado_via'].astype(str).str.get_dummies(sep=';')
        df_merged = pd.concat([df_merged, df_dummies], axis=1)
        # Excluir a coluna original
        df_merged = df_merged.drop(columns=['tracado_via'])

    # 5. Remover outras colunas desnecessárias
    colunas_para_remover = ['estado_fisico', 'data_inversa', 'municipio']
    df_merged = df_merged.drop(columns=[c for c in colunas_para_remover if c in df_merged.columns])

    logger.info(f"Mesclagem e limpeza concluídas. Shape final: {df_merged.shape}")
    return df_merged


def salvar_resultados(df_final: pd.DataFrame, data_dir: Path):
    """Salva a base tratada e uma amostra em arquivos CSV."""
    logger.info("Exportando dados tratados...")
    
    output_dir = data_dir / "tratados"
    output_dir.mkdir(parents=True, exist_ok=True) # Cria a pasta se não existir

    arquivo_completo = output_dir / "acidentes_frota_17-25.csv"
    arquivo_amostra = output_dir / "amostra_17-25.csv"

    # Salva dataset completo
    df_final.to_csv(arquivo_completo, index=False)
    logger.info(f"Dataset completo salvo em: {arquivo_completo}")

    # Salva amostra de 10%
    amostra = df_final.sample(frac=0.10, random_state=42)
    amostra.to_csv(arquivo_amostra, index=False)
    logger.info(f"Amostra de 10% salva em: {arquivo_amostra}")


# EXECUÇÃO PRINCIPAL

def main():
    logger.info("--- INÍCIO DO PROCESSAMENTO ---")
    
    # 1. Carregar Dados
    df_prf = carregar_dados_prf(BASE_DATA_DIR)
    if df_prf.empty:
        logger.error("Falha ao carregar dados da PRF. Encerrando execução.")
        return

    df_frota = carregar_dados_frota(BASE_DATA_DIR)
    if df_frota.empty:
        logger.warning("Aviso: Dados de frota vazios, o merge será feito apenas com dados da PRF.")

    # 2. Transformar
    df_final = mesclar_e_limpar_dados(df_prf, df_frota)
    
    # Exibir algumas estatísticas no terminal para validação
    logger.info(f"Distribuição de Gravidade:\n{df_final.get('gravidade', pd.Series()).value_counts(normalize=True)}")

    # 3. Exportar
    salvar_resultados(df_final, BASE_DATA_DIR)
    
    logger.info("--- PROCESSAMENTO CONCLUÍDO COM SUCESSO ---")

if __name__ == "__main__":
    main()