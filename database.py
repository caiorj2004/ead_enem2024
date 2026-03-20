"""
database.py – Conexão ao banco de dados e coleta dos dados do ENEM 2024.

Estratégia:
  1. Abre (ou cria) um banco SQLite local (`enem2024.db`).
  2. Se a tabela `enem2024` ainda não existir, verifica se há um CSV com os
     microdados reais em `data/MICRODADOS_ENEM_2024.csv`.  Caso contrário,
     gera um conjunto de dados sintéticos com a mesma estrutura para fins
     demonstrativos.
  3. Expõe a função pública `load_data()` que os demais módulos utilizam.
"""

import os
import sqlite3

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), "enem2024.db")
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "MICRODADOS_ENEM_2024.csv")

# Número de registros sintéticos gerados quando não há CSV disponível
SYNTHETIC_ROWS = 15_000

# Semente para reprodutibilidade
RNG_SEED = 42

# ---------------------------------------------------------------------------
# Dados auxiliares
# ---------------------------------------------------------------------------

UFS = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
    "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
    "RO", "RR", "RS", "SC", "SE", "SP", "TO",
]

# Pesos aproximados de população por UF (ordem igual a UFS acima)
UF_WEIGHTS = [
    0.4, 1.6, 2.1, 0.4, 7.3, 4.4, 1.5, 2.0, 3.4, 3.4,
    10.6, 1.3, 1.6, 4.1, 2.0, 4.7, 1.6, 5.6, 8.7, 1.8,
    0.8, 0.3, 5.9, 3.6, 1.1, 22.0, 0.8,
]


# ---------------------------------------------------------------------------
# Geração de dados sintéticos
# ---------------------------------------------------------------------------

def _create_synthetic_data(n: int = SYNTHETIC_ROWS) -> pd.DataFrame:
    """Gera um DataFrame sintético com a estrutura dos microdados do ENEM."""
    rng = np.random.default_rng(RNG_SEED)

    uf_weights = np.array(UF_WEIGHTS, dtype=float)
    uf_weights /= uf_weights.sum()

    # ------------------------------------------------------------------
    # Variáveis qualitativas
    # ------------------------------------------------------------------
    sg_uf = rng.choice(UFS, size=n, p=uf_weights)

    # Gênero – ligeiro predomínio feminino (reflexo do ENEM real)
    tp_sexo = rng.choice(["M", "F"], size=n, p=[0.43, 0.57])

    # Raça/cor  (0=Não declarado, 1=Branca, 2=Parda, 3=Preta, 4=Amarela, 5=Indígena)
    tp_cor_raca = rng.choice([0, 1, 2, 3, 4, 5], size=n,
                             p=[0.03, 0.27, 0.43, 0.12, 0.09, 0.06])

    # Tipo de escola  (1=Não respondeu, 2=Pública, 3=Privada, 4=Exterior)
    tp_escola = rng.choice([1, 2, 3, 4], size=n, p=[0.04, 0.61, 0.34, 0.01])

    # Situação de conclusão do Ensino Médio
    # 1=Já concluiu  2=Cursando e concluirá em 2024
    # 3=Cursando e não concluirá em 2024  4=Não está cursando
    tp_st_conclusao = rng.choice([1, 2, 3, 4], size=n, p=[0.35, 0.45, 0.12, 0.08])

    # Treineiro (0=Não, 1=Sim)
    in_treineiro = rng.choice([0, 1], size=n, p=[0.84, 0.16])

    # Faixa etária concentrada entre 16 e 30 anos com cauda até 60
    nu_idade = rng.integers(14, 61, size=n)

    # ------------------------------------------------------------------
    # Variáveis quantitativas (notas)
    # ------------------------------------------------------------------
    # Parâmetros realistas baseados nos resultados históricos do ENEM
    nota_cn = rng.normal(loc=490, scale=100, size=n).clip(0, 1000).round(1)
    nota_ch = rng.normal(loc=510, scale=95, size=n).clip(0, 1000).round(1)
    nota_lc = rng.normal(loc=525, scale=90, size=n).clip(0, 1000).round(1)
    nota_mt = rng.normal(loc=480, scale=115, size=n).clip(0, 1000).round(1)

    # Redação: múltiplos de 40 entre 0 e 1000 (escala do ENEM)
    redacao_choices = np.array([0] + list(range(40, 1001, 40)))
    redacao_probs = np.array(
        [0.04] + [0.96 / len(range(40, 1001, 40))] * len(range(40, 1001, 40))
    )
    redacao_probs /= redacao_probs.sum()
    nota_redacao = rng.choice(redacao_choices, size=n, p=redacao_probs).astype(float)

    # Zera notas de participantes que faltaram (≈5 % por prova, independente)
    for arr in [nota_cn, nota_ch, nota_lc, nota_mt]:
        mask = rng.random(n) < 0.05
        arr[mask] = np.nan

    df = pd.DataFrame(
        {
            "NU_INSCRICAO": np.arange(1, n + 1),
            "NU_ANO": 2024,
            "SG_UF_RESIDENCIA": sg_uf,
            "NU_IDADE": nu_idade,
            "TP_SEXO": tp_sexo,
            "TP_COR_RACA": tp_cor_raca,
            "TP_ESCOLA": tp_escola,
            "TP_ST_CONCLUSAO": tp_st_conclusao,
            "IN_TREINEIRO": in_treineiro,
            "NU_NOTA_CN": nota_cn,
            "NU_NOTA_CH": nota_ch,
            "NU_NOTA_LC": nota_lc,
            "NU_NOTA_MT": nota_mt,
            "NU_NOTA_REDACAO": nota_redacao,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Conexão ao banco
# ---------------------------------------------------------------------------

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Retorna uma conexão SQLite para o banco informado."""
    return sqlite3.connect(db_path)


def _ensure_table(conn: sqlite3.Connection) -> None:
    """Popula a tabela `enem2024` caso ela ainda não exista."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='enem2024'"
    )
    if cursor.fetchone() is not None:
        return  # tabela já existe

    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, sep=";", encoding="latin-1", low_memory=False)
        # Mantém apenas as colunas que o restante da aplicação precisa
        cols_needed = [
            "NU_INSCRICAO", "NU_ANO", "SG_UF_RESIDENCIA", "NU_IDADE",
            "TP_SEXO", "TP_COR_RACA", "TP_ESCOLA", "TP_ST_CONCLUSAO",
            "IN_TREINEIRO", "NU_NOTA_CN", "NU_NOTA_CH",
            "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_REDACAO",
        ]
        existing = [c for c in cols_needed if c in df.columns]
        df = df[existing]
    else:
        df = _create_synthetic_data()

    df.to_sql("enem2024", conn, if_exists="replace", index=False)
    conn.commit()


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def load_data(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Carrega os dados do ENEM 2024 do banco SQLite.

    Se o banco (ou a tabela) ainda não existir, os dados são criados
    automaticamente antes de serem retornados.

    Parameters
    ----------
    db_path : str
        Caminho para o arquivo SQLite.

    Returns
    -------
    pd.DataFrame
        DataFrame com os microdados do ENEM 2024.
    """
    conn = get_connection(db_path)
    try:
        _ensure_table(conn)
        df = pd.read_sql("SELECT * FROM enem2024", conn)
    finally:
        conn.close()
    return df
