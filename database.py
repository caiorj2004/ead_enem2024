"""
database.py – Conexão ao banco de dados e coleta dos dados do ENEM 2024.

Modos de operação
-----------------
1. **Banco real (PostgreSQL)**
   Quando `load_data()` recebe um `db_config` com credenciais válidas,
   conecta-se ao banco PostgreSQL e retorna os dados reais.
   As credenciais nunca devem ser commitadas no repositório; configure-as
   via Streamlit Cloud em *Settings → Secrets* seguindo o template em
   `.streamlit/secrets.toml`.

2. **Modo template / demonstração**
   Sem credenciais (ou se a conexão falhar), os dados são gerados
   sinteticamente com a mesma estrutura dos microdados do ENEM 2024,
   permitindo que o dashboard funcione antes das credenciais serem
   fornecidas.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configurações – modo template
# ---------------------------------------------------------------------------

_SQLITE_PATH = os.path.join(os.path.dirname(__file__), "enem2024.db")

# Número de registros sintéticos gerados no modo template
SYNTHETIC_ROWS = 15_000

# Semente para reprodutibilidade
RNG_SEED = 42

# Colunas que a aplicação espera encontrar no banco real
EXPECTED_COLUMNS = [
    "NU_INSCRICAO",
    "NU_ANO",
    "SG_UF_RESIDENCIA",
    "NU_IDADE",
    "TP_SEXO",
    "TP_COR_RACA",
    "TP_ESCOLA",
    "TP_ST_CONCLUSAO",
    "IN_TREINEIRO",
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
]

# ---------------------------------------------------------------------------
# Dados auxiliares para geração sintética
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
# PostgreSQL – conexão real
# ---------------------------------------------------------------------------

def _pg_load(db_config: dict[str, Any]) -> pd.DataFrame:
    """
    Conecta ao banco PostgreSQL usando as credenciais fornecidas e retorna
    os dados da tabela configurada.

    Parameters
    ----------
    db_config : dict com as chaves:
        host, port, dbname, user, password, table (opcional, padrão "enem2024")

    Returns
    -------
    pd.DataFrame com as colunas de EXPECTED_COLUMNS presentes na tabela.

    Raises
    ------
    ImportError  – se psycopg2 não estiver instalado.
    Exception    – qualquer erro de conexão ou query é propagado para que
                   `load_data()` possa fazer o fallback adequado.
    """
    try:
        import psycopg2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "psycopg2-binary não está instalado. "
            "Adicione-o a requirements.txt para usar o banco PostgreSQL."
        ) from exc

    table = db_config.get("table", "enem2024")
    # Validate table name to prevent SQL injection
    if not table.replace("_", "").isalnum():
        raise ValueError(f"Invalid table name: {table!r}")
    cols  = ", ".join(EXPECTED_COLUMNS)
    query = f"SELECT {cols} FROM {table};"

    conn = psycopg2.connect(
        host=db_config["host"],
        port=int(db_config.get("port", 5432)),
        dbname=db_config["dbname"],
        user=db_config["user"],
        password=db_config["password"],
    )
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    return df


# ---------------------------------------------------------------------------
# SQLite – modo template / demonstração
# ---------------------------------------------------------------------------

def _sqlite_load(db_path: str = _SQLITE_PATH) -> pd.DataFrame:
    """Carrega (ou cria) os dados sintéticos em um banco SQLite local."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='enem2024'"
        )
        if cursor.fetchone() is None:
            _create_synthetic_data().to_sql("enem2024", conn, if_exists="replace", index=False)
            conn.commit()
        df = pd.read_sql("SELECT * FROM enem2024", conn)
    finally:
        conn.close()
    return df


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def load_data(db_config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, bool]:
    """
    Carrega os dados do ENEM 2024.

    Tenta, em ordem:
      1. Conectar ao PostgreSQL usando `db_config` (credenciais reais).
      2. Usar banco SQLite local com dados sintéticos (modo template).

    Parameters
    ----------
    db_config : dict | None
        Dicionário com as credenciais do banco real (lido de st.secrets em
        app.py). Se ``None`` ou se a conexão falhar, usa o modo template.
        Chaves esperadas: ``host``, ``port``, ``dbname``, ``user``,
        ``password``, ``table`` (opcional).

    Returns
    -------
    df       : pd.DataFrame com os microdados.
    is_demo  : bool – True quando os dados são sintéticos (modo template).
    """
    if db_config:
        try:
            import psycopg2  # type: ignore[import-untyped]
            df = _pg_load(db_config)
            # Mantém apenas as colunas conhecidas que existirem
            existing = [c for c in EXPECTED_COLUMNS if c in df.columns]
            return df[existing], False
        except (ImportError, psycopg2.Error, ValueError):
            # Falha silenciosa: continua para o modo template
            pass

    return _sqlite_load(), True

