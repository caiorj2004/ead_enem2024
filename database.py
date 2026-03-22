"""
database.py – Conexão ao banco de dados e coleta dos dados do ENEM 2024.

Estrutura do banco real
-----------------------
Tabela 1 – ed_enem_2024_participantes  (microdados de inscrição)
Tabela 2 – ed_enem_2024_resultados     (notas por participante)
Tabela 3 – municipio                   (referência de municípios)

Não existe chave primária/estrangeira entre participantes e resultados,
por isso a estratégia é agregar ambas as tabelas ao nível municipal via
GROUP BY co_municipio_prova, e então fazer JOIN com a tabela de
municípios. Isso reproduz exatamente o pipeline do script Colab.

Modos de operação
-----------------
1. **Banco real (PostgreSQL)**
   Quando `load_data()` recebe um `db_config` com credenciais válidas,
   executa as 3 queries de agregação e retorna o DataFrame consolidado.
   Configure as credenciais no Streamlit Cloud em Settings → Secrets.

2. **Modo template / demonstração**
   Sem credenciais (ou se a conexão falhar), os dados são gerados
   sinteticamente ao nível municipal, permitindo que o dashboard funcione
   completamente antes das credenciais serem fornecidas.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------

_SQLITE_PATH = os.path.join(os.path.dirname(__file__), "enem2024.db")

SYNTHETIC_MUNICIPALITIES = 500
RNG_SEED = 42

# ---------------------------------------------------------------------------
# SQL queries – réplica do pipeline do Colab
# ---------------------------------------------------------------------------

QUERY_PARTICIPANTES = """
SELECT
    co_municipio_prova,
    COUNT(*) AS total_inscritos,
    AVG(idade_calculada) AS media_idade,
    SUM(CASE WHEN tp_sexo = 'Feminino'   THEN 1 ELSE 0 END) AS sexo_feminino,
    SUM(CASE WHEN tp_sexo = 'Masculino'  THEN 1 ELSE 0 END) AS sexo_masculino,
    SUM(CASE WHEN tp_cor_raca = 'Branca'   THEN 1 ELSE 0 END) AS raca_branca,
    SUM(CASE WHEN tp_cor_raca = 'Preta'    THEN 1 ELSE 0 END) AS raca_preta,
    SUM(CASE WHEN tp_cor_raca = 'Parda'    THEN 1 ELSE 0 END) AS raca_parda,
    SUM(CASE WHEN tp_cor_raca = 'Amarela'  THEN 1 ELSE 0 END) AS raca_amarela,
    SUM(CASE WHEN tp_cor_raca = 'Indígena' THEN 1 ELSE 0 END) AS raca_indigena,
    SUM(CASE WHEN tp_estado_civil = 'Solteiro(a)' THEN 1 ELSE 0 END) AS est_civil_solteiro,
    SUM(CASE WHEN tp_estado_civil = 'Casado(a)'   THEN 1 ELSE 0 END) AS est_civil_casado,
    SUM(CASE WHEN tp_nacionalidade = 'Brasileiro(a)' THEN 1 ELSE 0 END) AS nac_brasileiro,
    SUM(CASE WHEN tp_st_conclusao = 'Já concluí o Ensino Médio' THEN 1 ELSE 0 END) AS concluiu_em,
    SUM(CASE WHEN tp_ensino = 'Ensino Regular' THEN 1 ELSE 0 END) AS ensino_regular,
    SUM(CASE WHEN q006 IN ('D','E','F','G','H','I','J','K','L','M','N','O','P','Q')
             THEN 1 ELSE 0 END) AS classe_media_alta,
    SUM(CASE WHEN q020 = 'Sim' OR q020 = 'B' THEN 1 ELSE 0 END) AS tem_internet,
    SUM(CASE WHEN q021 = 'Sim' OR q021 IN ('B','C','D','E') THEN 1 ELSE 0 END) AS tem_computador
FROM ed_enem_2024_participantes
GROUP BY co_municipio_prova
"""

QUERY_RESULTADOS = """
SELECT
    co_municipio_prova,
    AVG(nota_cn_ciencias_da_natureza) AS media_cn,
    AVG(nota_ch_ciencias_humanas)     AS media_ch,
    AVG(nota_lc_linguagens_e_codigos) AS media_lc,
    AVG(nota_mt_matematica)           AS media_mt,
    AVG(nota_redacao)                 AS media_redacao,
    AVG(nota_media_5_notas)           AS media_geral
FROM ed_enem_2024_resultados
WHERE nota_redacao IS NOT NULL
GROUP BY co_municipio_prova
"""

QUERY_MUNICIPIOS = "SELECT codigo_municipio_dv, nome_municipio, cd_uf FROM municipio"

# ---------------------------------------------------------------------------
# Dados auxiliares para geração sintética
# ---------------------------------------------------------------------------

# (UF abbreviation, approximate number of IBGE municipalities)
_UF_MUN_COUNTS = [
    ("SP", 645), ("MG", 853), ("BA", 417), ("RS", 497), ("PR", 399),
    ("SC", 295), ("RJ",  92), ("CE", 184), ("PE", 185), ("PA", 144),
    ("GO", 246), ("PB", 223), ("MA", 217), ("PI", 224), ("AL", 102),
    ("RN", 167), ("ES",  78), ("MS",  79), ("MT", 141), ("AM",  62),
    ("DF",   1), ("SE",  75), ("TO", 139), ("RO",  52), ("AC",  22),
    ("AP",  16), ("RR",  15),
]
_UFS       = [u for u, _ in _UF_MUN_COUNTS]
_MUN_TOTAL = sum(c for _, c in _UF_MUN_COUNTS)
_UF_W      = [c / _MUN_TOTAL for _, c in _UF_MUN_COUNTS]


# ---------------------------------------------------------------------------
# Geração de dados sintéticos (nível municipal)
# ---------------------------------------------------------------------------

def _create_synthetic_data(n: int = SYNTHETIC_MUNICIPALITIES) -> pd.DataFrame:
    """
    Gera um DataFrame sintético com uma linha por município, reproduzindo a
    estrutura de colunas produzida pelo pipeline de agregação do Colab.
    """
    rng = np.random.default_rng(RNG_SEED)

    ufs = rng.choice(_UFS, size=n, p=_UF_W)

    # Código de município sintético (7 dígitos)
    cod_7 = [str(rng.integers(1_000_000, 9_999_999)) for _ in range(n)]

    # Nomes de municípios sintéticos
    municipio = [f"Município {i+1:04d}" for i in range(n)]

    # Total de inscritos – log-normal: cidades pequenas ~200, grandes ~20000
    total = np.round(rng.lognormal(mean=6.5, sigma=1.2, size=n)).astype(int).clip(30, 80_000)

    # Idade média
    media_idade = rng.uniform(17.5, 22.5, size=n).round(2)

    # Proporções demográficas por município
    pct_fem      = rng.uniform(0.45, 0.65, size=n)
    pct_parda    = rng.uniform(0.25, 0.60, size=n)
    pct_branca   = rng.uniform(0.15, 0.55, size=n)
    pct_preta    = rng.uniform(0.05, 0.20, size=n)
    pct_amarela  = rng.uniform(0.01, 0.08, size=n)
    pct_indigena = np.clip(1 - pct_parda - pct_branca - pct_preta - pct_amarela, 0.01, 0.15)

    sexo_feminino  = np.round(total * pct_fem).astype(int)
    sexo_masculino = total - sexo_feminino
    raca_parda    = np.round(total * pct_parda).astype(int)
    raca_branca   = np.round(total * pct_branca).astype(int)
    raca_preta    = np.round(total * pct_preta).astype(int)
    raca_amarela  = np.round(total * pct_amarela).astype(int)
    raca_indigena = np.round(total * pct_indigena).astype(int)

    est_civil_solteiro = np.round(total * rng.uniform(0.75, 0.92, size=n)).astype(int)
    est_civil_casado   = np.round(total * rng.uniform(0.02, 0.12, size=n)).astype(int)
    nac_brasileiro     = np.round(total * rng.uniform(0.95, 1.00, size=n)).astype(int)
    concluiu_em        = np.round(total * rng.uniform(0.20, 0.55, size=n)).astype(int)
    ensino_regular     = np.round(total * rng.uniform(0.35, 0.65, size=n)).astype(int)
    classe_media_alta  = np.round(total * rng.uniform(0.00, 0.15, size=n)).astype(int)
    tem_internet       = np.round(total * rng.uniform(0.70, 0.97, size=n)).astype(int)
    tem_computador     = np.round(total * rng.uniform(0.20, 0.65, size=n)).astype(int)

    # Notas médias por município (com correlação realista entre áreas)
    base_score = rng.normal(loc=500, scale=30, size=n)
    media_cn      = (base_score + rng.normal(0, 15, n)).clip(380, 680).round(2)
    media_ch      = (base_score + rng.normal(5, 12, n)).clip(390, 690).round(2)
    media_lc      = (base_score + rng.normal(8, 10, n)).clip(400, 695).round(2)
    media_mt      = (base_score + rng.normal(-5, 20, n)).clip(370, 700).round(2)
    media_redacao = (base_score * 1.15 + rng.normal(30, 50, n)).clip(350, 820).round(2)
    media_geral   = ((media_cn + media_ch + media_lc + media_mt + media_redacao) / 5).round(2)

    return pd.DataFrame({
        "cod_7":              cod_7,
        "municipio":          municipio,
        "uf":                 ufs,
        "total_inscritos":    total,
        "media_idade":        media_idade,
        "sexo_feminino":      sexo_feminino,
        "sexo_masculino":     sexo_masculino,
        "raca_branca":        raca_branca,
        "raca_preta":         raca_preta,
        "raca_parda":         raca_parda,
        "raca_amarela":       raca_amarela,
        "raca_indigena":      raca_indigena,
        "est_civil_solteiro": est_civil_solteiro,
        "est_civil_casado":   est_civil_casado,
        "nac_brasileiro":     nac_brasileiro,
        "concluiu_em":        concluiu_em,
        "ensino_regular":     ensino_regular,
        "classe_media_alta":  classe_media_alta,
        "tem_internet":       tem_internet,
        "tem_computador":     tem_computador,
        "media_cn":           media_cn,
        "media_ch":           media_ch,
        "media_lc":           media_lc,
        "media_mt":           media_mt,
        "media_redacao":      media_redacao,
        "media_geral":        media_geral,
    })


# ---------------------------------------------------------------------------
# PostgreSQL – conexão real (3 queries + merge)
# ---------------------------------------------------------------------------

def _pg_load(db_config: dict[str, Any]) -> pd.DataFrame:
    """
    Executa as 3 queries de agregação no PostgreSQL e retorna o DataFrame
    consolidado ao nível municipal, seguindo o pipeline do Colab.

    Raises
    ------
    ImportError      – se psycopg2 não estiver instalado.
    psycopg2.Error   – erros de conexão / query.
    """
    try:
        import psycopg2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "psycopg2-binary não está instalado. "
            "Adicione-o a requirements.txt para usar o banco PostgreSQL."
        ) from exc

    conn = psycopg2.connect(
        host=db_config["host"],
        port=int(db_config.get("port", 5432)),
        dbname=db_config["dbname"],
        user=db_config["user"],
        password=db_config["password"],
    )
    try:
        df_part = pd.read_sql(QUERY_PARTICIPANTES, conn)
        df_res  = pd.read_sql(QUERY_RESULTADOS, conn)
        df_mun  = pd.read_sql(QUERY_MUNICIPIOS, conn)
    finally:
        conn.close()

    # Renomeia e normaliza as chaves (mesmo mapeamento do Colab)
    rename_map = {
        "co_municipio_prova":   "cod_7",
        "codigo_municipio_dv":  "cod_7",
        "nome_municipio":       "municipio",
        "cd_uf":                "uf",
    }
    for df in (df_part, df_res, df_mun):
        df.rename(columns=rename_map, inplace=True)
        df["cod_7"] = df["cod_7"].astype(str).str.strip()

    df_consolidated = pd.merge(df_part, df_res, on="cod_7", how="inner")
    df_final        = pd.merge(df_consolidated, df_mun, on="cod_7", how="left")
    return df_final


# ---------------------------------------------------------------------------
# SQLite – modo template / demonstração
# ---------------------------------------------------------------------------

def _sqlite_load(db_path: str = _SQLITE_PATH) -> pd.DataFrame:
    """Carrega (ou cria) os dados sintéticos municipais em SQLite local."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='enem2024_municipal'"
        )
        if cursor.fetchone() is None:
            _create_synthetic_data().to_sql(
                "enem2024_municipal", conn, if_exists="replace", index=False
            )
            conn.commit()
        df = pd.read_sql("SELECT * FROM enem2024_municipal", conn)
    finally:
        conn.close()
    return df


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def load_data(db_config: dict[str, Any] | None = None) -> tuple[pd.DataFrame, bool]:
    """
    Carrega os dados do ENEM 2024 agregados ao nível municipal.

    Tenta, em ordem:
      1. Conectar ao PostgreSQL via `db_config` e executar o pipeline de 3 queries.
      2. Usar banco SQLite local com dados sintéticos municipais (modo template).

    Parameters
    ----------
    db_config : dict | None
        Credenciais do banco real (lidas de st.secrets em app.py).
        Chaves esperadas: ``host``, ``port``, ``dbname``, ``user``, ``password``.
        Se ``None`` ou se a conexão falhar, usa o modo template.

    Returns
    -------
    df       : pd.DataFrame com uma linha por município.
    is_demo  : bool – True quando os dados são sintéticos (modo template).
    """
    if db_config:
        try:
            import psycopg2  # type: ignore[import-untyped]
            df = _pg_load(db_config)
            return df, False
        except (ImportError, psycopg2.Error, ValueError):
            pass

    return _sqlite_load(), True
