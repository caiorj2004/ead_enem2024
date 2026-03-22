"""
database.py – Conexão ao banco de dados e coleta dos dados do ENEM 2024.

Estrutura do banco real
-----------------------
Tabela 1 – ed_enem_2024_participantes  (microdados de inscrição + questionário Q001-Q023)
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

# Bump this name whenever the schema changes to force SQLite cache recreation
_SQLITE_TABLE = "enem2024_municipal_v2"

SYNTHETIC_MUNICIPALITIES = 500
RNG_SEED = 42

# ---------------------------------------------------------------------------
# SQL queries – réplica do pipeline do Colab (Q001-Q023 exaustivo)
# ---------------------------------------------------------------------------

QUERY_PARTICIPANTES = """
SELECT
    co_municipio_prova,
    COUNT(*) AS total_inscritos,
    AVG(idade_calculada) AS media_idade,

    -- [DADOS DO PARTICIPANTE]
    SUM(CASE WHEN tp_sexo = 'Feminino'   THEN 1 ELSE 0 END) AS sexo_feminino,
    SUM(CASE WHEN tp_sexo = 'Masculino'  THEN 1 ELSE 0 END) AS sexo_masculino,
    SUM(CASE WHEN tp_cor_raca = 'Branca'   THEN 1 ELSE 0 END) AS raca_branca,
    SUM(CASE WHEN tp_cor_raca = 'Preta'    THEN 1 ELSE 0 END) AS raca_preta,
    SUM(CASE WHEN tp_cor_raca = 'Parda'    THEN 1 ELSE 0 END) AS raca_parda,
    SUM(CASE WHEN tp_cor_raca = 'Amarela'  THEN 1 ELSE 0 END) AS raca_amarela,
    SUM(CASE WHEN tp_cor_raca = 'Indígena' THEN 1 ELSE 0 END) AS raca_indigena,
    SUM(CASE WHEN tp_estado_civil = 'Solteiro(a)'    THEN 1 ELSE 0 END) AS est_civil_solteiro,
    SUM(CASE WHEN tp_estado_civil LIKE 'Casado%'     THEN 1 ELSE 0 END) AS est_civil_casado,
    SUM(CASE WHEN tp_nacionalidade = 'Brasileiro(a)' THEN 1 ELSE 0 END) AS nacionalidade_brasileiro,
    SUM(CASE WHEN in_treineiro = 'Sim'               THEN 1 ELSE 0 END) AS treineiro_sim,

    -- [QUESTIONÁRIO - ESCOLARIDADE E OCUPAÇÃO]
    SUM(CASE WHEN q001 = 'Completou a Pós-graduação' THEN 1 ELSE 0 END) AS escolaridade_pai_pos,
    SUM(CASE WHEN q002 = 'Completou a Pós-graduação' THEN 1 ELSE 0 END) AS escolaridade_mae_pos,
    SUM(CASE WHEN q003 LIKE 'Grupo 5%'               THEN 1 ELSE 0 END) AS ocupacao_pai_grupo5,
    SUM(CASE WHEN q004 LIKE 'Grupo 5%'               THEN 1 ELSE 0 END) AS ocupacao_mae_grupo5,

    -- [QUESTIONÁRIO - RENDA]
    SUM(CASE WHEN q006 = 'Sim'                         THEN 1 ELSE 0 END) AS possui_renda_sim,
    SUM(CASE WHEN q007 = 'Nenhuma renda'               THEN 1 ELSE 0 END) AS renda_familiar_nenhuma,
    SUM(CASE WHEN q007 = 'Acima de R$ 28.240,00'       THEN 1 ELSE 0 END) AS renda_familiar_classe_a,

    -- [QUESTIONÁRIO - BENS E TECNOLOGIA]
    SUM(CASE WHEN q008 LIKE 'Sim%'                                             THEN 1 ELSE 0 END) AS empregado_domestico_sim,
    SUM(CASE WHEN q009 LIKE 'Sim, um'                                          THEN 1 ELSE 0 END) AS banheiro_1,
    SUM(CASE WHEN q009 LIKE 'Sim, dois'                                        THEN 1 ELSE 0 END) AS banheiro_2,
    SUM(CASE WHEN q009 LIKE 'Sim, três%'                                       THEN 1 ELSE 0 END) AS banheiro_3_ou_mais,
    SUM(CASE WHEN q010 LIKE 'Sim, três%'                                       THEN 1 ELSE 0 END) AS quarto_3_ou_mais,
    SUM(CASE WHEN q011 LIKE 'Sim, um'                                          THEN 1 ELSE 0 END) AS carro_1,
    SUM(CASE WHEN q011 LIKE 'Sim, dois' OR q011 LIKE 'Sim, três%'             THEN 1 ELSE 0 END) AS carro_2_ou_mais,
    SUM(CASE WHEN q012 LIKE 'Sim%'                                             THEN 1 ELSE 0 END) AS motocicleta_sim,
    SUM(CASE WHEN q013 LIKE 'Sim%'                                             THEN 1 ELSE 0 END) AS geladeira_sim,
    SUM(CASE WHEN q014 = 'Sim'                                                 THEN 1 ELSE 0 END) AS freezer_sim,
    SUM(CASE WHEN q015 = 'Sim'                                                 THEN 1 ELSE 0 END) AS maquina_lavar_sim,
    SUM(CASE WHEN q016 = 'Sim'                                                 THEN 1 ELSE 0 END) AS micro_ondas_sim,
    SUM(CASE WHEN q017 = 'Sim'                                                 THEN 1 ELSE 0 END) AS aspirador_po_sim,
    SUM(CASE WHEN q018 LIKE 'Sim%'                                             THEN 1 ELSE 0 END) AS tv_sim,
    SUM(CASE WHEN q019 = 'Sim'                                                 THEN 1 ELSE 0 END) AS tv_assinatura_sim,
    SUM(CASE WHEN q020 = 'Sim'                                                 THEN 1 ELSE 0 END) AS internet_wifi_sim,
    SUM(CASE WHEN q021 LIKE 'Sim, um'                                          THEN 1 ELSE 0 END) AS computador_1,
    SUM(CASE WHEN q021 LIKE 'Sim, dois' OR q021 LIKE 'Sim, três%'
             OR q021 LIKE 'Sim, quatro%'                                       THEN 1 ELSE 0 END) AS computador_2_ou_mais,
    SUM(CASE WHEN q022 LIKE 'Sim, três%' OR q022 LIKE 'Sim, quatro%'          THEN 1 ELSE 0 END) AS celular_3_ou_mais,

    -- [QUESTIONÁRIO - ESCOLA]
    SUM(CASE WHEN q023 = 'Somente em escola pública'          THEN 1 ELSE 0 END) AS tipo_escola_publica,
    SUM(CASE WHEN q023 LIKE 'Somente em escola privada%'      THEN 1 ELSE 0 END) AS tipo_escola_privada

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
    estrutura de colunas produzida pelo pipeline de agregação do Colab
    (Q001-Q023 exaustivo + notas renomeadas para nota_*).
    """
    rng = np.random.default_rng(RNG_SEED)

    ufs       = rng.choice(_UFS, size=n, p=_UF_W)
    cod_7     = [str(rng.integers(1_000_000, 9_999_999)) for _ in range(n)]
    municipio = [f"Município {i+1:04d}" for i in range(n)]

    # Total de inscritos – log-normal
    total      = np.round(rng.lognormal(mean=6.5, sigma=1.2, size=n)).astype(int).clip(30, 80_000)
    media_idade = rng.uniform(17.5, 22.5, size=n).round(2)

    def _cnt(pct: np.ndarray) -> np.ndarray:
        """Convert proportion array to rounded count based on `total`."""
        return np.round(total * pct).astype(int)

    # ── Demografics ────────────────────────────────────────────────────────
    pct_fem      = rng.uniform(0.45, 0.65, size=n)
    pct_parda    = rng.uniform(0.25, 0.60, size=n)
    pct_branca   = rng.uniform(0.15, 0.55, size=n)
    pct_preta    = rng.uniform(0.05, 0.20, size=n)
    pct_amarela  = rng.uniform(0.01, 0.08, size=n)
    pct_indigena = np.clip(1 - pct_parda - pct_branca - pct_preta - pct_amarela, 0.01, 0.15)

    sexo_feminino           = _cnt(pct_fem)
    sexo_masculino          = total - sexo_feminino
    raca_parda              = _cnt(pct_parda)
    raca_branca             = _cnt(pct_branca)
    raca_preta              = _cnt(pct_preta)
    raca_amarela            = _cnt(pct_amarela)
    raca_indigena           = _cnt(pct_indigena)
    est_civil_solteiro      = _cnt(rng.uniform(0.75, 0.92, size=n))
    est_civil_casado        = _cnt(rng.uniform(0.02, 0.12, size=n))
    nacionalidade_brasileiro = _cnt(rng.uniform(0.95, 1.00, size=n))
    treineiro_sim           = _cnt(rng.uniform(0.05, 0.20, size=n))

    # ── Questionário: Escolaridade e Ocupação (Q001-Q004) ─────────────────
    escolaridade_pai_pos = _cnt(rng.uniform(0.02, 0.12, size=n))
    escolaridade_mae_pos = _cnt(rng.uniform(0.03, 0.14, size=n))
    ocupacao_pai_grupo5  = _cnt(rng.uniform(0.01, 0.10, size=n))
    ocupacao_mae_grupo5  = _cnt(rng.uniform(0.01, 0.08, size=n))

    # ── Questionário: Renda (Q006-Q007) ───────────────────────────────────
    possui_renda_sim       = _cnt(rng.uniform(0.20, 0.60, size=n))
    renda_familiar_nenhuma = _cnt(rng.uniform(0.05, 0.25, size=n))
    renda_familiar_classe_a = _cnt(rng.uniform(0.00, 0.08, size=n))

    # ── Questionário: Bens e Tecnologia (Q008-Q022) ───────────────────────
    empregado_domestico_sim = _cnt(rng.uniform(0.02, 0.15, size=n))
    banheiro_1              = _cnt(rng.uniform(0.40, 0.65, size=n))
    banheiro_2              = _cnt(rng.uniform(0.15, 0.35, size=n))
    banheiro_3_ou_mais      = _cnt(rng.uniform(0.02, 0.12, size=n))
    quarto_3_ou_mais        = _cnt(rng.uniform(0.10, 0.35, size=n))
    carro_1                 = _cnt(rng.uniform(0.25, 0.55, size=n))
    carro_2_ou_mais         = _cnt(rng.uniform(0.05, 0.20, size=n))
    motocicleta_sim         = _cnt(rng.uniform(0.20, 0.55, size=n))
    geladeira_sim           = _cnt(rng.uniform(0.80, 0.98, size=n))
    freezer_sim             = _cnt(rng.uniform(0.10, 0.40, size=n))
    maquina_lavar_sim       = _cnt(rng.uniform(0.50, 0.85, size=n))
    micro_ondas_sim         = _cnt(rng.uniform(0.40, 0.75, size=n))
    aspirador_po_sim        = _cnt(rng.uniform(0.05, 0.30, size=n))
    tv_sim                  = _cnt(rng.uniform(0.75, 0.97, size=n))
    tv_assinatura_sim       = _cnt(rng.uniform(0.15, 0.50, size=n))
    internet_wifi_sim       = _cnt(rng.uniform(0.55, 0.95, size=n))
    computador_1            = _cnt(rng.uniform(0.25, 0.55, size=n))
    computador_2_ou_mais    = _cnt(rng.uniform(0.05, 0.20, size=n))
    celular_3_ou_mais       = _cnt(rng.uniform(0.05, 0.25, size=n))

    # ── Questionário: Escola (Q023) ───────────────────────────────────────
    pct_publica         = rng.uniform(0.35, 0.85, size=n)
    tipo_escola_publica = _cnt(pct_publica)
    tipo_escola_privada = _cnt(np.clip(1 - pct_publica, 0.05, 0.60))

    # ── Notas médias municipais ───────────────────────────────────────────
    base_score           = rng.normal(loc=500, scale=30, size=n)
    nota_ciencias_natureza = (base_score + rng.normal(0, 15, n)).clip(380, 680).round(2)
    nota_ciencias_humanas  = (base_score + rng.normal(5, 12, n)).clip(390, 690).round(2)
    nota_linguagens        = (base_score + rng.normal(8, 10, n)).clip(400, 695).round(2)
    nota_matematica        = (base_score + rng.normal(-5, 20, n)).clip(370, 700).round(2)
    nota_redacao           = (base_score * 1.15 + rng.normal(30, 50, n)).clip(350, 820).round(2)
    nota_geral_media       = (
        (nota_ciencias_natureza + nota_ciencias_humanas + nota_linguagens
         + nota_matematica + nota_redacao) / 5
    ).round(2)

    return pd.DataFrame({
        "cod_7":                    cod_7,
        "municipio":                municipio,
        "uf":                       ufs,
        "total_inscritos":          total,
        "media_idade":              media_idade,
        # Demografics
        "sexo_feminino":            sexo_feminino,
        "sexo_masculino":           sexo_masculino,
        "raca_branca":              raca_branca,
        "raca_preta":               raca_preta,
        "raca_parda":               raca_parda,
        "raca_amarela":             raca_amarela,
        "raca_indigena":            raca_indigena,
        "est_civil_solteiro":       est_civil_solteiro,
        "est_civil_casado":         est_civil_casado,
        "nacionalidade_brasileiro": nacionalidade_brasileiro,
        "treineiro_sim":            treineiro_sim,
        # Q001-Q004 Escolaridade / Ocupação
        "escolaridade_pai_pos":     escolaridade_pai_pos,
        "escolaridade_mae_pos":     escolaridade_mae_pos,
        "ocupacao_pai_grupo5":      ocupacao_pai_grupo5,
        "ocupacao_mae_grupo5":      ocupacao_mae_grupo5,
        # Q006-Q007 Renda
        "possui_renda_sim":         possui_renda_sim,
        "renda_familiar_nenhuma":   renda_familiar_nenhuma,
        "renda_familiar_classe_a":  renda_familiar_classe_a,
        # Q008-Q022 Bens e Tecnologia
        "empregado_domestico_sim":  empregado_domestico_sim,
        "banheiro_1":               banheiro_1,
        "banheiro_2":               banheiro_2,
        "banheiro_3_ou_mais":       banheiro_3_ou_mais,
        "quarto_3_ou_mais":         quarto_3_ou_mais,
        "carro_1":                  carro_1,
        "carro_2_ou_mais":          carro_2_ou_mais,
        "motocicleta_sim":          motocicleta_sim,
        "geladeira_sim":            geladeira_sim,
        "freezer_sim":              freezer_sim,
        "maquina_lavar_sim":        maquina_lavar_sim,
        "micro_ondas_sim":          micro_ondas_sim,
        "aspirador_po_sim":         aspirador_po_sim,
        "tv_sim":                   tv_sim,
        "tv_assinatura_sim":        tv_assinatura_sim,
        "internet_wifi_sim":        internet_wifi_sim,
        "computador_1":             computador_1,
        "computador_2_ou_mais":     computador_2_ou_mais,
        "celular_3_ou_mais":        celular_3_ou_mais,
        # Q023 Escola
        "tipo_escola_publica":      tipo_escola_publica,
        "tipo_escola_privada":      tipo_escola_privada,
        # Notas (renamed to nota_*)
        "nota_ciencias_natureza":   nota_ciencias_natureza,
        "nota_ciencias_humanas":    nota_ciencias_humanas,
        "nota_linguagens":          nota_linguagens,
        "nota_matematica":          nota_matematica,
        "nota_redacao":             nota_redacao,
        "nota_geral_media":         nota_geral_media,
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

    # Renomeia chaves e colunas de notas (mesmo mapeamento do Colab)
    rename_map = {
        "co_municipio_prova":  "cod_7",
        "codigo_municipio_dv": "cod_7",
        "nome_municipio":      "municipio",
        "cd_uf":               "uf",
        # Score renames
        "media_cn":       "nota_ciencias_natureza",
        "media_ch":       "nota_ciencias_humanas",
        "media_lc":       "nota_linguagens",
        "media_mt":       "nota_matematica",
        "media_redacao":  "nota_redacao",
        "media_geral":    "nota_geral_media",
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
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (_SQLITE_TABLE,),
        )
        if cursor.fetchone() is None:
            _create_synthetic_data().to_sql(
                _SQLITE_TABLE, conn, if_exists="replace", index=False
            )
            conn.commit()
        df = pd.read_sql(f"SELECT * FROM {_SQLITE_TABLE}", conn)  # noqa: S608
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
