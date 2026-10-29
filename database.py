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

Modo de operação
----------------
Requer credenciais PostgreSQL configuradas em Streamlit Cloud via
Settings → Secrets (chaves: host, port, dbname, user, password).
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

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
    SUM(CASE WHEN tp_cor_raca = 'Branca'        THEN 1 ELSE 0 END) AS raca_branca,
    SUM(CASE WHEN tp_cor_raca = 'Preta'         THEN 1 ELSE 0 END) AS raca_preta,
    SUM(CASE WHEN tp_cor_raca = 'Parda'         THEN 1 ELSE 0 END) AS raca_parda,
    SUM(CASE WHEN tp_cor_raca = 'Amarela'       THEN 1 ELSE 0 END) AS raca_amarela,
    SUM(CASE WHEN tp_cor_raca = 'Indígena'      THEN 1 ELSE 0 END) AS raca_indigena,
    SUM(CASE WHEN tp_cor_raca = 'Não declarado' THEN 1 ELSE 0 END) AS raca_nao_declarado,
    SUM(CASE WHEN tp_estado_civil = 'Solteiro(a)'              THEN 1 ELSE 0 END) AS est_civil_solteiro,
    SUM(CASE WHEN tp_estado_civil LIKE 'Casado%'               THEN 1 ELSE 0 END) AS est_civil_casado,
    SUM(CASE WHEN tp_estado_civil LIKE 'Divorciado%'           THEN 1 ELSE 0 END) AS est_civil_divorciado,
    SUM(CASE WHEN tp_estado_civil = 'Viúvo(a)'                 THEN 1 ELSE 0 END) AS est_civil_viuvo,
    SUM(CASE WHEN tp_estado_civil = 'Não informado'            THEN 1 ELSE 0 END) AS est_civil_nao_informado,
    SUM(CASE WHEN tp_nacionalidade = 'Brasileiro(a)'                               THEN 1 ELSE 0 END) AS nacionalidade_brasileiro,
    SUM(CASE WHEN tp_nacionalidade LIKE 'Brasileiro(a) Nato%'                      THEN 1 ELSE 0 END) AS nacionalidade_nato_exterior,
    SUM(CASE WHEN tp_nacionalidade LIKE 'Brasileiro(a) Natural%'                   THEN 1 ELSE 0 END) AS nacionalidade_naturalizado,
    SUM(CASE WHEN tp_nacionalidade = 'Estrangeiro(a)'                              THEN 1 ELSE 0 END) AS nacionalidade_estrangeiro,
    SUM(CASE WHEN tp_nacionalidade = 'Não informado'                               THEN 1 ELSE 0 END) AS nacionalidade_nao_informado,
    SUM(CASE WHEN in_treineiro = 'Sim' THEN 1 ELSE 0 END) AS treineiro_sim,
    SUM(CASE WHEN in_treineiro = 'Não' THEN 1 ELSE 0 END) AS treineiro_nao,
    SUM(CASE WHEN tp_st_conclusao = 'Já concluí o Ensino Médio'                                   THEN 1 ELSE 0 END) AS conclusao_ja_concluiu,
    SUM(CASE WHEN tp_st_conclusao = 'Estou cursando e concluirei o Ensino Médio em 2024'           THEN 1 ELSE 0 END) AS conclusao_cursando_2024,
    SUM(CASE WHEN tp_st_conclusao LIKE 'Estou cursando e concluirei o Ensino Médio após%'          THEN 1 ELSE 0 END) AS conclusao_cursando_apos,
    SUM(CASE WHEN tp_st_conclusao LIKE 'Não concluí%'                                              THEN 1 ELSE 0 END) AS conclusao_nao_concluiu,

    -- [QUESTIONÁRIO - ESCOLARIDADE DO PAI (Q001)]
    SUM(CASE WHEN q001 = 'Nunca estudou'                                                                          THEN 1 ELSE 0 END) AS escolaridade_pai_nunca,
    SUM(CASE WHEN q001 = 'Não completou a 4ª série/5º ano do ensino fundamental'                                  THEN 1 ELSE 0 END) AS escolaridade_pai_fund1_inc,
    SUM(CASE WHEN q001 = 'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do ensino fundamental' THEN 1 ELSE 0 END) AS escolaridade_pai_fund1,
    SUM(CASE WHEN q001 = 'Completou a 8ª série/9º ano do ensino fundamental, mas não completou o Ensino Médio'    THEN 1 ELSE 0 END) AS escolaridade_pai_fund2,
    SUM(CASE WHEN q001 = 'Completou o Ensino Médio, mas não completou a Faculdade'                                THEN 1 ELSE 0 END) AS escolaridade_pai_medio,
    SUM(CASE WHEN q001 = 'Completou a Faculdade, mas não completou a Pós-graduação'                               THEN 1 ELSE 0 END) AS escolaridade_pai_superior,
    SUM(CASE WHEN q001 = 'Completou a Pós-graduação'                                                              THEN 1 ELSE 0 END) AS escolaridade_pai_pos,
    SUM(CASE WHEN q001 = 'Não sei'                                                                                THEN 1 ELSE 0 END) AS escolaridade_pai_nao_sei,

    -- [QUESTIONÁRIO - ESCOLARIDADE DA MÃE (Q002)]
    SUM(CASE WHEN q002 = 'Nunca estudou'                                                                          THEN 1 ELSE 0 END) AS escolaridade_mae_nunca,
    SUM(CASE WHEN q002 = 'Não completou a 4ª série/5º ano do ensino fundamental'                                  THEN 1 ELSE 0 END) AS escolaridade_mae_fund1_inc,
    SUM(CASE WHEN q002 = 'Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do ensino fundamental' THEN 1 ELSE 0 END) AS escolaridade_mae_fund1,
    SUM(CASE WHEN q002 = 'Completou a 8ª série/9º ano do ensino fundamental, mas não completou o Ensino Médio'    THEN 1 ELSE 0 END) AS escolaridade_mae_fund2,
    SUM(CASE WHEN q002 = 'Completou o Ensino Médio, mas não completou a Faculdade'                                THEN 1 ELSE 0 END) AS escolaridade_mae_medio,
    SUM(CASE WHEN q002 = 'Completou a Faculdade, mas não completou a Pós-graduação'                               THEN 1 ELSE 0 END) AS escolaridade_mae_superior,
    SUM(CASE WHEN q002 = 'Completou a Pós-graduação'                                                              THEN 1 ELSE 0 END) AS escolaridade_mae_pos,
    SUM(CASE WHEN q002 = 'Não sei'                                                                                THEN 1 ELSE 0 END) AS escolaridade_mae_nao_sei,

    -- [QUESTIONÁRIO - OCUPAÇÃO DO PAI (Q003)]
    SUM(CASE WHEN q003 LIKE 'Grupo 1%' THEN 1 ELSE 0 END) AS ocupacao_pai_grupo1,
    SUM(CASE WHEN q003 LIKE 'Grupo 2%' THEN 1 ELSE 0 END) AS ocupacao_pai_grupo2,
    SUM(CASE WHEN q003 LIKE 'Grupo 3%' THEN 1 ELSE 0 END) AS ocupacao_pai_grupo3,
    SUM(CASE WHEN q003 LIKE 'Grupo 4%' THEN 1 ELSE 0 END) AS ocupacao_pai_grupo4,
    SUM(CASE WHEN q003 LIKE 'Grupo 5%' THEN 1 ELSE 0 END) AS ocupacao_pai_grupo5,
    SUM(CASE WHEN q003 LIKE 'Não sei%' THEN 1 ELSE 0 END) AS ocupacao_pai_nao_sei,

    -- [QUESTIONÁRIO - OCUPAÇÃO DA MÃE (Q004)]
    SUM(CASE WHEN q004 LIKE 'Grupo 1%' THEN 1 ELSE 0 END) AS ocupacao_mae_grupo1,
    SUM(CASE WHEN q004 LIKE 'Grupo 2%' THEN 1 ELSE 0 END) AS ocupacao_mae_grupo2,
    SUM(CASE WHEN q004 LIKE 'Grupo 3%' THEN 1 ELSE 0 END) AS ocupacao_mae_grupo3,
    SUM(CASE WHEN q004 LIKE 'Grupo 4%' THEN 1 ELSE 0 END) AS ocupacao_mae_grupo4,
    SUM(CASE WHEN q004 LIKE 'Grupo 5%' THEN 1 ELSE 0 END) AS ocupacao_mae_grupo5,
    SUM(CASE WHEN q004 LIKE 'Não sei%' THEN 1 ELSE 0 END) AS ocupacao_mae_nao_sei,

    -- [QUESTIONÁRIO - RENDA]
    SUM(CASE WHEN q006 = 'Sim'                                   THEN 1 ELSE 0 END) AS possui_renda_sim,
    SUM(CASE WHEN q007 = 'Nenhuma renda'                         THEN 1 ELSE 0 END) AS renda_familiar_nenhuma,
    SUM(CASE WHEN q007 = 'Até R$ 1.412,00'                       THEN 1 ELSE 0 END) AS renda_familiar_ate_1412,
    SUM(CASE WHEN q007 = 'De R$ 1.412,01 até R$ 2.118,00'        THEN 1 ELSE 0 END) AS renda_familiar_1412_2118,
    SUM(CASE WHEN q007 = 'De R$ 2.118,01 até R$ 2.824,00'        THEN 1 ELSE 0 END) AS renda_familiar_2118_2824,
    SUM(CASE WHEN q007 = 'De R$ 2.824,01 até R$ 3.530,00'        THEN 1 ELSE 0 END) AS renda_familiar_2824_3530,
    SUM(CASE WHEN q007 = 'De R$ 3.530,01 até R$ 4.236,00'        THEN 1 ELSE 0 END) AS renda_familiar_3530_4236,
    SUM(CASE WHEN q007 = 'De R$ 4.236,01 até R$ 5.648,00'        THEN 1 ELSE 0 END) AS renda_familiar_4236_5648,
    SUM(CASE WHEN q007 = 'De R$ 5.648,01 até R$ 7.060,00'        THEN 1 ELSE 0 END) AS renda_familiar_5648_7060,
    SUM(CASE WHEN q007 = 'De R$ 7.060,01 até R$ 8.472,00'        THEN 1 ELSE 0 END) AS renda_familiar_7060_8472,
    SUM(CASE WHEN q007 = 'De R$ 8.472,01 até R$ 9.884,00'        THEN 1 ELSE 0 END) AS renda_familiar_8472_9884,
    SUM(CASE WHEN q007 = 'De R$ 9.884,01 até R$ 11.296,00'       THEN 1 ELSE 0 END) AS renda_familiar_9884_11296,
    SUM(CASE WHEN q007 = 'De R$ 11.296,01 até R$ 12.708,00'      THEN 1 ELSE 0 END) AS renda_familiar_11296_12708,
    SUM(CASE WHEN q007 = 'De R$ 12.708,01 até R$ 14.120,00'      THEN 1 ELSE 0 END) AS renda_familiar_12708_14120,
    SUM(CASE WHEN q007 = 'De R$ 14.120,01 até R$ 16.944,00'      THEN 1 ELSE 0 END) AS renda_familiar_14120_16944,
    SUM(CASE WHEN q007 = 'De R$ 16.944,01 até R$ 21.180,00'      THEN 1 ELSE 0 END) AS renda_familiar_16944_21180,
    SUM(CASE WHEN q007 = 'De R$ 21.180,01 até R$ 28.240,00'      THEN 1 ELSE 0 END) AS renda_familiar_21180_28240,
    SUM(CASE WHEN q007 = 'Acima de R$ 28.240,00'                 THEN 1 ELSE 0 END) AS renda_familiar_classe_a,

    -- [QUESTIONÁRIO - BENS E TECNOLOGIA]
    SUM(CASE WHEN q008 LIKE 'Sim%'                                             THEN 1 ELSE 0 END) AS empregado_domestico_sim,
    SUM(CASE WHEN q009 = 'Não'           THEN 1 ELSE 0 END) AS banheiro_nao,
    SUM(CASE WHEN q009 LIKE 'Sim, um'    THEN 1 ELSE 0 END) AS banheiro_1,
    SUM(CASE WHEN q009 LIKE 'Sim, dois'  THEN 1 ELSE 0 END) AS banheiro_2,
    SUM(CASE WHEN q009 LIKE 'Sim, três%' THEN 1 ELSE 0 END) AS banheiro_3_ou_mais,
    SUM(CASE WHEN q010 = 'Não'           THEN 1 ELSE 0 END) AS quarto_nao,
    SUM(CASE WHEN q010 LIKE 'Sim, um'    THEN 1 ELSE 0 END) AS quarto_1,
    SUM(CASE WHEN q010 LIKE 'Sim, dois'  THEN 1 ELSE 0 END) AS quarto_2,
    SUM(CASE WHEN q010 LIKE 'Sim, três%' THEN 1 ELSE 0 END) AS quarto_3_ou_mais,
    SUM(CASE WHEN q011 = 'Não'           THEN 1 ELSE 0 END) AS carro_nao,
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
    SUM(CASE WHEN q022 = 'Não'           THEN 1 ELSE 0 END) AS celular_nao,
    SUM(CASE WHEN q022 LIKE 'Sim, um'    THEN 1 ELSE 0 END) AS celular_1,
    SUM(CASE WHEN q022 LIKE 'Sim, dois'  THEN 1 ELSE 0 END) AS celular_2,
    SUM(CASE WHEN q022 LIKE 'Sim, três%' OR q022 LIKE 'Sim, quatro%'          THEN 1 ELSE 0 END) AS celular_3_ou_mais,

    -- [QUESTIONÁRIO - ESCOLA]
    SUM(CASE WHEN q023 = 'Não frequentei escola de Ensino Médio'              THEN 1 ELSE 0 END) AS tipo_escola_nao_frequentou,
    SUM(CASE WHEN q023 LIKE 'Parte em escola pública e parte em escol%'       THEN 1 ELSE 0 END) AS tipo_escola_mista,
    SUM(CASE WHEN q023 LIKE 'Somente em escola privada com bolsa de e%'       THEN 1 ELSE 0 END) AS tipo_escola_privada_bolsa,
    SUM(CASE WHEN q023 LIKE 'Somente em escola privada sem bolsa de e%'       THEN 1 ELSE 0 END) AS tipo_escola_privada_sem_bolsa,
    SUM(CASE WHEN q023 = 'Somente em escola pública'                          THEN 1 ELSE 0 END) AS tipo_escola_publica,
    SUM(CASE WHEN q023 LIKE 'Somente em escola privada%'                      THEN 1 ELSE 0 END) AS tipo_escola_privada

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
# API pública
# ---------------------------------------------------------------------------

def load_data(db_config: dict[str, Any]) -> pd.DataFrame:
    """
    Carrega os dados do ENEM 2024 agregados ao nível municipal via PostgreSQL.

    Parameters
    ----------
    db_config : dict
        Credenciais do banco (lidas de st.secrets["database"] em app.py).
        Chaves esperadas: ``host``, ``port``, ``dbname``, ``user``, ``password``.

    Returns
    -------
    pd.DataFrame com uma linha por município.

    Raises
    ------
    ValueError       – se ``db_config`` for None ou vazio.
    ImportError      – se psycopg2-binary não estiver instalado.
    psycopg2.Error   – erros de conexão ou query.
    """
    if not db_config:
        raise ValueError(
            "Credenciais do banco de dados não configuradas. "
            "Configure as variáveis em Settings → Secrets no Streamlit Cloud."
        )
    return _pg_load(db_config)


# ---------------------------------------------------------------------------
# Statistical sampling – SQL templates (individual-level, not aggregated)
# ---------------------------------------------------------------------------
# ed_enem_2024_resultados has one row per participant with individual scores.
# ed_enem_2024_participantes has the demographic columns (cor_raca, etc.) but
# shares no individual-level join key with resultados – both tables are linked
# only at the co_municipio_prova (municipal) level, exactly like the
# aggregated pipeline.  All sampling queries therefore use resultados alone.
# nota_media_5_notas is a pre-computed column in ed_enem_2024_resultados.

_QUERY_MOMENTOS = """
SELECT
    COUNT(*)                               AS pop_n,
    AVG(nota_media_5_notas)                AS pop_media,
    VARIANCE(nota_media_5_notas)           AS pop_variancia,
    STDDEV(nota_media_5_notas)             AS pop_desvio_padrao,
    MIN(nota_media_5_notas)                AS pop_minimo,
    MAX(nota_media_5_notas)                AS pop_maximo
FROM ed_enem_2024_resultados
WHERE nota_redacao       IS NOT NULL
  AND nota_media_5_notas IS NOT NULL
"""

# {n} is substituted as a plain integer by load_sampling_data().
_QUERY_AAS = """
SELECT
    r.nota_media_5_notas              AS nota_media,
    r.nota_cn_ciencias_da_natureza    AS nota_cn,
    r.nota_ch_ciencias_humanas        AS nota_ch,
    r.nota_lc_linguagens_e_codigos    AS nota_lc,
    r.nota_mt_matematica              AS nota_mt,
    r.nota_redacao,
    r.co_municipio_prova              AS municipio,
    m.cd_uf                           AS uf
FROM ed_enem_2024_resultados r
JOIN municipio m ON m.codigo_municipio_dv = r.co_municipio_prova
WHERE r.nota_redacao       IS NOT NULL
  AND r.nota_media_5_notas IS NOT NULL
ORDER BY RANDOM()
LIMIT {n}
"""

# {pop_n} and {n} are substituted as plain integers.
# Stratification is by UF (cd_uf in the municipio reference table), obtained
# via a JOIN since resultados only stores co_municipio_prova.
_QUERY_ESTRATIFICADA = """
WITH ranked AS (
    SELECT
        r.nota_media_5_notas              AS nota_media,
        r.nota_cn_ciencias_da_natureza    AS nota_cn,
        r.nota_ch_ciencias_humanas        AS nota_ch,
        r.nota_lc_linguagens_e_codigos    AS nota_lc,
        r.nota_mt_matematica              AS nota_mt,
        r.nota_redacao,
        r.co_municipio_prova              AS municipio,
        m.cd_uf                           AS uf,
        COUNT(*) OVER (
            PARTITION BY m.cd_uf
        )                                 AS tamanho_estrato,
        ROW_NUMBER() OVER (
            PARTITION BY m.cd_uf
            ORDER BY RANDOM()
        )                                 AS rn
    FROM ed_enem_2024_resultados r
    JOIN municipio m ON m.codigo_municipio_dv = r.co_municipio_prova
    WHERE r.nota_redacao       IS NOT NULL
      AND r.nota_media_5_notas IS NOT NULL
)
SELECT nota_media, nota_cn, nota_ch, nota_lc, nota_mt, nota_redacao,
       municipio, uf
FROM ranked
WHERE rn <= GREATEST(1, ROUND((tamanho_estrato::float / {pop_n}) * {n}))
"""

# {k} is the systematic interval (N // n); {n} caps the result.
_QUERY_SISTEMATICA = """
WITH numbered AS (
    SELECT
        r.nota_media_5_notas              AS nota_media,
        r.nota_cn_ciencias_da_natureza    AS nota_cn,
        r.nota_ch_ciencias_humanas        AS nota_ch,
        r.nota_lc_linguagens_e_codigos    AS nota_lc,
        r.nota_mt_matematica              AS nota_mt,
        r.nota_redacao,
        r.co_municipio_prova              AS municipio,
        m.cd_uf                           AS uf,
        ROW_NUMBER() OVER (
            ORDER BY r.co_municipio_prova
        )                                 AS rn
    FROM ed_enem_2024_resultados r
    JOIN municipio m ON m.codigo_municipio_dv = r.co_municipio_prova
    WHERE r.nota_redacao       IS NOT NULL
      AND r.nota_media_5_notas IS NOT NULL
)
SELECT nota_media, nota_cn, nota_ch, nota_lc, nota_mt, nota_redacao,
       municipio, uf
FROM numbered
WHERE MOD(rn, {k}::bigint) = 0
LIMIT {n}
"""


# ---------------------------------------------------------------------------
# Sampling – participantes table (all 4.3 M registered participants)
# ---------------------------------------------------------------------------
# Variable for Cochran formula: idade_calculada (age in years).
# Stratification: tp_cor_raca (race/color).
# E = 0.5 years (half-year acceptable error for mean-age estimation).

_QUERY_MOMENTOS_PART = """
SELECT
    COUNT(*)                        AS pop_n,
    AVG(idade_calculada)            AS pop_media,
    VARIANCE(idade_calculada)       AS pop_variancia,
    STDDEV(idade_calculada)         AS pop_desvio_padrao,
    MIN(idade_calculada)            AS pop_minimo,
    MAX(idade_calculada)            AS pop_maximo
FROM ed_enem_2024_participantes
"""

_QUERY_COR_RACA_POP = """
SELECT
    COALESCE(tp_cor_raca, 'Não declarado') AS cor_raca,
    COUNT(*) AS pop_count
FROM ed_enem_2024_participantes
GROUP BY tp_cor_raca
ORDER BY pop_count DESC
"""

_QUERY_SEXO_POP = """
SELECT
    COALESCE(tp_sexo, 'Não informado') AS sexo,
    COUNT(*) AS pop_count
FROM ed_enem_2024_participantes
GROUP BY tp_sexo
ORDER BY pop_count DESC
"""

# {n} substituted as plain integer.
_QUERY_AAS_PART = """
SELECT
    idade_calculada     AS idade,
    tp_cor_raca         AS cor_raca,
    tp_sexo             AS sexo,
    co_municipio_prova  AS municipio
FROM ed_enem_2024_participantes
ORDER BY RANDOM()
LIMIT {n}
"""

# {pop_n} and {n} substituted as plain integers.
# Stratification by tp_cor_raca (race/color).
_QUERY_ESTRATIFICADA_PART = """
WITH ranked AS (
    SELECT
        idade_calculada     AS idade,
        tp_cor_raca         AS cor_raca,
        tp_sexo             AS sexo,
        co_municipio_prova  AS municipio,
        COUNT(*) OVER (
            PARTITION BY tp_cor_raca
        )                   AS tamanho_estrato,
        ROW_NUMBER() OVER (
            PARTITION BY tp_cor_raca
            ORDER BY RANDOM()
        )                   AS rn
    FROM ed_enem_2024_participantes
)
SELECT idade, cor_raca, sexo, municipio
FROM ranked
WHERE rn <= GREATEST(1, ROUND((tamanho_estrato::float / {pop_n}) * {n}))
"""

# {k} systematic interval; {n} caps the result.
_QUERY_SISTEMATICA_PART = """
WITH numbered AS (
    SELECT
        idade_calculada     AS idade,
        tp_cor_raca         AS cor_raca,
        tp_sexo             AS sexo,
        co_municipio_prova  AS municipio,
        ROW_NUMBER() OVER (
            ORDER BY co_municipio_prova
        )                   AS rn
    FROM ed_enem_2024_participantes
)
SELECT idade, cor_raca, sexo, municipio
FROM numbered
WHERE MOD(rn, {k}::bigint) = 0
LIMIT {n}
"""


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _calcular_n_amostra(
    pop_n: int,
    sigma: float,
    Z: float = 1.96,
    E: float = 3.0,
) -> int:
    """
    Calcula o tamanho mínimo da amostra (Cochran) com correção para população finita.

    n₀ = (Z² × σ²) / E²
    n  = n₀ / (1 + (n₀ − 1) / N)
    """
    n0 = (Z ** 2 * sigma ** 2) / (E ** 2)
    n = n0 / (1.0 + (n0 - 1.0) / pop_n)
    return max(1, int(math.ceil(n)))


def load_sampling_data(db_config: dict[str, Any]) -> dict:
    """
    Coleta momentos populacionais e gera três amostras estatísticas para
    **duas** tabelas via PostgreSQL (individual-level, não agregado):

    * ``ed_enem_2024_participantes`` – todos os inscritos (N ≈ 4,3 M).
      Variável de Cochran: ``idade_calculada``  (E = 0,5 ano, 95 % confiança).
      Estratificação: ``tp_cor_raca``.

    * ``ed_enem_2024_resultados`` – participantes com as 5 notas válidas (N ≈ 3 M).
      Variável de Cochran: ``nota_media_5_notas`` (E = 3 pontos, 95 % confiança).
      Estratificação: ``cd_uf`` (via JOIN com a tabela ``municipio``).

    Returns
    -------
    dict com chaves:

    Resultados:
        momentos_res, n_amostra_res, k_res, aas_res, estratificada_res, sistematica_res

    Participantes:
        momentos_part, n_amostra_part, k_part, aas_part, estratificada_part, sistematica_part

    Compartilhados:
        Z, E_res, E_part, cor_raca_pop (DataFrame), sexo_pop (DataFrame)

    Raises
    ------
    ValueError    – credenciais não configuradas
    ImportError   – psycopg2-binary não instalado
    psycopg2.Error – erro de conexão ou query
    """
    if not db_config:
        raise ValueError(
            "Credenciais do banco de dados não configuradas. "
            "Configure as variáveis em Settings → Secrets no Streamlit Cloud."
        )
    try:
        import psycopg2  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "psycopg2-binary não está instalado. "
            "Adicione-o a requirements.txt para usar o banco PostgreSQL."
        ) from exc

    Z     = 1.96
    E_res  = 3.0   # acceptable error in score points
    E_part = 0.5   # acceptable error in age years

    conn = psycopg2.connect(
        host=db_config["host"],
        port=int(db_config.get("port", 5432)),
        dbname=db_config["dbname"],
        user=db_config["user"],
        password=db_config["password"],
    )
    try:
        # ---- RESULTADOS ----
        row_res = pd.read_sql(_QUERY_MOMENTOS, conn).iloc[0]
        pop_n_res = int(row_res["pop_n"])
        momentos_res = {
            "N":             pop_n_res,
            "media":         float(row_res["pop_media"]),
            "variancia":     float(row_res["pop_variancia"]),
            "desvio_padrao": float(row_res["pop_desvio_padrao"]),
            "minimo":        float(row_res["pop_minimo"]),
            "maximo":        float(row_res["pop_maximo"]),
        }
        n_res = _calcular_n_amostra(pop_n_res, momentos_res["desvio_padrao"], Z, E_res)
        k_res = max(1, pop_n_res // n_res)

        df_aas_res = pd.read_sql(_QUERY_AAS.format(n=n_res), conn)
        df_est_res = pd.read_sql(_QUERY_ESTRATIFICADA.format(pop_n=pop_n_res, n=n_res), conn)
        df_sis_res = pd.read_sql(_QUERY_SISTEMATICA.format(k=k_res, n=n_res), conn)

        # ---- PARTICIPANTES ----
        row_part = pd.read_sql(_QUERY_MOMENTOS_PART, conn).iloc[0]
        pop_n_part = int(row_part["pop_n"])
        momentos_part = {
            "N":             pop_n_part,
            "media":         float(row_part["pop_media"]),
            "variancia":     float(row_part["pop_variancia"]),
            "desvio_padrao": float(row_part["pop_desvio_padrao"]),
            "minimo":        float(row_part["pop_minimo"]),
            "maximo":        float(row_part["pop_maximo"]),
        }
        n_part = _calcular_n_amostra(pop_n_part, momentos_part["desvio_padrao"], Z, E_part)
        k_part = max(1, pop_n_part // n_part)

        df_cor_raca_pop = pd.read_sql(_QUERY_COR_RACA_POP, conn)
        df_sexo_pop     = pd.read_sql(_QUERY_SEXO_POP, conn)

        df_aas_part = pd.read_sql(_QUERY_AAS_PART.format(n=n_part), conn)
        df_est_part = pd.read_sql(
            _QUERY_ESTRATIFICADA_PART.format(pop_n=pop_n_part, n=n_part), conn
        )
        df_sis_part = pd.read_sql(_QUERY_SISTEMATICA_PART.format(k=k_part, n=n_part), conn)
    finally:
        conn.close()

    return {
        # Resultados (participants with all 5 valid scores)
        "momentos_res":      momentos_res,
        "n_amostra_res":     n_res,
        "k_res":             k_res,
        "aas_res":           df_aas_res,
        "estratificada_res": df_est_res,
        "sistematica_res":   df_sis_res,
        # Participantes (all registered participants)
        "momentos_part":      momentos_part,
        "n_amostra_part":     n_part,
        "k_part":             k_part,
        "aas_part":           df_aas_part,
        "estratificada_part": df_est_part,
        "sistematica_part":   df_sis_part,
        # Shared parameters and population class distributions
        "Z":            Z,
        "E_res":        E_res,
        "E_part":       E_part,
        "cor_raca_pop": df_cor_raca_pop,
        "sexo_pop":     df_sexo_pop,
    }


