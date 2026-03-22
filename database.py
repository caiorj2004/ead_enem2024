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

