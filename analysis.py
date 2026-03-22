"""
analysis.py – Operações estatísticas e de análise dos dados do ENEM 2024.

O DataFrame de entrada tem uma linha por município (estrutura produzida
pelo pipeline de agregação em database.py) com as seguintes colunas:

  Identificação:   cod_7, municipio, uf
  Participantes:   total_inscritos, media_idade
    Demografics:   sexo_feminino, sexo_masculino
                   raca_branca, raca_preta, raca_parda, raca_amarela, raca_indigena
                   est_civil_solteiro, est_civil_casado
                   nacionalidade_brasileiro, treineiro_sim
    Q001-Q004:     escolaridade_pai/mae_nunca, _fund1_inc, _fund1, _fund2, _medio, _superior, _pos, _nao_sei
                   ocupacao_pai/mae_grupo1..5, _nao_sei
    Q006-Q007:     possui_renda_sim, renda_familiar_nenhuma, _ate_1412, _1412_2118, _2118_2824,
                   _2824_3530, _3530_4236, _4236_5648, _5648_7060, _7060_8472, _8472_9884,
                   _9884_11296, _11296_12708, _12708_14120, _14120_16944, _16944_21180,
                   _21180_28240, renda_familiar_classe_a
    Q008-Q022:     empregado_domestico_sim, banheiro_nao/1/2/3_ou_mais,
                   quarto_nao/1/2/3_ou_mais, carro_nao/1/2_ou_mais, motocicleta_sim,
                   geladeira_sim, freezer_sim, maquina_lavar_sim, micro_ondas_sim,
                   aspirador_po_sim, tv_sim, tv_assinatura_sim, internet_wifi_sim,
                   computador_1/2_ou_mais, celular_nao/1/2/3_ou_mais
    Q023:          tipo_escola_nao_frequentou, tipo_escola_mista,
                   tipo_escola_privada_bolsa, tipo_escola_privada_sem_bolsa,
                   tipo_escola_publica, tipo_escola_privada
  Resultados:      nota_ciencias_natureza, nota_ciencias_humanas, nota_linguagens,
                   nota_matematica, nota_redacao, nota_geral_media

Após apply_labels() são adicionadas colunas derivadas pct_* (proporções).

Funções exportadas
------------------
  apply_labels        – mapeia uf numérico para sigla e calcula pct_* columns
  frequency_table     – tabela de distribuição de frequência para variável categórica
  inscribed_by_uf     – agrega inscritos por UF
  descriptive_stats   – estatísticas descritivas de colunas numéricas
  correlation_matrix  – matriz de correlação (Pearson/Spearman/Kendall)
  mean_scores_by_group – médias ponderadas de uma nota agrupadas por UF
  normality_test      – Shapiro-Wilk
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Mapeamento IBGE código numérico de UF → sigla
# ---------------------------------------------------------------------------

IBGE_UF_MAP: dict[int, str] = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA",
    16: "AP", 17: "TO", 21: "MA", 22: "PI", 23: "CE",
    24: "RN", 25: "PB", 26: "PE", 27: "AL", 28: "SE",
    29: "BA", 31: "MG", 32: "ES", 33: "RJ", 35: "SP",
    41: "PR", 42: "SC", 43: "RS", 50: "MS", 51: "MT",
    52: "GO", 53: "DF",
}

# ---------------------------------------------------------------------------
# Variáveis qualitativas (ao nível municipal)
# ---------------------------------------------------------------------------

QUALITATIVE_VARS: list[dict] = [
    {"col": "uf", "title": "UF (Estado)"},
]

# ---------------------------------------------------------------------------
# Variáveis quantitativas – notas médias municipais
# ---------------------------------------------------------------------------

SCORE_COLS: list[str] = [
    "nota_ciencias_natureza",
    "nota_ciencias_humanas",
    "nota_linguagens",
    "nota_matematica",
    "nota_redacao",
    "nota_geral_media",
]

SCORE_LABELS: dict[str, str] = {
    "nota_ciencias_natureza": "Ciências da Natureza",
    "nota_ciencias_humanas":  "Ciências Humanas",
    "nota_linguagens":        "Linguagens e Códigos",
    "nota_matematica":        "Matemática",
    "nota_redacao":           "Redação",
    "nota_geral_media":       "Média Geral",
}

# ---------------------------------------------------------------------------
# Colunas de proporção (derivadas em apply_labels)
# ---------------------------------------------------------------------------

# Mapeamento: coluna_pct → coluna_contagem_origem
PROPORTION_MAP: dict[str, str] = {
    # Demografics – Sexo
    "pct_feminino":              "sexo_feminino",
    "pct_masculino":             "sexo_masculino",
    # Demografics – Raça/Cor
    "pct_branca":                "raca_branca",
    "pct_preta":                 "raca_preta",
    "pct_parda":                 "raca_parda",
    "pct_amarela":               "raca_amarela",
    "pct_indigena":              "raca_indigena",
    "pct_raca_nao_declarado":    "raca_nao_declarado",
    # Demografics – Estado Civil
    "pct_solteiro":              "est_civil_solteiro",
    "pct_casado":                "est_civil_casado",
    "pct_divorciado":            "est_civil_divorciado",
    "pct_viuvo":                 "est_civil_viuvo",
    "pct_est_civil_nao_info":    "est_civil_nao_informado",
    # Demografics – Perfil / Treineiro / Conclusão
    "pct_brasileiro":            "nacionalidade_brasileiro",
    "pct_nato_exterior":         "nacionalidade_nato_exterior",
    "pct_naturalizado":          "nacionalidade_naturalizado",
    "pct_estrangeiro":           "nacionalidade_estrangeiro",
    "pct_nac_nao_informado":     "nacionalidade_nao_informado",
    "pct_treineiro":             "treineiro_sim",
    "pct_nao_treineiro":         "treineiro_nao",
    "pct_conclusao_concluiu":    "conclusao_ja_concluiu",
    "pct_conclusao_em_2024":     "conclusao_cursando_2024",
    "pct_conclusao_apos_2024":   "conclusao_cursando_apos",
    "pct_conclusao_nao_concluiu": "conclusao_nao_concluiu",
    # Q001-Q004 Escolaridade e Ocupação
    "pct_escolaridade_pai_nunca":     "escolaridade_pai_nunca",
    "pct_escolaridade_pai_fund1_inc": "escolaridade_pai_fund1_inc",
    "pct_escolaridade_pai_fund1":     "escolaridade_pai_fund1",
    "pct_escolaridade_pai_fund2":     "escolaridade_pai_fund2",
    "pct_escolaridade_pai_medio":     "escolaridade_pai_medio",
    "pct_escolaridade_pai_superior":  "escolaridade_pai_superior",
    "pct_escolaridade_pai_pos":       "escolaridade_pai_pos",
    "pct_escolaridade_pai_nao_sei":   "escolaridade_pai_nao_sei",
    "pct_escolaridade_mae_nunca":     "escolaridade_mae_nunca",
    "pct_escolaridade_mae_fund1_inc": "escolaridade_mae_fund1_inc",
    "pct_escolaridade_mae_fund1":     "escolaridade_mae_fund1",
    "pct_escolaridade_mae_fund2":     "escolaridade_mae_fund2",
    "pct_escolaridade_mae_medio":     "escolaridade_mae_medio",
    "pct_escolaridade_mae_superior":  "escolaridade_mae_superior",
    "pct_escolaridade_mae_pos":       "escolaridade_mae_pos",
    "pct_escolaridade_mae_nao_sei":   "escolaridade_mae_nao_sei",
    "pct_ocupacao_pai_grupo1":        "ocupacao_pai_grupo1",
    "pct_ocupacao_pai_grupo2":        "ocupacao_pai_grupo2",
    "pct_ocupacao_pai_grupo3":        "ocupacao_pai_grupo3",
    "pct_ocupacao_pai_grupo4":        "ocupacao_pai_grupo4",
    "pct_ocupacao_pai_grupo5":        "ocupacao_pai_grupo5",
    "pct_ocupacao_pai_nao_sei":       "ocupacao_pai_nao_sei",
    "pct_ocupacao_mae_grupo1":        "ocupacao_mae_grupo1",
    "pct_ocupacao_mae_grupo2":        "ocupacao_mae_grupo2",
    "pct_ocupacao_mae_grupo3":        "ocupacao_mae_grupo3",
    "pct_ocupacao_mae_grupo4":        "ocupacao_mae_grupo4",
    "pct_ocupacao_mae_grupo5":        "ocupacao_mae_grupo5",
    "pct_ocupacao_mae_nao_sei":       "ocupacao_mae_nao_sei",
    # Q006-Q007 Renda
    "pct_possui_renda":               "possui_renda_sim",
    "pct_renda_nenhuma":              "renda_familiar_nenhuma",
    "pct_renda_ate_1412":             "renda_familiar_ate_1412",
    "pct_renda_1412_2118":            "renda_familiar_1412_2118",
    "pct_renda_2118_2824":            "renda_familiar_2118_2824",
    "pct_renda_2824_3530":            "renda_familiar_2824_3530",
    "pct_renda_3530_4236":            "renda_familiar_3530_4236",
    "pct_renda_4236_5648":            "renda_familiar_4236_5648",
    "pct_renda_5648_7060":            "renda_familiar_5648_7060",
    "pct_renda_7060_8472":            "renda_familiar_7060_8472",
    "pct_renda_8472_9884":            "renda_familiar_8472_9884",
    "pct_renda_9884_11296":           "renda_familiar_9884_11296",
    "pct_renda_11296_12708":          "renda_familiar_11296_12708",
    "pct_renda_12708_14120":          "renda_familiar_12708_14120",
    "pct_renda_14120_16944":          "renda_familiar_14120_16944",
    "pct_renda_16944_21180":          "renda_familiar_16944_21180",
    "pct_renda_21180_28240":          "renda_familiar_21180_28240",
    "pct_renda_classe_a":             "renda_familiar_classe_a",
    # Q008-Q022 Bens e Tecnologia
    "pct_empregado_domestico":   "empregado_domestico_sim",
    "pct_banheiro_nao":          "banheiro_nao",
    "pct_banheiro_1":            "banheiro_1",
    "pct_banheiro_2":            "banheiro_2",
    "pct_banheiro_3_mais":       "banheiro_3_ou_mais",
    "pct_quarto_nao":            "quarto_nao",
    "pct_quarto_1":              "quarto_1",
    "pct_quarto_2":              "quarto_2",
    "pct_quarto_3_mais":         "quarto_3_ou_mais",
    "pct_carro_nao":             "carro_nao",
    "pct_carro_1":               "carro_1",
    "pct_carro_2_mais":          "carro_2_ou_mais",
    "pct_motocicleta":           "motocicleta_sim",
    "pct_geladeira":             "geladeira_sim",
    "pct_freezer":               "freezer_sim",
    "pct_maquina_lavar":         "maquina_lavar_sim",
    "pct_micro_ondas":           "micro_ondas_sim",
    "pct_aspirador":             "aspirador_po_sim",
    "pct_tv":                    "tv_sim",
    "pct_tv_assinatura":         "tv_assinatura_sim",
    "pct_internet":              "internet_wifi_sim",
    "pct_computador_1":          "computador_1",
    "pct_computador_2_mais":     "computador_2_ou_mais",
    "pct_celular_nao":           "celular_nao",
    "pct_celular_1":             "celular_1",
    "pct_celular_2":             "celular_2",
    "pct_celular_3_mais":        "celular_3_ou_mais",
    # Q023 Escola
    "pct_escola_nao_frequentou":      "tipo_escola_nao_frequentou",
    "pct_escola_mista":               "tipo_escola_mista",
    "pct_escola_privada_bolsa":       "tipo_escola_privada_bolsa",
    "pct_escola_privada_sem_bolsa":   "tipo_escola_privada_sem_bolsa",
    "pct_escola_publica":             "tipo_escola_publica",
    "pct_escola_privada":             "tipo_escola_privada",
}

PROPORTION_LABELS: dict[str, str] = {
    # Sexo
    "pct_feminino":              "% Feminino",
    "pct_masculino":             "% Masculino",
    # Raça/Cor
    "pct_branca":                "% Branca",
    "pct_preta":                 "% Preta",
    "pct_parda":                 "% Parda",
    "pct_amarela":               "% Amarela",
    "pct_indigena":              "% Indígena",
    "pct_raca_nao_declarado":    "% Não declarado",
    # Estado Civil
    "pct_solteiro":              "% Solteiro(a)",
    "pct_casado":                "% Casado(a)/Companheiro(a)",
    "pct_divorciado":            "% Divorciado(a)/Separado(a)",
    "pct_viuvo":                 "% Viúvo(a)",
    "pct_est_civil_nao_info":    "% Est. Civil Não informado",
    # Perfil / Treineiro / Conclusão
    "pct_brasileiro":            "% Brasileiro(a)",
    "pct_nato_exterior":         "% Brasileiro(a) Nato – exterior",
    "pct_naturalizado":          "% Naturalizado(a)",
    "pct_estrangeiro":           "% Estrangeiro(a)",
    "pct_nac_nao_informado":     "% Nac. Não informado",
    "pct_treineiro":             "% Treineiro",
    "pct_nao_treineiro":         "% Não Treineiro",
    "pct_conclusao_concluiu":    "% Já concluiu EM",
    "pct_conclusao_em_2024":     "% Cursando – conclui em 2024",
    "pct_conclusao_apos_2024":   "% Cursando – conclui após 2024",
    "pct_conclusao_nao_concluiu": "% Não concluiu EM",
    # Q001-Q004 Escolaridade e Ocupação
    "pct_escolaridade_pai_nunca":     "% Pai – Nunca estudou",
    "pct_escolaridade_pai_fund1_inc": "% Pai – Não completou 4ª série/5º ano EF",
    "pct_escolaridade_pai_fund1":     "% Pai – Completou 4ª–5º ano EF",
    "pct_escolaridade_pai_fund2":     "% Pai – Completou 8ª–9º ano EF",
    "pct_escolaridade_pai_medio":     "% Pai – Completou Ensino Médio",
    "pct_escolaridade_pai_superior":  "% Pai – Completou Faculdade",
    "pct_escolaridade_pai_pos":       "% Pai – Pós-graduado",
    "pct_escolaridade_pai_nao_sei":   "% Pai – Não sei",
    "pct_escolaridade_mae_nunca":     "% Mãe – Nunca estudou",
    "pct_escolaridade_mae_fund1_inc": "% Mãe – Não completou 4ª série/5º ano EF",
    "pct_escolaridade_mae_fund1":     "% Mãe – Completou 4ª–5º ano EF",
    "pct_escolaridade_mae_fund2":     "% Mãe – Completou 8ª–9º ano EF",
    "pct_escolaridade_mae_medio":     "% Mãe – Completou Ensino Médio",
    "pct_escolaridade_mae_superior":  "% Mãe – Completou Faculdade",
    "pct_escolaridade_mae_pos":       "% Mãe – Pós-graduada",
    "pct_escolaridade_mae_nao_sei":   "% Mãe – Não sei",
    "pct_ocupacao_pai_grupo1":        "% Pai – Grupo 1 (agric./extrat.)",
    "pct_ocupacao_pai_grupo2":        "% Pai – Grupo 2 (serv. gerais)",
    "pct_ocupacao_pai_grupo3":        "% Pai – Grupo 3 (ofícios/indústria)",
    "pct_ocupacao_pai_grupo4":        "% Pai – Grupo 4 (técn./comércio)",
    "pct_ocupacao_pai_grupo5":        "% Pai – Grupo 5 (alta qualif.)",
    "pct_ocupacao_pai_nao_sei":       "% Pai – Ocupação Não sei",
    "pct_ocupacao_mae_grupo1":        "% Mãe – Grupo 1 (agric./extrat.)",
    "pct_ocupacao_mae_grupo2":        "% Mãe – Grupo 2 (serv. gerais)",
    "pct_ocupacao_mae_grupo3":        "% Mãe – Grupo 3 (ofícios/indústria)",
    "pct_ocupacao_mae_grupo4":        "% Mãe – Grupo 4 (técn./comércio)",
    "pct_ocupacao_mae_grupo5":        "% Mãe – Grupo 5 (alta qualif.)",
    "pct_ocupacao_mae_nao_sei":       "% Mãe – Ocupação Não sei",
    # Q006-Q007 Renda
    "pct_possui_renda":               "% com Renda",
    "pct_renda_nenhuma":              "% Renda Nenhuma",
    "pct_renda_ate_1412":             "% Renda até R$1.412",
    "pct_renda_1412_2118":            "% Renda R$1.412–2.118",
    "pct_renda_2118_2824":            "% Renda R$2.118–2.824",
    "pct_renda_2824_3530":            "% Renda R$2.824–3.530",
    "pct_renda_3530_4236":            "% Renda R$3.530–4.236",
    "pct_renda_4236_5648":            "% Renda R$4.236–5.648",
    "pct_renda_5648_7060":            "% Renda R$5.648–7.060",
    "pct_renda_7060_8472":            "% Renda R$7.060–8.472",
    "pct_renda_8472_9884":            "% Renda R$8.472–9.884",
    "pct_renda_9884_11296":           "% Renda R$9.884–11.296",
    "pct_renda_11296_12708":          "% Renda R$11.296–12.708",
    "pct_renda_12708_14120":          "% Renda R$12.708–14.120",
    "pct_renda_14120_16944":          "% Renda R$14.120–16.944",
    "pct_renda_16944_21180":          "% Renda R$16.944–21.180",
    "pct_renda_21180_28240":          "% Renda R$21.180–28.240",
    "pct_renda_classe_a":             "% Classe A (>R$28.240)",
    # Q008-Q022 Bens e Tecnologia
    "pct_empregado_domestico":   "% Empregado Doméstico",
    "pct_banheiro_nao":          "% Sem Banheiro",
    "pct_banheiro_1":            "% 1 Banheiro",
    "pct_banheiro_2":            "% 2 Banheiros",
    "pct_banheiro_3_mais":       "% 3+ Banheiros",
    "pct_quarto_nao":            "% Sem Quarto",
    "pct_quarto_1":              "% 1 Quarto",
    "pct_quarto_2":              "% 2 Quartos",
    "pct_quarto_3_mais":         "% 3+ Quartos",
    "pct_carro_nao":             "% Sem Carro",
    "pct_carro_1":               "% 1 Carro",
    "pct_carro_2_mais":          "% 2+ Carros",
    "pct_motocicleta":           "% Motocicleta",
    "pct_geladeira":             "% Geladeira",
    "pct_freezer":               "% Freezer",
    "pct_maquina_lavar":         "% Máquina de Lavar",
    "pct_micro_ondas":           "% Micro-ondas",
    "pct_aspirador":             "% Aspirador de Pó",
    "pct_tv":                    "% TV",
    "pct_tv_assinatura":         "% TV por Assinatura",
    "pct_internet":              "% Internet (Wi-Fi)",
    "pct_computador_1":               "% 1 Computador",
    "pct_computador_2_mais":          "% 2+ Computadores",
    "pct_celular_nao":                "% Sem Celular",
    "pct_celular_1":                  "% 1 Celular",
    "pct_celular_2":                  "% 2 Celulares",
    "pct_celular_3_mais":             "% 3+ Celulares",
    # Q023 Escola
    "pct_escola_nao_frequentou":      "% Não frequentou EM",
    "pct_escola_mista":               "% Escola Mista (pública/privada)",
    "pct_escola_privada_bolsa":       "% Escola Privada (com bolsa)",
    "pct_escola_privada_sem_bolsa":   "% Escola Privada (sem bolsa)",
    "pct_escola_publica":             "% Escola Pública",
    "pct_escola_privada":             "% Escola Privada (total)",
}

PROPORTION_COLS: list[str] = list(PROPORTION_MAP.keys())

# Group keys for UI organisation (used in app.py demographic selector)
PROPORTION_GROUPS: dict[str, list[str]] = {
    "Sexo": [
        "pct_feminino", "pct_masculino",
    ],
    "Raça/Cor": [
        "pct_branca", "pct_parda", "pct_preta", "pct_amarela",
        "pct_indigena", "pct_raca_nao_declarado",
    ],
    "Estado Civil": [
        "pct_solteiro", "pct_casado", "pct_divorciado",
        "pct_viuvo", "pct_est_civil_nao_info",
    ],
    "Perfil": [
        "pct_brasileiro", "pct_nato_exterior", "pct_naturalizado",
        "pct_estrangeiro", "pct_nac_nao_informado",
    ],
    "Treineiro": [
        "pct_treineiro", "pct_nao_treineiro",
    ],
    "Situação de Conclusão (Ensino Médio)": [
        "pct_conclusao_concluiu", "pct_conclusao_em_2024",
        "pct_conclusao_apos_2024", "pct_conclusao_nao_concluiu",
    ],
    "Escolaridade do Pai": [
        "pct_escolaridade_pai_nunca", "pct_escolaridade_pai_fund1_inc",
        "pct_escolaridade_pai_fund1", "pct_escolaridade_pai_fund2",
        "pct_escolaridade_pai_medio", "pct_escolaridade_pai_superior",
        "pct_escolaridade_pai_pos", "pct_escolaridade_pai_nao_sei",
    ],
    "Escolaridade da Mãe": [
        "pct_escolaridade_mae_nunca", "pct_escolaridade_mae_fund1_inc",
        "pct_escolaridade_mae_fund1", "pct_escolaridade_mae_fund2",
        "pct_escolaridade_mae_medio", "pct_escolaridade_mae_superior",
        "pct_escolaridade_mae_pos", "pct_escolaridade_mae_nao_sei",
    ],
    "Ocupação do Pai": [
        "pct_ocupacao_pai_grupo1", "pct_ocupacao_pai_grupo2",
        "pct_ocupacao_pai_grupo3", "pct_ocupacao_pai_grupo4",
        "pct_ocupacao_pai_grupo5", "pct_ocupacao_pai_nao_sei",
    ],
    "Ocupação da Mãe": [
        "pct_ocupacao_mae_grupo1", "pct_ocupacao_mae_grupo2",
        "pct_ocupacao_mae_grupo3", "pct_ocupacao_mae_grupo4",
        "pct_ocupacao_mae_grupo5", "pct_ocupacao_mae_nao_sei",
    ],
    "Renda Familiar": [
        "pct_possui_renda", "pct_renda_nenhuma",
        "pct_renda_ate_1412", "pct_renda_1412_2118", "pct_renda_2118_2824",
        "pct_renda_2824_3530", "pct_renda_3530_4236", "pct_renda_4236_5648",
        "pct_renda_5648_7060", "pct_renda_7060_8472", "pct_renda_8472_9884",
        "pct_renda_9884_11296", "pct_renda_11296_12708", "pct_renda_12708_14120",
        "pct_renda_14120_16944", "pct_renda_16944_21180", "pct_renda_21180_28240",
        "pct_renda_classe_a",
    ],
    "Bens do Domicílio": [
        "pct_geladeira", "pct_freezer", "pct_maquina_lavar", "pct_micro_ondas",
        "pct_aspirador", "pct_tv", "pct_tv_assinatura", "pct_empregado_domestico",
    ],
    "Moradia (Quartos e Banheiros)": [
        "pct_banheiro_nao", "pct_banheiro_1", "pct_banheiro_2", "pct_banheiro_3_mais",
        "pct_quarto_nao", "pct_quarto_1", "pct_quarto_2", "pct_quarto_3_mais",
    ],
    "Transporte": [
        "pct_carro_nao", "pct_carro_1", "pct_carro_2_mais", "pct_motocicleta",
    ],
    "Tecnologia e Conectividade": [
        "pct_internet", "pct_computador_1", "pct_computador_2_mais",
        "pct_celular_nao", "pct_celular_1", "pct_celular_2", "pct_celular_3_mais",
    ],
    "Tipo de Escola (EM)": [
        "pct_escola_nao_frequentou", "pct_escola_mista",
        "pct_escola_privada_bolsa", "pct_escola_privada_sem_bolsa",
        "pct_escola_publica", "pct_escola_privada",
    ],
}

# Columns shown in the heatmap (scores + sexo + raça/cor + escola)
HEATMAP_COLS: list[str] = [
    *SCORE_COLS,
    "pct_feminino", "pct_masculino",
    "pct_branca", "pct_parda", "pct_preta", "pct_amarela",
    "pct_indigena", "pct_raca_nao_declarado",
    "pct_escola_publica", "pct_escola_privada",
]


# ---------------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------------

def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna uma cópia do DataFrame com:
    1. Coluna `uf` convertida de código IBGE numérico para sigla (quando aplicável).
    2. Colunas `pct_*` com a proporção de cada grupo em relação a `total_inscritos`.
    """
    df = df.copy()

    # Converte código numérico de UF para sigla, se necessário
    if "uf" in df.columns:
        sample = df["uf"].dropna().iloc[0] if len(df) > 0 else None
        if sample is not None:
            try:
                numeric_val = int(float(str(sample)))
                if numeric_val in IBGE_UF_MAP:
                    df["uf"] = (
                        pd.to_numeric(df["uf"], errors="coerce")
                        .map(IBGE_UF_MAP)
                        .fillna(df["uf"])
                    )
            except (ValueError, TypeError):
                pass  # already a string abbreviation

    # Calcula proporções (evita divisão por zero)
    if "total_inscritos" in df.columns:
        total = df["total_inscritos"].replace(0, np.nan)
        for pct_col, count_col in PROPORTION_MAP.items():
            if count_col in df.columns:
                df[pct_col] = (df[count_col] / total * 100).round(2)

    return df


# ---------------------------------------------------------------------------
# Tabelas de frequência
# ---------------------------------------------------------------------------

def frequency_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Tabela de distribuição de frequência para uma variável categórica.

    Returns
    -------
    pd.DataFrame com colunas:
        Categoria | Freq. Absoluta | Freq. Relativa (%) |
        Freq. Absoluta Acumulada | Freq. Relativa Acumulada (%)
    """
    counts = df[column].value_counts(dropna=False)
    total  = counts.sum()

    result = pd.DataFrame({
        "Categoria":          counts.index.astype(str),
        "Freq. Absoluta":     counts.values,
        "Freq. Relativa (%)": (counts.values / total * 100).round(2),
    })
    result["Freq. Absoluta Acumulada"]     = result["Freq. Absoluta"].cumsum()
    result["Freq. Relativa Acumulada (%)"] = result["Freq. Relativa (%)"].cumsum().round(2)
    return result.reset_index(drop=True)


def inscribed_by_uf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega o total de inscritos por UF (soma de total_inscritos nos municípios).

    Returns
    -------
    pd.DataFrame com colunas: uf | municipios | total_inscritos
    """
    agg = (
        df.groupby("uf")
        .agg(municipios=("cod_7", "count"), total_inscritos=("total_inscritos", "sum"))
        .reset_index()
        .sort_values("total_inscritos", ascending=False)
    )
    return agg


# ---------------------------------------------------------------------------
# Estatísticas descritivas
# ---------------------------------------------------------------------------

def descriptive_stats(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Estatísticas descritivas das médias municipais de notas.

    Parameters
    ----------
    columns : Lista de colunas. Se None, usa SCORE_COLS disponíveis no df.
    """
    if columns is None:
        columns = [c for c in SCORE_COLS if c in df.columns]

    subset = df[columns]
    stats  = subset.describe().rename(index={
        "mean":  "Média",
        "std":   "Desvio Padrão",
        "min":   "Mínimo",
        "25%":   "Q1 (25%)",
        "50%":   "Mediana",
        "75%":   "Q3 (75%)",
        "max":   "Máximo",
    }).drop(index="count", errors="ignore")
    stats.loc["Assimetria (Skew)"] = subset.skew().round(4)
    stats.loc["Curtose"]           = subset.kurtosis().round(4)
    stats.loc["CV (%)"]            = (subset.std() / subset.mean() * 100).round(2)

    stats = stats.rename(columns=SCORE_LABELS)
    return stats.round(4)


# ---------------------------------------------------------------------------
# Correlação
# ---------------------------------------------------------------------------

def correlation_matrix(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Matriz de correlação entre variáveis quantitativas.

    Parameters
    ----------
    columns : Lista de colunas numéricas. Se None, usa SCORE_COLS disponíveis.
    method  : 'pearson', 'spearman' ou 'kendall'.
    """
    if columns is None:
        columns = [c for c in SCORE_COLS if c in df.columns]

    all_labels = {**SCORE_LABELS, **PROPORTION_LABELS}
    corr = df[columns].corr(method=method).round(4)
    corr = corr.rename(index=all_labels, columns=all_labels)
    return corr


# ---------------------------------------------------------------------------
# Análises agrupadas
# ---------------------------------------------------------------------------

def mean_scores_by_group(
    df: pd.DataFrame,
    score_col: str,
    group_col: str = "uf",
) -> pd.DataFrame:
    """
    Média ponderada (por total_inscritos) e N de municípios de `score_col`
    para cada grupo de `group_col`.

    Returns
    -------
    pd.DataFrame com colunas: Categoria | Média Ponderada | N municípios
    """
    if "total_inscritos" in df.columns and score_col in SCORE_COLS:
        # Weighted mean: reflect actual participant counts.
        # include_groups=False requires pandas >= 2.2 (see requirements.txt).
        agg = (
            df.dropna(subset=[score_col, group_col])
            .groupby(group_col)
            .apply(
                lambda g: pd.Series({
                    "Média Ponderada": np.average(g[score_col], weights=g["total_inscritos"]),
                    "N municípios":    len(g),
                }),
                include_groups=False,
            )
            .round(2)
            .reset_index()
            .rename(columns={group_col: "Categoria"})
            .sort_values("Média Ponderada", ascending=False)
        )
    else:
        agg = (
            df.groupby(group_col)[score_col]
            .agg(**{"Média Ponderada": "mean", "N municípios": "count"})
            .round(2)
            .reset_index()
            .rename(columns={group_col: "Categoria"})
            .sort_values("Média Ponderada", ascending=False)
        )
    return agg


# ---------------------------------------------------------------------------
# Normalidade
# ---------------------------------------------------------------------------

def normality_test(df: pd.DataFrame, column: str) -> dict:
    """
    Teste de Shapiro-Wilk para normalidade.
    Usa uma amostra de até 5000 linhas para grandes DataFrames.
    """
    sample = df[column].dropna()
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=42)

    stat, pval = scipy_stats.shapiro(sample)
    return {
        "teste":       "Shapiro-Wilk",
        "estatística": round(float(stat), 6),
        "p_valor":     round(float(pval), 6),
        "normal":      bool(pval > 0.05),
    }
