"""
analysis.py – Operações estatísticas e de análise dos dados do ENEM 2024.

O DataFrame de entrada tem uma linha por município (estrutura produzida
pelo pipeline de agregação em database.py) com as seguintes colunas:

  Identificação:  cod_7, municipio, uf
  Participantes:  total_inscritos, media_idade
                  sexo_feminino, sexo_masculino
                  raca_branca, raca_preta, raca_parda, raca_amarela, raca_indigena
                  est_civil_solteiro, est_civil_casado, nac_brasileiro
                  concluiu_em, ensino_regular, classe_media_alta
                  tem_internet, tem_computador
  Resultados:     media_cn, media_ch, media_lc, media_mt, media_redacao, media_geral

Após apply_labels() são adicionadas colunas derivadas pct_* (proporções).

Funções exportadas
------------------
  apply_labels        – mapeia uf numérico para sigla e calcula pct_* columns
  frequency_table     – tabela de distribuição de frequência para variável categórica
  descriptive_stats   – estatísticas descritivas de colunas numéricas
  correlation_matrix  – matriz de correlação (Pearson/Spearman/Kendall)
  mean_scores_by_group – médias de uma nota agrupadas por UF
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

# Única variável categórica genuína no DataFrame consolidado
QUALITATIVE_VARS: list[dict] = [
    {"col": "uf", "title": "UF (Estado)"},
]

# ---------------------------------------------------------------------------
# Variáveis quantitativas – notas médias municipais
# ---------------------------------------------------------------------------

SCORE_COLS: list[str] = [
    "media_cn",
    "media_ch",
    "media_lc",
    "media_mt",
    "media_redacao",
    "media_geral",
]

SCORE_LABELS: dict[str, str] = {
    "media_cn":      "Ciências da Natureza",
    "media_ch":      "Ciências Humanas",
    "media_lc":      "Linguagens e Códigos",
    "media_mt":      "Matemática",
    "media_redacao": "Redação",
    "media_geral":   "Média Geral",
}

# ---------------------------------------------------------------------------
# Colunas de proporção (derivadas em apply_labels)
# ---------------------------------------------------------------------------

# Mapeamento: coluna_pct → coluna_contagem_origem
PROPORTION_MAP: dict[str, str] = {
    "pct_feminino":        "sexo_feminino",
    "pct_parda":           "raca_parda",
    "pct_branca":          "raca_branca",
    "pct_preta":           "raca_preta",
    "pct_amarela":         "raca_amarela",
    "pct_indigena":        "raca_indigena",
    "pct_solteiro":        "est_civil_solteiro",
    "pct_concluiu_em":     "concluiu_em",
    "pct_ensino_regular":  "ensino_regular",
    "pct_classe_media_alta": "classe_media_alta",
    "pct_internet":        "tem_internet",
    "pct_computador":      "tem_computador",
}

PROPORTION_LABELS: dict[str, str] = {
    "pct_feminino":          "% Feminino",
    "pct_parda":             "% Parda",
    "pct_branca":            "% Branca",
    "pct_preta":             "% Preta",
    "pct_amarela":           "% Amarela",
    "pct_indigena":          "% Indígena",
    "pct_solteiro":          "% Solteiro(a)",
    "pct_concluiu_em":       "% Concluiu EM",
    "pct_ensino_regular":    "% Ensino Regular",
    "pct_classe_media_alta": "% Classe Média/Alta",
    "pct_internet":          "% com Internet",
    "pct_computador":        "% com Computador",
}

PROPORTION_COLS: list[str] = list(PROPORTION_MAP.keys())


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

    Ao trabalhar com dados municipais, cada linha representa um município;
    portanto a frequência absoluta conta municípios por categoria.

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
    result["Freq. Absoluta Acumulada"]    = result["Freq. Absoluta"].cumsum()
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
        "count": "N (municípios)",
        "mean":  "Média",
        "std":   "Desvio Padrão",
        "min":   "Mínimo",
        "25%":   "Q1 (25%)",
        "50%":   "Mediana",
        "75%":   "Q3 (75%)",
        "max":   "Máximo",
    })
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
