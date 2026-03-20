"""
analysis.py – Operações estatísticas e de análise dos dados do ENEM 2024.

Expõe funções auxiliares usadas pelo aplicativo Streamlit (app.py):
  - apply_labels          → adiciona colunas com rótulos legíveis
  - frequency_table       → tabela de distribuição de frequência
  - descriptive_stats     → estatísticas descritivas (incluindo assimetria e curtose)
  - correlation_matrix    → matriz de correlação entre variáveis quantitativas
  - mean_scores_by_group  → média de uma nota agrupada por categoria
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Mapeamentos de rótulos
# ---------------------------------------------------------------------------

COR_RACA_MAP: dict[int, str] = {
    0: "Não declarado",
    1: "Branca",
    2: "Parda",
    3: "Preta",
    4: "Amarela",
    5: "Indígena",
}

SEXO_MAP: dict[str, str] = {
    "M": "Masculino",
    "F": "Feminino",
}

ESCOLA_MAP: dict[int, str] = {
    1: "Não respondeu",
    2: "Pública",
    3: "Privada",
    4: "Exterior",
}

ST_CONCLUSAO_MAP: dict[int, str] = {
    1: "Já concluiu",
    2: "Cursando – conclui em 2024",
    3: "Cursando – não conclui em 2024",
    4: "Não está cursando",
}

TREINEIRO_MAP: dict[int, str] = {
    0: "Não",
    1: "Sim",
}

# Rótulos exibidos nos gráficos/tabelas para variáveis categóricas
QUALITATIVE_VARS: list[dict] = [
    {"col": "TP_SEXO",          "label_col": "SEXO_DESC",         "title": "Sexo"},
    {"col": "TP_COR_RACA",      "label_col": "COR_RACA_DESC",     "title": "Raça/Cor"},
    {"col": "TP_ESCOLA",        "label_col": "ESCOLA_DESC",       "title": "Tipo de Escola"},
    {"col": "TP_ST_CONCLUSAO",  "label_col": "ST_CONCLUSAO_DESC", "title": "Situação de Conclusão"},
    {"col": "IN_TREINEIRO",     "label_col": "TREINEIRO_DESC",    "title": "Treineiro"},
    {"col": "SG_UF_RESIDENCIA", "label_col": "SG_UF_RESIDENCIA",  "title": "UF de Residência"},
]

# Variáveis quantitativas (notas)
QUANTITATIVE_VARS: list[str] = [
    "NU_NOTA_CN",
    "NU_NOTA_CH",
    "NU_NOTA_LC",
    "NU_NOTA_MT",
    "NU_NOTA_REDACAO",
]

SCORE_LABELS: dict[str, str] = {
    "NU_NOTA_CN":      "Ciências da Natureza",
    "NU_NOTA_CH":      "Ciências Humanas",
    "NU_NOTA_LC":      "Linguagens e Códigos",
    "NU_NOTA_MT":      "Matemática",
    "NU_NOTA_REDACAO": "Redação",
}


# ---------------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------------

def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna uma cópia do DataFrame com colunas de rótulos adicionadas
    para cada variável categórica codificada numericamente.
    """
    df = df.copy()
    mappings = [
        ("TP_COR_RACA",     COR_RACA_MAP,     "COR_RACA_DESC"),
        ("TP_SEXO",         SEXO_MAP,         "SEXO_DESC"),
        ("TP_ESCOLA",       ESCOLA_MAP,       "ESCOLA_DESC"),
        ("TP_ST_CONCLUSAO", ST_CONCLUSAO_MAP, "ST_CONCLUSAO_DESC"),
        ("IN_TREINEIRO",    TREINEIRO_MAP,    "TREINEIRO_DESC"),
    ]
    for src_col, mapping, dst_col in mappings:
        if src_col in df.columns:
            df[dst_col] = df[src_col].map(mapping)
    return df


# ---------------------------------------------------------------------------
# Tabelas de frequência
# ---------------------------------------------------------------------------

def frequency_table(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Calcula a tabela de distribuição de frequência para uma variável categórica.

    Returns
    -------
    pd.DataFrame com colunas:
        Categoria | Freq. Absoluta | Freq. Relativa (%) |
        Freq. Absoluta Acumulada | Freq. Relativa Acumulada (%)
    """
    counts = df[column].value_counts(dropna=False)
    total = counts.sum()

    result = pd.DataFrame(
        {
            "Categoria": counts.index.astype(str),
            "Freq. Absoluta": counts.values,
            "Freq. Relativa (%)": (counts.values / total * 100).round(2),
        }
    )
    result["Freq. Absoluta Acumulada"] = result["Freq. Absoluta"].cumsum()
    result["Freq. Relativa Acumulada (%)"] = result["Freq. Relativa (%)"].cumsum().round(2)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Estatísticas descritivas
# ---------------------------------------------------------------------------

def descriptive_stats(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Retorna um DataFrame com as principais estatísticas descritivas das
    variáveis quantitativas, incluindo mediana, coeficiente de variação,
    assimetria e curtose.

    Parameters
    ----------
    df      : DataFrame com os dados.
    columns : Lista de colunas a analisar. Se None, usa QUANTITATIVE_VARS.

    Returns
    -------
    pd.DataFrame com índices = estatísticas e colunas = variáveis.
    """
    if columns is None:
        columns = [c for c in QUANTITATIVE_VARS if c in df.columns]

    subset = df[columns]

    stats = subset.describe().rename(
        index={
            "count": "N",
            "mean":  "Média",
            "std":   "Desvio Padrão",
            "min":   "Mínimo",
            "25%":   "Q1 (25%)",
            "50%":   "Mediana",
            "75%":   "Q3 (75%)",
            "max":   "Máximo",
        }
    )

    stats.loc["Mediana"]  = subset.median().round(2)
    stats.loc["Assimetria (Skew)"] = subset.skew().round(4)
    stats.loc["Curtose"]  = subset.kurtosis().round(4)
    stats.loc["CV (%)"]   = (subset.std() / subset.mean() * 100).round(2)

    # Rename columns with friendly labels
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
    Calcula a matriz de correlação entre as variáveis quantitativas.

    Parameters
    ----------
    columns : Lista de colunas. Se None, usa QUANTITATIVE_VARS.
    method  : 'pearson', 'spearman' ou 'kendall'.
    """
    if columns is None:
        columns = [c for c in QUANTITATIVE_VARS if c in df.columns]

    corr = df[columns].corr(method=method).round(4)
    corr = corr.rename(index=SCORE_LABELS, columns=SCORE_LABELS)
    return corr


# ---------------------------------------------------------------------------
# Análises agrupadas
# ---------------------------------------------------------------------------

def mean_scores_by_group(
    df: pd.DataFrame,
    score_col: str,
    group_col: str,
) -> pd.DataFrame:
    """
    Calcula média, mediana e N de `score_col` para cada grupo de `group_col`.

    Returns
    -------
    pd.DataFrame com colunas: Categoria | Média | Mediana | N
    """
    agg = (
        df.groupby(group_col)[score_col]
        .agg(Média="mean", Mediana="median", N="count")
        .round(2)
        .reset_index()
        .rename(columns={group_col: "Categoria"})
        .sort_values("Média", ascending=False)
    )
    return agg


def score_percentiles(df: pd.DataFrame, column: str, q: list[float] | None = None) -> pd.Series:
    """
    Retorna percentis de uma coluna de notas.

    Parameters
    ----------
    q : Lista de percentis (0–100). Padrão: [10, 25, 50, 75, 90].
    """
    if q is None:
        q = [10, 25, 50, 75, 90]
    percentiles = np.percentile(df[column].dropna(), q)
    return pd.Series(percentiles, index=[f"P{int(p)}" for p in q])


def normality_test(df: pd.DataFrame, column: str) -> dict:
    """
    Realiza o teste de Shapiro-Wilk (amostra ≤ 5000) ou D'Agostino-Pearson
    (amostra > 5000) para verificar normalidade.

    Returns
    -------
    dict com chaves: 'teste', 'estatística', 'p_valor', 'normal'
    """
    sample = df[column].dropna()
    n = len(sample)
    if n > 5000:
        sample = sample.sample(5000, random_state=42)

    stat, pval = scipy_stats.shapiro(sample)
    return {
        "teste": "Shapiro-Wilk",
        "estatística": round(float(stat), 6),
        "p_valor": round(float(pval), 6),
        "normal": bool(pval > 0.05),
    }
