"""
app.py – Dashboard ENEM 2024 (Streamlit)

Os dados estão agregados ao nível municipal: uma linha por município,
produzida pelo pipeline de 3 tabelas (participantes, resultados, municipio).

Páginas disponíveis no menu lateral:
  🏠 Visão Geral           → KPIs nacionais, inscritos por UF, médias por área
  📊 Variáveis Qualitativas → distribuição de municípios por UF; composição demográfica
  📈 Variáveis Quantitativas → histogramas e box plots das médias municipais de notas
  🔗 Análise de Correlação  → heatmap e scatter matrix notas × proporções demográficas
"""

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------
# Configuração global: formato numérico brasileiro (. = mil, , = decimal)
# ---------------------------------------------------------------------------

_ptbr_template = go.layout.Template(layout=go.Layout(separators=",."))
pio.templates["ptbr"] = _ptbr_template
pio.templates.default = "plotly+ptbr"

from analysis import (
    HEATMAP_COLS,
    PROPORTION_COLS,
    PROPORTION_GROUPS,
    PROPORTION_LABELS,
    QUALITATIVE_VARS,
    SCORE_COLS,
    SCORE_LABELS,
    apply_labels,
    correlation_matrix,
    descriptive_stats,
    frequency_table,
    inscribed_by_uf,
    mean_scores_by_group,
    normality_test,
    NORMALITY_TESTS,
)
from database import load_data, load_sampling_data

warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")


# ---------------------------------------------------------------------------
# Leitura das credenciais do banco de dados (st.secrets)
# ---------------------------------------------------------------------------

def _get_db_config() -> dict | None:
    """
    Lê as credenciais do banco PostgreSQL a partir de st.secrets.

    Aceita dois formatos:
    - Seção [database] com as chaves host/port/dbname/user/password (recomendado).
    - Chaves host/port/dbname/user/password definidas no nível raiz do secrets.

    Retorna None quando as credenciais não estiverem configuradas ou ainda
    contiverem os valores de placeholder.
    """
    _DB_KEYS = {"host", "port", "dbname", "user", "password"}
    try:
        # Formato preferencial: [database] section
        try:
            section = dict(st.secrets["database"])
        except KeyError:
            # Fallback: chaves definidas no nível raiz
            section = {k: st.secrets[k] for k in _DB_KEYS if k in st.secrets}

        if not section or str(section.get("host", "")).upper().startswith("SEU_"):
            return None
        return section
    except (KeyError, FileNotFoundError):
        return None


# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Dashboard ENEM 2024",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cache – carrega os dados uma única vez
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Carregando dados do ENEM 2024…", ttl=3600, persist="disk")
def get_data(db_config: dict) -> pd.DataFrame:
    df = load_data(db_config)
    df = apply_labels(df)
    return df


@st.cache_data(show_spinner="Calculando amostras…", ttl=3600, persist="disk")
def get_sampling_data(db_config: dict) -> dict:
    return load_sampling_data(db_config)


# ---------------------------------------------------------------------------
# Helpers (definidos aqui para estarem disponíveis antes de qualquer uso)
# ---------------------------------------------------------------------------

def score_label(col: str) -> str:
    return SCORE_LABELS.get(col, PROPORTION_LABELS.get(col, col))


def _fmt_br(value: float | int, decimals: int | None = None) -> str:
    """Formata número no padrão brasileiro (. para milhares, , para decimais)."""
    if pd.isna(value):
        return "–"
    if decimals is not None:
        s = f"{float(value):,.{decimals}f}"
    else:
        s = f"{int(round(float(value))):,}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def card_metric(col, label, value, delta=None):
    with col:
        st.metric(label=label, value=value, delta=delta)


_db_config = _get_db_config()

if not _db_config:
    st.error(
        "⚠️ **Credenciais do banco de dados não configuradas.**\n\n"
        "Configure as variáveis `host`, `port`, `dbname`, `user` e `password` "
        "em **Settings → Secrets** no Streamlit Cloud para carregar os dados reais do ENEM 2024."
    )
    st.stop()

df_full = get_data(_db_config)

# ---------------------------------------------------------------------------
# Sidebar – navegação + filtros globais
# ---------------------------------------------------------------------------

st.sidebar.title("📚 ENEM 2024")
st.sidebar.markdown("---")

PAGES = [
    "📖 Introdução",
    "🏠 Visão Geral",
    "📊 Variáveis Qualitativas",
    "📈 Variáveis Quantitativas",
    "🔗 Análise de Correlação",
    "🔬 Amostragem",
]
page = st.sidebar.radio("Navegação", PAGES)

if page not in ("📖 Introdução", "🔬 Amostragem"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros globais")

    # Determine whether the current page is in a sampling mode.
    # Session state persists widget values across reruns, so we can read
    # the mode selector before it is rendered again this cycle.
    _SAMPLING_MODES_SET = {
        "AAS – Amostragem Aleatória Simples",
        "Estratificada (por cor/raça)",
        "Estratificada (por UF)",
        "Sistemática",
    }
    _PAGE_MODE_KEYS = {
        "📊 Variáveis Qualitativas":  "qual_modo_sel",
        "📈 Variáveis Quantitativas": "quant_modo_sel",
        "🔗 Análise de Correlação":   "corr_modo_sel",
    }
    _current_mode = st.session_state.get(_PAGE_MODE_KEYS.get(page, ""), "Agregação Municipal")
    _is_sampling_mode = _current_mode in _SAMPLING_MODES_SET

    if _is_sampling_mode:
        st.sidebar.info("🔬 Filtro de UF desativado no modo de amostragem.")
        df = df_full.copy()
    else:
        # Filtro por UF
        all_ufs = sorted(df_full["uf"].dropna().unique().tolist())
        sel_ufs = st.sidebar.multiselect("UF (Estado)", all_ufs, default=all_ufs)

        # Aplica filtro de UF
        df = df_full[df_full["uf"].isin(sel_ufs)].copy()

        st.sidebar.markdown(f"**Municípios filtrados:** {_fmt_br(len(df))}")

        if df.empty:
            st.warning(
                "⚠️ Nenhum município corresponde aos filtros selecionados. "
                "Selecione ao menos uma UF."
            )
            st.stop()
else:
    df = df_full.copy()

# ---------------------------------------------------------------------------
# Paleta de cores padrão
# ---------------------------------------------------------------------------

PALETTE = px.colors.qualitative.Plotly


# ===========================================================================
# PÁGINA 0 – INTRODUÇÃO
# ===========================================================================

if page == "📖 Introdução":
    st.title("📖 Introdução")
    st.markdown("---")

    st.markdown(
        """
## Sobre este dashboard

Este painel interativo apresenta uma **análise exploratória dos microdados do ENEM 2024**,
organizada em quatro seções temáticas:

| Aba | Conteúdo |
|-----|----------|
| 🏠 Visão Geral | KPIs nacionais, total de inscritos por UF e médias por área de conhecimento |
| 📊 Variáveis Qualitativas | Distribuição de municípios por UF e composição demográfica dos participantes |
| 📈 Variáveis Quantitativas | Histogramas, box plots e estatísticas descritivas das notas municipais |
| 🔗 Análise de Correlação | Heatmap e scatter matrix correlacionando notas com indicadores socioeconômicos |
| 🔬 Amostragem | Dimensionamento amostral e comparação de três métodos de amostragem |

---

## Etapas de desenvolvimento

1. **Extração dos microdados** – os arquivos originais do INEP foram carregados em um banco
   de dados PostgreSQL via Google Colab.
2. **Agregação municipal** – os registros individuais foram agrupados por município
   (`co_municipio_prova`), calculando médias ponderadas, contagens e proporções
   demográficas por meio de queries SQL.
3. **Modelagem** – o resultado foi exportado como um DataFrame pandas único
   (uma linha por município) com as colunas de notas, inscritos e indicadores
   socioeconômicos do questionário socioeconômico (Q001-Q023).
4. **Amostragem estatística** – foram aplicados três métodos de amostragem
   (**Aleatória Simples**, **Estratificada** e **Sistemática**) sobre duas tabelas
   de granularidade individual: `ed_enem_2024_participantes` (N ≈ 4,3 M inscritos,
   estratificada por cor/raça) e `ed_enem_2024_resultados` (N ≈ 3 M com as 5
   notas válidas, estratificada por UF). Os tamanhos amostrais foram
   calculados pela fórmula de Cochran com correção para população finita
   (confiança 95 %). Acesse a aba **🔬 Amostragem** ou use o filtro
   **Modo de visualização** nas abas Variáveis para explorar os resultados.
5. **Visualização** – o dashboard foi construído com **Streamlit** e **Plotly Express**.

---

## Método de agregação dos dados

Cada linha do conjunto de dados representa **um município**.
As variáveis numéricas (notas por área, idade) foram calculadas como
**médias ponderadas pelo número de inscritos**, de modo que municípios maiores
contribuem proporcionalmente mais para os totais nacionais e estaduais.
As variáveis demográficas (proporções de gênero, renda, escolaridade dos pais etc.)
representam a **média das proporções municipais** dentro de cada estado.

---

## ⚠️ Observação sobre o filtro de UF

O **filtro de UF** disponível no painel lateral afeta os indicadores e gráficos
que dependem do subconjunto de municípios selecionados — por exemplo, os KPIs
da Visão Geral, os histogramas e as estatísticas descritivas.

Entretanto, os **objetos de comparação entre UFs** — como os gráficos de barras
por estado, os box plots por UF e as tabelas de médias ponderadas por estado —
**utilizam sempre o conjunto completo de dados** (`df_full`), independentemente
do filtro aplicado. Isso garante que a comparação entre estados permaneça
consistente e não seja distorcida pela seleção parcial de municípios.
        """
    )


# ===========================================================================
# PÁGINA 1 – VISÃO GERAL
# ===========================================================================

elif page == "🏠 Visão Geral":
    st.title("🏠 Visão Geral – ENEM 2024")
    st.markdown(
        "Análise exploratória dos microdados do **ENEM 2024** agregados ao nível municipal. "
        "Cada ponto de dados representa um município. "
        "Use os filtros no painel lateral para refinar a visualização."
    )
    st.markdown("---")

    # --- KPIs ---
    n_municipios   = len(df)
    total_inscritos = int(df["total_inscritos"].sum())
    media_idade     = float(
        np.average(df["media_idade"], weights=df["total_inscritos"])
        if "total_inscritos" in df.columns else df["media_idade"].mean()
    )
    available_scores = [c for c in SCORE_COLS if c in df.columns]
    media_geral_br   = (
        float(np.average(df["nota_geral_media"], weights=df["total_inscritos"]))
        if "nota_geral_media" in df.columns
        else float(df[available_scores].mean().mean())
    )

    c1, c2, c3, c4 = st.columns(4)
    card_metric(c1, "Municípios analisados",             _fmt_br(n_municipios))
    card_metric(c2, "Total de inscritos",                _fmt_br(total_inscritos))
    card_metric(c3, "Média de idade (ponderada)",        f"{_fmt_br(media_idade, 1)} anos")
    card_metric(c4, "Média geral nacional (ponderada)",  _fmt_br(media_geral_br, 1))

    st.markdown("---")

    # --- Inscritos por UF ---
    col_uf, col_scores = st.columns(2)

    with col_uf:
        st.subheader("Total de Inscritos por UF")
        uf_agg = inscribed_by_uf(df_full).head(27)
        fig_uf = px.bar(
            uf_agg,
            x="uf",
            y="total_inscritos",
            color="total_inscritos",
            color_continuous_scale="Blues",
            text=[_fmt_br(v) for v in uf_agg["total_inscritos"]],
            labels={"uf": "UF", "total_inscritos": "Inscritos"},
        )
        fig_uf.update_traces(texttemplate="%{text}", textposition="outside")
        fig_uf.update_layout(height=400, showlegend=False,
                              coloraxis_showscale=False,
                              xaxis_title="UF", yaxis_title="Inscritos")
        st.plotly_chart(fig_uf, use_container_width=True)

    with col_scores:
        st.subheader("Médias Nacionais por Área de Conhecimento")
        if available_scores:
            medias = pd.Series({
                SCORE_LABELS[c]: float(np.average(df[c].dropna(),
                                                   weights=df.loc[df[c].notna(), "total_inscritos"]))
                for c in available_scores if c in df.columns
            })
            fig_medias = px.bar(
                x=medias.index,
                y=medias.values,
                color=medias.values,
                color_continuous_scale="Viridis",
                text=[_fmt_br(v, 1) for v in medias.values],
                labels={"x": "Área", "y": "Nota"},
            )
            fig_medias.update_traces(texttemplate="%{text}", textposition="outside")
            fig_medias.update_layout(height=400, showlegend=False,
                                     coloraxis_showscale=False,
                                     xaxis_title="Área", yaxis_title="Média")
            st.plotly_chart(fig_medias, use_container_width=True)

    # --- Prévia dos dados ---
    with st.expander("📋 Prévia dos dados municipais (primeiras 200 linhas)"):
        preview_df = df.head(200).copy()
        _drop = {"cod_7"}
        _first = [c for c in ["municipio", "uf"] if c in preview_df.columns]
        _rest  = [c for c in preview_df.columns if c not in set(_first) | _drop]
        # Format percentage and score columns with Brazilian locale
        _pct_cols   = [c for c in preview_df.columns if c.startswith("pct_")]
        _score_cols = [c for c in SCORE_COLS if c in preview_df.columns]
        for _c in _pct_cols:
            preview_df[_c] = preview_df[_c].apply(lambda v: _fmt_br(v, 2))
        for _c in _score_cols:
            preview_df[_c] = preview_df[_c].apply(
                lambda v: _fmt_br(v, 1) if pd.notna(v) else "–"
            )
        st.dataframe(preview_df[_first + _rest], use_container_width=True)


# ===========================================================================
# PÁGINA 2 – VARIÁVEIS QUALITATIVAS
# ===========================================================================

elif page == "📊 Variáveis Qualitativas":
    st.title("📊 Variáveis Qualitativas")

    _QUAL_MODO_OPTS = [
        "Agregação Municipal",
        "AAS – Amostragem Aleatória Simples",
        "Estratificada (por cor/raça)",
        "Sistemática",
    ]
    _qual_modo = st.selectbox(
        "Modo de visualização",
        _QUAL_MODO_OPTS,
        key="qual_modo_sel",
        help=(
            "**Agregação Municipal**: dados agregados por município (uma linha por município).\n\n"
            "**AAS / Estratificada / Sistemática**: dados individuais amostrados da tabela "
            "`ed_enem_2024_participantes` (N ≈ 4,3 M inscritos)."
        ),
    )
    st.markdown("---")

    if _qual_modo == "Agregação Municipal":
        st.markdown(
            "Tabelas de distribuição de frequência e gráficos para a variável "
            "categórica principal (UF) e a composição demográfica dos municípios."
        )
        st.markdown("---")

    # ---- Seção 1: Distribuição por UF ----
        st.subheader("🔹 Distribuição de Municípios por UF")

        freq_df = frequency_table(df_full, "uf")
        tab_table, tab_bar = st.tabs(
            ["Tabela de Frequência", "Gráfico de Barras"]
        )

        with tab_table:
            st.markdown("Cada linha representa quantos municípios estão em cada estado.")
            freq_df_fmt = freq_df.copy()
            for _c in ["Freq. Relativa (%)", "Freq. Relativa Acumulada (%)"]:
                if _c in freq_df_fmt.columns:
                    freq_df_fmt[_c] = freq_df_fmt[_c].apply(lambda v: _fmt_br(v, 2))
            st.dataframe(freq_df_fmt, use_container_width=True, hide_index=True)

        with tab_bar:
            _freq_plot = freq_df.copy()
            _freq_plot["_text"] = _freq_plot["Freq. Relativa (%)"].apply(
                lambda v: f"{_fmt_br(v, 1)}%"
            )
            fig_bar = px.bar(
                _freq_plot,
                x="Categoria",
                y="Freq. Absoluta",
                text="_text",
                color_discrete_sequence=["#636EFA"],
                labels={"Freq. Absoluta": "Nº de Municípios"},
                title="Número de Municípios por UF",
            )
            fig_bar.update_traces(texttemplate="%{text}", textposition="outside")
            fig_bar.update_layout(showlegend=False, height=440,
                                   xaxis_title="UF", yaxis_title="Municípios")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

        # ---- Seção 2: Total de Inscritos por UF ----
        st.subheader("🔹 Total de Inscritos por UF")
        uf_ins = inscribed_by_uf(df_full)

        col_ins_bar, col_ins_tab = st.columns([2, 1])
        with col_ins_bar:
            fig_ins = px.bar(
                uf_ins,
                x="uf",
                y="total_inscritos",
                color="total_inscritos",
                color_continuous_scale="Teal",
                text=[_fmt_br(v) for v in uf_ins["total_inscritos"]],
                title="Total de Inscritos por Estado",
                labels={"uf": "UF", "total_inscritos": "Inscritos"},
            )
            fig_ins.update_traces(texttemplate="%{text}", textposition="outside")
            fig_ins.update_layout(height=420, showlegend=False,
                                   coloraxis_showscale=False,
                                   xaxis_title="UF", yaxis_title="Inscritos")
            st.plotly_chart(fig_ins, use_container_width=True)

        with col_ins_tab:
            st.markdown("**Tabela resumo**")
            _tab_ins = uf_ins.rename(columns={"uf": "UF", "municipios": "Municípios",
                                               "total_inscritos": "Inscritos"}).copy()
            _tab_ins["Inscritos"] = _tab_ins["Inscritos"].apply(_fmt_br)
            st.dataframe(
                _tab_ins,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")

        # ---- Seção 3: Composição Demográfica por UF ----
        st.subheader("🔹 Composição Demográfica por UF")
        st.markdown(
            "Para cada estado, a média das proporções dos municípios "
            "em cada grupo demográfico ou socioeconômico (Q001-Q023)."
        )

        demo_cols = PROPORTION_GROUPS

        demo_choice = st.selectbox(
            "Grupo demográfico",
            list(demo_cols.keys()),
            key="demo_group_sel",
        )
        chosen_cols = [c for c in demo_cols[demo_choice] if c in df_full.columns]

        if chosen_cols:
            uf_demo = (
                df_full.groupby("uf")[chosen_cols].mean().round(2).reset_index()
            )
            uf_demo_melt = uf_demo.melt(id_vars="uf", var_name="Indicador", value_name="Proporção (%)")
            uf_demo_melt["Indicador"] = uf_demo_melt["Indicador"].map(PROPORTION_LABELS)

            fig_demo = px.bar(
                uf_demo_melt,
                x="uf",
                y="Proporção (%)",
                color="Indicador",
                barmode="group",
                color_discrete_sequence=PALETTE,
                title=f"Média Municipal de {demo_choice} por UF",
                labels={"uf": "UF"},
            )
            fig_demo.update_layout(height=460, xaxis_title="UF")
            st.plotly_chart(fig_demo, use_container_width=True)
        else:
            st.info("Colunas de proporção não disponíveis. Verifique se os dados foram carregados corretamente.")

    else:
        # ---- Modo amostragem ----
        _QUAL_SAMPLE_MAP = {
            "AAS – Amostragem Aleatória Simples": "aas_part",
            "Estratificada (por cor/raça)":       "estratificada_part",
            "Sistemática":                         "sistematica_part",
        }
        _qual_sample_key = _QUAL_SAMPLE_MAP[_qual_modo]

        try:
            _q_sampling = get_sampling_data(_db_config)
        except Exception as _exc:
            st.error(f"Erro ao carregar dados de amostragem: {_exc}")
            st.stop()

        _df_q = _q_sampling[_qual_sample_key]
        _mom_part_q = _q_sampling["momentos_part"]
        _n_part_q   = _q_sampling["n_amostra_part"]
        _k_part_q   = _q_sampling["k_part"]
        _df_cor_pop  = _q_sampling["cor_raca_pop"]
        _df_sexo_pop = _q_sampling["sexo_pop"]

        st.markdown(
            f"Dados individuais amostrados da tabela **`ed_enem_2024_participantes`** "
            f"(N = {_fmt_br(_mom_part_q['N'])} inscritos) pelo método **{_qual_modo}**."
        )

        if _df_q.empty:
            st.warning("A amostra está vazia. Verifique os dados no banco.")
            st.stop()

        st.markdown(f"**n (amostra) = {_fmt_br(len(_df_q))}**")
        st.markdown("---")

        # ---- Distribuição de Cor/Raça ----
        st.subheader("🔹 Distribuição por Cor/Raça")
        if "cor_raca" in _df_q.columns:
            _cr_counts = _df_q["cor_raca"].fillna("Não declarado").value_counts().reset_index()
            _cr_counts.columns = ["Cor/Raça", "n"]
            _cr_counts["% amostra"] = (_cr_counts["n"] / _cr_counts["n"].sum() * 100).round(2)

            _cr_pop = _df_cor_pop.rename(columns={"cor_raca": "Cor/Raça", "pop_count": "N pop."}).copy()
            _cr_pop["% pop."] = (_cr_pop["N pop."] / _cr_pop["N pop."].sum() * 100).round(2)

            _cr_merged = pd.merge(_cr_pop[["Cor/Raça", "N pop.", "% pop."]], _cr_counts, on="Cor/Raça", how="outer").fillna(0)
            _cr_merged["N pop."] = _cr_merged["N pop."].astype(int)
            _cr_merged["n"]      = _cr_merged["n"].astype(int)
            _cr_merged = _cr_merged.sort_values("% pop.", ascending=False)

            _cr_col_chart, _cr_col_tab = st.columns(2)
            with _cr_col_chart:
                _fig_cr = go.Figure()
                _fig_cr.add_bar(x=_cr_merged["Cor/Raça"], y=_cr_merged["% pop."],
                                name="População (%)", marker_color="#636EFA")
                _fig_cr.add_bar(x=_cr_merged["Cor/Raça"], y=_cr_merged["% amostra"],
                                name="Amostra (%)", marker_color="#EF553B")
                _fig_cr.update_layout(
                    title="Proporção por Cor/Raça: Amostra × População",
                    barmode="group", xaxis_title="Cor/Raça", yaxis_title="Proporção (%)",
                    height=380,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(_fig_cr, use_container_width=True)
            with _cr_col_tab:
                st.markdown("**Tabela: Cor/Raça × Amostra**")
                st.dataframe(_cr_merged.rename(columns={"n": "Amostra (n)", "% amostra": "Amostra (%)"}),
                             use_container_width=True, hide_index=True)
        else:
            st.info("Coluna `cor_raca` não disponível na amostra.")

        st.markdown("---")

        # ---- Distribuição de Sexo ----
        st.subheader("🔹 Distribuição por Sexo")
        if "sexo" in _df_q.columns:
            _sx_counts = _df_q["sexo"].fillna("Não informado").value_counts().reset_index()
            _sx_counts.columns = ["Sexo", "n"]
            _sx_counts["% amostra"] = (_sx_counts["n"] / _sx_counts["n"].sum() * 100).round(2)

            _sx_pop = _df_sexo_pop.rename(columns={"sexo": "Sexo", "pop_count": "N pop."}).copy()
            _sx_pop["% pop."] = (_sx_pop["N pop."] / _sx_pop["N pop."].sum() * 100).round(2)

            _sx_merged = pd.merge(_sx_pop[["Sexo", "N pop.", "% pop."]], _sx_counts, on="Sexo", how="outer").fillna(0)
            _sx_merged["N pop."] = _sx_merged["N pop."].astype(int)
            _sx_merged["n"]      = _sx_merged["n"].astype(int)
            _sx_merged = _sx_merged.sort_values("% pop.", ascending=False)

            _sx_col_chart, _sx_col_tab = st.columns(2)
            with _sx_col_chart:
                _fig_sx = go.Figure()
                _fig_sx.add_bar(x=_sx_merged["Sexo"], y=_sx_merged["% pop."],
                                name="População (%)", marker_color="#636EFA")
                _fig_sx.add_bar(x=_sx_merged["Sexo"], y=_sx_merged["% amostra"],
                                name="Amostra (%)", marker_color="#EF553B")
                _fig_sx.update_layout(
                    title="Proporção por Sexo: Amostra × População",
                    barmode="group", xaxis_title="Sexo", yaxis_title="Proporção (%)",
                    height=320,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(_fig_sx, use_container_width=True)
            with _sx_col_tab:
                st.markdown("**Tabela: Sexo × Amostra**")
                st.dataframe(_sx_merged.rename(columns={"n": "Amostra (n)", "% amostra": "Amostra (%)"}),
                             use_container_width=True, hide_index=True)
        else:
            st.info("Coluna `sexo` não disponível na amostra.")

        st.markdown("---")

        # ---- Distribuição de Idade ----
        st.subheader("🔹 Distribuição de Idade")
        if "idade" in _df_q.columns:
            _id_vals = _df_q["idade"].dropna()
            _fig_id = px.histogram(
                _id_vals, nbins=40,
                title="Distribuição de Idade (amostra)",
                labels={"value": "Idade", "count": "Participantes"},
                color_discrete_sequence=["#636EFA"],
            )
            _fig_id.add_vline(x=float(_id_vals.mean()), line_dash="dash", line_color="blue",
                              annotation_text=f"x̄ = {_fmt_br(float(_id_vals.mean()), 1)}",
                              annotation_position="top right")
            _fig_id.add_vline(x=_mom_part_q["media"], line_dash="dot", line_color="red",
                              annotation_text=f"μ pop. = {_fmt_br(_mom_part_q['media'], 1)}",
                              annotation_position="top left")
            _fig_id.update_layout(height=360, xaxis_title="Idade", yaxis_title="Participantes")
            st.plotly_chart(_fig_id, use_container_width=True)
        else:
            st.info("Coluna `idade` não disponível na amostra.")


# ===========================================================================
# PÁGINA 3 – VARIÁVEIS QUANTITATIVAS
# ===========================================================================

elif page == "📈 Variáveis Quantitativas":
    st.title("📈 Variáveis Quantitativas")

    _QUANT_MODO_OPTS = [
        "Agregação Municipal",
        "AAS – Amostragem Aleatória Simples",
        "Estratificada (por UF)",
        "Sistemática",
    ]
    _quant_modo = st.selectbox(
        "Modo de visualização",
        _QUANT_MODO_OPTS,
        key="quant_modo_sel",
        help=(
            "**Agregação Municipal**: médias calculadas por município (uma linha por município).\n\n"
            "**AAS / Estratificada / Sistemática**: dados individuais amostrados da tabela "
            "`ed_enem_2024_resultados` (N ≈ 3 M participantes com as 5 notas válidas)."
        ),
    )
    st.markdown("---")

    if _quant_modo == "Agregação Municipal":
        st.markdown(
            "Histogramas, box plots e estatísticas descritivas das **médias municipais** "
            "de notas do ENEM 2024. Cada observação representa um município."
        )
        st.markdown("---")

        available_scores = [c for c in SCORE_COLS if c in df.columns]
        available_props  = [c for c in PROPORTION_COLS if c in df.columns]

        # --- Estatísticas descritivas ---
        st.subheader("Estatísticas Descritivas das Médias Municipais")
        stats_df = descriptive_stats(df, columns=available_scores)
        stats_df_fmt = stats_df.apply(
            lambda col: col.apply(lambda v: _fmt_br(v, 4) if pd.notna(v) else "–")
        )
        st.dataframe(stats_df_fmt, use_container_width=True)
        st.markdown("---")

        # ---- Histogramas ----
        st.subheader("Histogramas")

        all_quant_options = {score_label(c): c for c in available_scores + available_props}
        sel_score_label = st.selectbox(
            "Selecione a variável",
            list(all_quant_options.keys()),
            key="hist_select",
        )
        sel_score_col = all_quant_options[sel_score_label]

        sel_test = st.selectbox(
            "Teste de normalidade",
            NORMALITY_TESTS,
            key="norm_test_select",
        )

        _TEST_DESCRIPTIONS = {
            "Shapiro-Wilk": (
                "O **Shapiro-Wilk** avalia se uma amostra provém de uma população normalmente "
                "distribuída calculando a correlação entre os dados ordenados e os quantis esperados "
                "de uma normal. É considerado um dos testes mais poderosos para amostras pequenas "
                "(n < 2 000), mas perde sensibilidade em amostras muito grandes."
            ),
            "Anderson-Darling": (
                "O **Anderson-Darling** é uma versão aprimorada do teste de Kolmogorov-Smirnov que "
                "atribui maior peso às caudas da distribuição. Por isso, é especialmente eficaz para "
                "detectar desvios de normalidade nas regiões extremas dos dados. Retorna uma "
                "estatística que é comparada a valores críticos tabelados para diferentes níveis de "
                "significância."
            ),
            "D'Agostino-Pearson": (
                "O **D'Agostino-Pearson** combina medidas de assimetria (*skewness*) e curtose "
                "(*kurtosis*) em uma única estatística qui-quadrado com 2 graus de liberdade. É "
                "robusto para amostras de tamanho moderado a grande e detecta bem desvios tanto na "
                "forma simétrica quanto no achatamento da curva."
            ),
            "Jarque-Bera": (
                "O **Jarque-Bera** também baseia sua estatística na assimetria e na curtose, mas usa "
                "uma formulação ligeiramente diferente, sendo muito utilizado em econometria. É "
                "assintoticamente distribuído como qui-quadrado com 2 graus de liberdade, portanto "
                "funciona melhor com amostras grandes. Em amostras pequenas pode apresentar baixo "
                "poder estatístico."
            ),
            "Kolmogorov-Smirnov": (
                "O **Kolmogorov-Smirnov** (KS) mede a maior distância absoluta entre a função de "
                "distribuição empírica da amostra e a distribuição normal teórica parametrizada pela "
                "média e desvio-padrão observados. É um teste não-paramétrico de uso geral, embora "
                "seja menos sensível nas caudas do que o Anderson-Darling."
            ),
        }

        col_hist, col_info = st.columns([3, 1])

        with col_hist:
            hist_data = df[sel_score_col].dropna()
            fig_hist = px.histogram(
                hist_data,
                nbins=50,
                color_discrete_sequence=["#636EFA"],
                labels={"value": sel_score_label, "count": "Municípios"},
                title=f"Histograma – {sel_score_label} (distribuição municipal)",
            )
            mean_val   = hist_data.mean()
            median_val = hist_data.median()
            fig_hist.add_vline(x=mean_val, line_dash="dash", line_color="red",
                               annotation_text=f"Média: {_fmt_br(mean_val, 1)}",
                               annotation_position="top right")
            fig_hist.add_vline(x=median_val, line_dash="dot", line_color="green",
                               annotation_text=f"Mediana: {_fmt_br(median_val, 1)}",
                               annotation_position="top left")
            fig_hist.update_layout(height=420, xaxis_title=sel_score_label,
                                    yaxis_title="Municípios")
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_info:
            norm = normality_test(df, sel_score_col, test=sel_test)
            st.markdown("**Teste de Normalidade**")
            st.caption(_TEST_DESCRIPTIONS[sel_test])
            st.markdown(f"- Teste: {norm['teste']}")
            st.markdown(f"- Estatística: {_fmt_br(norm['estatística'], 4)}")
            st.markdown(f"- p-valor: {_fmt_br(norm['p_valor'], 6)}")
            resultado = "✅ Normal" if norm["normal"] else "❌ Não Normal"
            st.markdown(f"- Resultado: {resultado}")

            st.markdown("**Percentis**")
            for pct in [10, 25, 50, 75, 90]:
                st.markdown(f"- P{pct}: {_fmt_br(np.percentile(hist_data, pct), 1)}")

        st.markdown("---")

        # ---- Todos os histogramas de notas lado a lado ----
        st.subheader("Todos os Histogramas de Notas")
        cols = st.columns(len(available_scores))
        for i, score_col in enumerate(available_scores):
            with cols[i]:
                data_col = df[score_col].dropna()
                fig_mini = px.histogram(
                    data_col,
                    nbins=40,
                    title=score_label(score_col),
                    color_discrete_sequence=[PALETTE[i % len(PALETTE)]],
                    labels={"value": "Média Municipal"},
                )
                fig_mini.update_layout(height=280, showlegend=False,
                                       xaxis_title="Nota Média",
                                       yaxis_title="Municípios",
                                       margin=dict(l=20, r=10, t=50, b=30))
                st.plotly_chart(fig_mini, use_container_width=True)

        st.markdown("---")

        # ---- Box Plot – todas as notas ----
        st.subheader("Box Plots das Médias Municipais")

        melt_df = df[available_scores].melt(var_name="Área", value_name="Nota Média").dropna()
        melt_df["Área"] = melt_df["Área"].map(SCORE_LABELS)

        fig_box = px.box(
            melt_df,
            x="Área",
            y="Nota Média",
            color="Área",
            color_discrete_sequence=PALETTE,
            points=False,
            title="Box Plot – Médias Municipais por Área de Conhecimento",
        )
        fig_box.update_layout(height=480, showlegend=False,
                              xaxis_title="Área", yaxis_title="Nota Média Municipal")
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")

        # ---- Box plot por UF ----
        st.subheader("Box Plot por UF")
        score_opt_box = {score_label(c): c for c in available_scores}
        sel_score_box_lbl = st.selectbox("Área de conhecimento", list(score_opt_box.keys()),
                                          key="box_score_sel")
        sel_score_box_col = score_opt_box[sel_score_box_lbl]

        fig_box_uf = px.box(
            df_full.dropna(subset=[sel_score_box_col, "uf"]),
            x="uf",
            y=sel_score_box_col,
            color="uf",
            color_discrete_sequence=PALETTE,
            points=False,
            title=f"{sel_score_box_lbl} – distribuição municipal por UF",
        )
        fig_box_uf.update_layout(height=500, showlegend=False,
                                  xaxis_title="UF",
                                  yaxis_title="Nota Média Municipal")
        st.plotly_chart(fig_box_uf, use_container_width=True)

        with st.expander(f"📋 Médias ponderadas de {sel_score_box_lbl} por UF"):
            grp_table = mean_scores_by_group(df_full, sel_score_box_col, "uf")
            grp_table_fmt = grp_table.copy()
            grp_table_fmt["Média Ponderada"] = grp_table_fmt["Média Ponderada"].apply(
                lambda v: _fmt_br(v, 2) if pd.notna(v) else "–"
            )
            st.dataframe(grp_table_fmt, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ---- Médias ponderadas por UF (barras) ----
        st.subheader("Médias Ponderadas por UF")
        score_opt_grp = {score_label(c): c for c in available_scores}
        sel_grp_score = st.selectbox("Nota", list(score_opt_grp.keys()), key="quant_uf_score")
        sel_grp_col   = score_opt_grp[sel_grp_score]

        grp_means = mean_scores_by_group(df_full, sel_grp_col, "uf")

        fig_grp = px.bar(
            grp_means,
            x="Categoria",
            y="Média Ponderada",
            color="Média Ponderada",
            color_continuous_scale="Tealrose",
            text=[_fmt_br(v, 1) for v in grp_means["Média Ponderada"]],
            title=f"Média Ponderada de {sel_grp_score} por UF",
        )
        fig_grp.update_traces(texttemplate="%{text}", textposition="outside")
        fig_grp.update_layout(height=450, showlegend=False,
                               coloraxis_showscale=False,
                               xaxis_title="UF", yaxis_title="Nota Média Ponderada",
                               yaxis_range=[300, 600])
        st.plotly_chart(fig_grp, use_container_width=True)

    else:
        # ---- Modo amostragem ----
        _QUANT_SAMPLE_MAP = {
            "AAS – Amostragem Aleatória Simples": "aas_res",
            "Estratificada (por UF)":             "estratificada_res",
            "Sistemática":                        "sistematica_res",
        }
        _quant_sample_key = _QUANT_SAMPLE_MAP[_quant_modo]

        try:
            _qt_sampling = get_sampling_data(_db_config)
        except Exception as _exc:
            st.error(f"Erro ao carregar dados de amostragem: {_exc}")
            st.stop()

        _df_qt      = _qt_sampling[_quant_sample_key]
        _mom_res_qt = _qt_sampling["momentos_res"]
        _n_res_qt   = _qt_sampling["n_amostra_res"]

        st.markdown(
            f"Dados individuais amostrados da tabela **`ed_enem_2024_resultados`** "
            f"(N = {_fmt_br(_mom_res_qt['N'])} participantes com as 5 notas válidas) "
            f"pelo método **{_quant_modo}**."
        )

        if _df_qt.empty:
            st.warning("A amostra está vazia. Verifique os dados no banco.")
            st.stop()

        st.markdown(f"**n (amostra) = {_fmt_br(len(_df_qt))}**")
        st.markdown("---")

        _SCORE_AREA_COLS_QT = {
            "nota_cn":      "Ciências da Natureza",
            "nota_ch":      "Ciências Humanas",
            "nota_lc":      "Linguagens",
            "nota_mt":      "Matemática",
            "nota_redacao": "Redação",
            "nota_media":   "Nota Média",
        }
        _avail_qt = [c for c in _SCORE_AREA_COLS_QT if c in _df_qt.columns]

        # ---- Estatísticas descritivas da amostra ----
        st.subheader("Estatísticas Descritivas da Amostra")
        _qt_stats_rows = []
        for _c in _avail_qt:
            _v = _df_qt[_c].dropna()
            _qt_stats_rows.append({
                "Variável":       _SCORE_AREA_COLS_QT[_c],
                "n":              _fmt_br(len(_v)),
                "Média (x̄)":     _fmt_br(float(_v.mean()), 2),
                "Desvio Padrão": _fmt_br(float(_v.std()), 2),
                "Mínimo":        _fmt_br(float(_v.min()), 2),
                "Máximo":        _fmt_br(float(_v.max()), 2),
            })
        if _qt_stats_rows:
            st.dataframe(pd.DataFrame(_qt_stats_rows), use_container_width=True, hide_index=True)

        # Show comparison for nota_media vs population
        if "nota_media" in _df_qt.columns:
            _qtv = _df_qt["nota_media"].dropna()
            _qt_cmp = pd.DataFrame([
                {"Estatística": "N / n",        "População": _fmt_br(_mom_res_qt["N"]),                  "Amostra": _fmt_br(len(_qtv))},
                {"Estatística": "Média",        "População": _fmt_br(_mom_res_qt["media"], 4),           "Amostra": _fmt_br(float(_qtv.mean()), 4)},
                {"Estatística": "Desvio Padrão","População": _fmt_br(_mom_res_qt["desvio_padrao"], 4),   "Amostra": _fmt_br(float(_qtv.std()), 4)},
                {"Estatística": "Mínimo",       "População": _fmt_br(_mom_res_qt["minimo"], 2),          "Amostra": _fmt_br(float(_qtv.min()), 2)},
                {"Estatística": "Máximo",       "População": _fmt_br(_mom_res_qt["maximo"], 2),          "Amostra": _fmt_br(float(_qtv.max()), 2)},
            ])
            st.markdown("**Comparação Amostra × População (Nota Média)**")
            st.dataframe(_qt_cmp, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ---- Seletor de variável + histograma ----
        st.subheader("Histogramas")
        _qt_var_opts = {_SCORE_AREA_COLS_QT[c]: c for c in _avail_qt}
        _qt_sel_lbl = st.selectbox("Selecione a variável", list(_qt_var_opts.keys()), key="qt_hist_sel")
        _qt_sel_col = _qt_var_opts[_qt_sel_lbl]

        _qt_data = _df_qt[_qt_sel_col].dropna()
        _qt_mean = float(_qt_data.mean())
        _fig_qt  = px.histogram(
            _qt_data, nbins=50,
            title=f"Histograma – {_qt_sel_lbl} ({_quant_modo})",
            labels={"value": _qt_sel_lbl, "count": "Participantes"},
            color_discrete_sequence=["#636EFA"],
        )
        _fig_qt.add_vline(x=_qt_mean, line_dash="dash", line_color="blue",
                          annotation_text=f"x̄ = {_fmt_br(_qt_mean, 1)}",
                          annotation_position="top right")
        if _qt_sel_col == "nota_media":
            _fig_qt.add_vline(x=_mom_res_qt["media"], line_dash="dot", line_color="red",
                              annotation_text=f"μ pop. = {_fmt_br(_mom_res_qt['media'], 1)}",
                              annotation_position="top left")
        _fig_qt.update_layout(height=420, xaxis_title=_qt_sel_lbl, yaxis_title="Participantes")
        st.plotly_chart(_fig_qt, use_container_width=True)

        st.markdown("---")

        # ---- Todos os histogramas lado a lado ----
        st.subheader("Todos os Histogramas de Notas")
        _all_note_cols = [c for c in _avail_qt if c != "nota_media"]
        if _all_note_cols:
            _qt_cols = st.columns(len(_all_note_cols))
            for _i, _c in enumerate(_all_note_cols):
                with _qt_cols[_i]:
                    _d = _df_qt[_c].dropna()
                    _f = px.histogram(
                        _d, nbins=40,
                        title=_SCORE_AREA_COLS_QT[_c],
                        color_discrete_sequence=[PALETTE[_i % len(PALETTE)]],
                        labels={"value": "Nota"},
                    )
                    _f.update_layout(height=280, showlegend=False,
                                     xaxis_title="Nota", yaxis_title="Participantes",
                                     margin=dict(l=20, r=10, t=50, b=30))
                    st.plotly_chart(_f, use_container_width=True)

        st.markdown("---")

        # ---- Box Plots ----
        st.subheader("Box Plots das Notas")
        if _all_note_cols:
            _melt_qt = _df_qt[_all_note_cols].melt(var_name="Área", value_name="Nota").dropna()
            _melt_qt["Área"] = _melt_qt["Área"].map(_SCORE_AREA_COLS_QT)
            _fig_box_qt = px.box(
                _melt_qt, x="Área", y="Nota", color="Área",
                color_discrete_sequence=PALETTE, points=False,
                title=f"Box Plot – Notas por Área ({_quant_modo})",
            )
            _fig_box_qt.update_layout(height=460, showlegend=False,
                                       xaxis_title="Área", yaxis_title="Nota")
            st.plotly_chart(_fig_box_qt, use_container_width=True)


# ===========================================================================
# PÁGINA 4 – ANÁLISE DE CORRELAÇÃO
# ===========================================================================

elif page == "🔗 Análise de Correlação":
    st.title("🔗 Análise de Correlação")
    st.markdown(
        "Correlação entre as médias municipais de notas e os indicadores "
        "socioeconômicos e demográficos."
    )
    st.markdown("---")

    available_scores = [c for c in SCORE_COLS if c in df.columns]
    available_props  = [c for c in PROPORTION_COLS if c in df.columns]
    all_num_cols     = available_scores + available_props

    # ---- Heatmap ----
    st.subheader("Matriz de Correlação (Heatmap)")

    method = st.selectbox(
        "Método de correlação",
        ["pearson", "spearman", "kendall"],
        format_func=lambda m: m.capitalize(),
    )

    heatmap_cols = [c for c in HEATMAP_COLS if c in df.columns]
    corr_df      = correlation_matrix(df, columns=heatmap_cols, method=method)

    corr_values = corr_df.values.tolist()
    labels      = corr_df.columns.tolist()

    fig_heat = ff.create_annotated_heatmap(
        z=corr_values,
        x=labels,
        y=labels,
        colorscale="RdBu",
        reversescale=True,
        zmin=-1,
        zmax=1,
        annotation_text=[[_fmt_br(v, 2) for v in row] for row in corr_values],
        showscale=True,
    )
    fig_heat.update_layout(
        height=max(500, len(labels) * 45),
        title=f"Correlação de {method.capitalize()} entre notas e indicadores municipais",
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("📋 Ver tabela da matriz de correlação"):
        corr_df_fmt = corr_df.apply(
            lambda col: col.apply(lambda v: _fmt_br(v, 4) if pd.notna(v) else "–")
        )
        st.dataframe(corr_df_fmt, use_container_width=True)

    st.markdown("---")

    # ---- Scatter matrix de notas ----
    st.subheader("Scatter Matrix das Notas")
    st.markdown(
        "_Distribuição par a par das médias municipais de notas. "
        "Colorida por UF. Usa amostra de até 2000 municípios para performance._"
    )

    score_options_multi = {score_label(c): c for c in available_scores}
    sel_pair_labels = st.multiselect(
        "Áreas de conhecimento",
        list(score_options_multi.keys()),
        default=list(score_options_multi.keys()),
    )

    if len(sel_pair_labels) >= 2:
        sel_pair_cols = [score_options_multi[lbl] for lbl in sel_pair_labels]
        sample_df     = df[sel_pair_cols + ["uf"]].dropna().sample(
            min(2000, len(df)), random_state=42
        )
        renamed = {c: score_label(c) for c in sel_pair_cols}
        sample_df = sample_df.rename(columns=renamed)

        fig_scatter = px.scatter_matrix(
            sample_df,
            dimensions=list(renamed.values()),
            color="uf",
            color_discrete_sequence=PALETTE,
            title="Scatter Matrix das Médias Municipais de Notas (por UF)",
            opacity=0.5,
        )
        fig_scatter.update_traces(marker=dict(size=3))
        fig_scatter.update_layout(height=700)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Correlation table for selected variables
        pair_corr = df[sel_pair_cols].corr(method=method).round(4)
        pair_corr = pair_corr.rename(index=renamed, columns=renamed)
        with st.expander("📋 Tabela de Correlação entre as variáveis selecionadas"):
            pair_corr_fmt = pair_corr.apply(
                lambda col: col.apply(lambda v: _fmt_br(v, 4) if pd.notna(v) else "–")
            )
            st.dataframe(pair_corr_fmt, use_container_width=True)
    else:
        st.info("Selecione pelo menos 2 áreas de conhecimento para exibir o gráfico.")

    st.markdown("---")

    # ---- Scatter indicador × nota ----
    st.subheader("Dispersão: Indicador Demográfico × Nota")
    st.markdown(
        "Visualize a relação entre um indicador socioeconômico/demográfico "
        "e a média de uma área de conhecimento por município."
    )

    prop_options  = {PROPORTION_LABELS.get(c, c): c for c in available_props}
    score_options = {score_label(c): c for c in available_scores}

    if prop_options and score_options:
        col_sc1, col_sc2 = st.columns(2)
        sel_x_lbl = col_sc1.selectbox("Indicador (eixo X)", list(prop_options.keys()),
                                       key="sc_x")
        sel_y_lbl = col_sc2.selectbox("Nota (eixo Y)", list(score_options.keys()),
                                       key="sc_y")
        sel_x_col = prop_options[sel_x_lbl]
        sel_y_col = score_options[sel_y_lbl]

        scatter_df = df[[sel_x_col, sel_y_col, "municipio", "total_inscritos"]].dropna().copy()
        # Pre-format values for hover labels (Brazilian number format)
        scatter_df["_x_fmt"]   = scatter_df[sel_x_col].apply(lambda v: _fmt_br(v, 3))
        scatter_df["_y_fmt"]   = scatter_df[sel_y_col].apply(lambda v: _fmt_br(v, 1))
        scatter_df["_ins_fmt"] = scatter_df["total_inscritos"].apply(_fmt_br)
        # Add OLS trend line without requiring statsmodels
        x_vals    = scatter_df[sel_x_col].values
        y_vals    = scatter_df[sel_y_col].values
        poly_coef = np.polyfit(x_vals, y_vals, 1)
        x_range   = np.linspace(x_vals.min(), x_vals.max(), 200)
        fig_sc = go.Figure()
        fig_sc.add_scatter(
            x=scatter_df[sel_x_col],
            y=scatter_df[sel_y_col],
            mode="markers",
            marker=dict(color="#636EFA", size=4, opacity=0.45),
            customdata=scatter_df[["municipio", "total_inscritos", "_x_fmt", "_y_fmt", "_ins_fmt"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + f"{sel_x_lbl}: " + "%{customdata[2]}<br>"
                + f"{sel_y_lbl}: " + "%{customdata[3]}<br>"
                "Inscritos: %{customdata[4]}<extra></extra>"
            ),
            name="Municípios",
            showlegend=False,
        )
        fig_sc.add_scatter(
            x=x_range, y=np.polyval(poly_coef, x_range),
            mode="lines", line=dict(color="black", width=2, dash="dash"),
            name="Tendência (OLS)", showlegend=False,
        )
        fig_sc.update_layout(
            title=f"{sel_y_lbl} × {sel_x_lbl} (por município)",
            xaxis_title=sel_x_lbl,
            yaxis_title=sel_y_lbl,
            height=520,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        # Correlação pontual
        corr_val = scatter_df[[sel_x_col, sel_y_col]].corr(method=method).iloc[0, 1]
        st.markdown(
            f"**Correlação de {method.capitalize()} entre "
            f"_{sel_x_lbl}_ e _{sel_y_lbl}_: `{_fmt_br(corr_val, 4)}`**"
        )
    else:
        st.info("Colunas de proporção não disponíveis. Verifique se os dados foram carregados corretamente.")


# ===========================================================================
# PÁGINA 5 – AMOSTRAGEM ESTATÍSTICA
# ===========================================================================

elif page == "🔬 Amostragem":
    st.title("🔬 Amostragem Estatística")
    st.markdown(
        "Dimensionamento amostral e comparação dos métodos de amostragem "
        "aplicados às duas tabelas do banco de dados do ENEM 2024.\n\n"
        "* **`ed_enem_2024_participantes`** – todos os inscritos (inclui ausentes e treineiros)\n"
        "* **`ed_enem_2024_resultados`** – participantes **com as 5 notas válidas** "
        "(realizaram todas as provas)"
    )
    st.markdown("---")

    # Load sampling data (cached)
    try:
        sampling = get_sampling_data(_db_config)
    except Exception as exc:
        st.error(f"Erro ao carregar dados de amostragem: {exc}")
        st.stop()

    # Unpack resultados
    mom_res = sampling["momentos_res"]
    n_res   = sampling["n_amostra_res"]
    k_res   = sampling["k_res"]
    Z       = sampling["Z"]
    E_res   = sampling["E_res"]

    # Unpack participantes
    mom_part = sampling["momentos_part"]
    n_part   = sampling["n_amostra_part"]
    k_part   = sampling["k_part"]
    E_part   = sampling["E_part"]

    # Population class distributions
    df_cor_raca_pop = sampling["cor_raca_pop"]
    df_sexo_pop     = sampling["sexo_pop"]

    # ---- 1. Parâmetros Populacionais ----
    st.subheader("1. Parâmetros Populacionais")

    _pop_col_part, _pop_col_res = st.columns(2)

    with _pop_col_part:
        st.markdown("#### Participantes (todos os inscritos)")
        _c1, _c2, _c3 = st.columns(3)
        _c1.metric("N", _fmt_br(mom_part["N"]))
        _c2.metric("Média de Idade", _fmt_br(mom_part["media"], 1))
        _c3.metric("Desvio Padrão", _fmt_br(mom_part["desvio_padrao"], 2))
        with st.expander("📋 Parâmetros detalhados"):
            st.dataframe(
                pd.DataFrame([
                    {"Parâmetro": "N (inscritos)",         "Símbolo": "N",   "Valor": _fmt_br(mom_part["N"])},
                    {"Parâmetro": "Média de Idade",        "Símbolo": "μ",   "Valor": _fmt_br(mom_part["media"], 4)},
                    {"Parâmetro": "Variância da Idade",    "Símbolo": "σ²",  "Valor": _fmt_br(mom_part["variancia"], 4)},
                    {"Parâmetro": "Desvio Padrão da Idade","Símbolo": "σ",   "Valor": _fmt_br(mom_part["desvio_padrao"], 4)},
                    {"Parâmetro": "Idade mínima",          "Símbolo": "min", "Valor": _fmt_br(mom_part["minimo"], 0)},
                    {"Parâmetro": "Idade máxima",          "Símbolo": "max", "Valor": _fmt_br(mom_part["maximo"], 0)},
                ]),
                use_container_width=True, hide_index=True,
            )

    with _pop_col_res:
        st.markdown("#### Resultados (5 notas válidas)")
        _c1, _c2, _c3 = st.columns(3)
        _c1.metric("N", _fmt_br(mom_res["N"]))
        _c2.metric("Média da Nota", _fmt_br(mom_res["media"], 2))
        _c3.metric("Desvio Padrão", _fmt_br(mom_res["desvio_padrao"], 2))
        with st.expander("📋 Parâmetros detalhados"):
            st.dataframe(
                pd.DataFrame([
                    {"Parâmetro": "N (com 5 notas válidas)", "Símbolo": "N",   "Valor": _fmt_br(mom_res["N"])},
                    {"Parâmetro": "Média da Nota",           "Símbolo": "μ",   "Valor": _fmt_br(mom_res["media"], 4)},
                    {"Parâmetro": "Variância",               "Símbolo": "σ²",  "Valor": _fmt_br(mom_res["variancia"], 4)},
                    {"Parâmetro": "Desvio Padrão",           "Símbolo": "σ",   "Valor": _fmt_br(mom_res["desvio_padrao"], 4)},
                    {"Parâmetro": "Mínimo",                  "Símbolo": "min", "Valor": _fmt_br(mom_res["minimo"], 2)},
                    {"Parâmetro": "Máximo",                  "Símbolo": "max", "Valor": _fmt_br(mom_res["maximo"], 2)},
                ]),
                use_container_width=True, hide_index=True,
            )

    st.markdown("---")

    # ---- 2. Cálculo do tamanho amostral ----
    st.subheader("2. Cálculo do Tamanho Amostral Mínimo")
    st.markdown("Fórmula de Cochran com correção para população finita (95 % de confiança).")

    _n0_part = (Z ** 2 * mom_part["desvio_padrao"] ** 2) / (E_part ** 2)
    _n0_res  = (Z ** 2 * mom_res["desvio_padrao"] ** 2)  / (E_res  ** 2)

    _ncol_part, _ncol_res = st.columns(2)
    with _ncol_part:
        st.markdown(
            f"**Participantes** _(variável: idade)_\n\n"
            f"| Parâmetro | Valor |\n"
            f"|-----------|-------|\n"
            f"| Nível de confiança | 95 % |\n"
            f"| Z | {_fmt_br(Z, 2)} |\n"
            f"| Erro amostral (E) | {_fmt_br(E_part, 1)} {'ano' if E_part < 2 else 'anos'} |\n"
            f"| Desvio padrão (σ) | {_fmt_br(mom_part['desvio_padrao'], 2)} |\n"
            f"| N | {_fmt_br(mom_part['N'])} |\n\n"
            f"n₀ = (Z² × σ²) / E² = {_fmt_br(_n0_part, 1)}\n\n"
            f"**n = {_fmt_br(n_part)}**  |  k = {_fmt_br(k_part)}"
        )
        st.metric("Tamanho da amostra (n)", _fmt_br(n_part))
        st.metric("Intervalo sistemático (k)", _fmt_br(k_part))

    with _ncol_res:
        st.markdown(
            f"**Resultados** _(variável: nota média)_\n\n"
            f"| Parâmetro | Valor |\n"
            f"|-----------|-------|\n"
            f"| Nível de confiança | 95 % |\n"
            f"| Z | {_fmt_br(Z, 2)} |\n"
            f"| Erro amostral (E) | {_fmt_br(E_res, 1)} pontos |\n"
            f"| Desvio padrão (σ) | {_fmt_br(mom_res['desvio_padrao'], 2)} |\n"
            f"| N | {_fmt_br(mom_res['N'])} |\n\n"
            f"n₀ = (Z² × σ²) / E² = {_fmt_br(_n0_res, 1)}\n\n"
            f"**n = {_fmt_br(n_res)}**  |  k = {_fmt_br(k_res)}"
        )
        st.metric("Tamanho da amostra (n)", _fmt_br(n_res))
        st.metric("Intervalo sistemático (k)", _fmt_br(k_res))

    st.markdown("---")

    # ---- 3. Comparação Amostra × População ----
    st.subheader("3. Comparação Amostra × População")

    _SCORE_AREA_COLS = {
        "nota_cn":      "Ciências da Natureza",
        "nota_ch":      "Ciências Humanas",
        "nota_lc":      "Linguagens",
        "nota_mt":      "Matemática",
        "nota_redacao": "Redação",
    }

    # ---------- helpers: class-count comparison ----------

    def _class_comparison_chart(
        df_sample: pd.DataFrame,
        col: str,
        df_pop: pd.DataFrame,
        pop_col: str,
        pop_count_col: str,
        title: str,
    ) -> None:
        """Bar chart comparing class % in population vs sample."""
        if col not in df_sample.columns:
            return
        sample_counts = (
            df_sample[col].fillna("Não declarado")
            .value_counts()
            .reset_index()
        )
        sample_counts.columns = [col, "sample_count"]
        total_sample = sample_counts["sample_count"].sum()
        sample_counts["sample_%"] = sample_counts["sample_count"] / total_sample * 100

        _pop = df_pop.rename(columns={pop_col: col, pop_count_col: "pop_count"}).copy()
        pop_total = _pop["pop_count"].sum()
        _pop["pop_%"] = _pop["pop_count"] / pop_total * 100

        merged = pd.merge(
            _pop[[col, "pop_%"]], sample_counts[[col, "sample_%"]], on=col, how="outer"
        ).fillna(0)

        _fig = go.Figure()
        _fig.add_bar(x=merged[col], y=merged["pop_%"],    name="População (%)", marker_color="#636EFA")
        _fig.add_bar(x=merged[col], y=merged["sample_%"], name="Amostra (%)",   marker_color="#EF553B")
        _fig.update_layout(
            title=title,
            barmode="group",
            xaxis_title=col.replace("_", " ").title(),
            yaxis_title="Proporção (%)",
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(_fig, use_container_width=True)

    def _class_comparison_table(
        df_sample: pd.DataFrame,
        col: str,
        df_pop: pd.DataFrame,
        pop_col: str,
        pop_count_col: str,
    ) -> None:
        """Table comparing class counts and % between population and sample."""
        if col not in df_sample.columns:
            return
        sample_counts = (
            df_sample[col].fillna("Não declarado")
            .value_counts()
            .reset_index()
        )
        sample_counts.columns = [col, "Amostra (n)"]
        sample_counts["Amostra (%)"] = (
            sample_counts["Amostra (n)"] / sample_counts["Amostra (n)"].sum() * 100
        ).round(2)

        _pop = df_pop.rename(columns={pop_col: col, pop_count_col: "Pop. (N)"}).copy()
        _pop["Pop. (%)"] = (_pop["Pop. (N)"] / _pop["Pop. (N)"].sum() * 100).round(2)

        merged = pd.merge(
            _pop[[col, "Pop. (N)", "Pop. (%)"]],
            sample_counts[[col, "Amostra (n)", "Amostra (%)"]],
            on=col, how="outer",
        ).fillna(0)
        merged["Amostra (n)"] = merged["Amostra (n)"].astype(int)
        merged["Pop. (N)"]    = merged["Pop. (N)"].astype(int)
        merged = merged.sort_values("Pop. (%)", ascending=False)
        st.dataframe(merged, use_container_width=True, hide_index=True)

    # ---------- render helpers ----------

    def _render_sample_tab_res(label: str, df_s: pd.DataFrame, descricao: str) -> None:
        """Renders one sampling method tab for ed_enem_2024_resultados."""
        st.markdown(descricao)
        if df_s.empty:
            st.warning("A amostra está vazia. Verifique os dados no banco.")
            return

        vals = df_s["nota_media"].dropna()
        stats = {
            "n":             len(vals),
            "media":         float(vals.mean()),
            "desvio_padrao": float(vals.std()),
            "minimo":        float(vals.min()),
            "maximo":        float(vals.max()),
        }

        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("n (amostra)", _fmt_br(stats["n"]))
        _m2.metric(
            "Média (x̄)", _fmt_br(stats["media"], 2),
            f"{_fmt_br(stats['media'] - mom_res['media'], 2)} vs μ pop.",
        )
        _m3.metric(
            "Desvio Padrão (s)", _fmt_br(stats["desvio_padrao"], 2),
            f"{_fmt_br(stats['desvio_padrao'] - mom_res['desvio_padrao'], 2)} vs σ pop.",
        )
        _rel_err = abs(stats["media"] - mom_res["media"]) / mom_res["media"] * 100
        _m4.metric("Erro Relativo da Média", f"{_fmt_br(_rel_err, 4)}%")

        _cmp_df = pd.DataFrame([
            {"Estatística": "N / n",        "População": _fmt_br(mom_res["N"]),                "Amostra": _fmt_br(stats["n"])},
            {"Estatística": "Média",        "População": _fmt_br(mom_res["media"], 4),         "Amostra": _fmt_br(stats["media"], 4)},
            {"Estatística": "Desvio Padrão","População": _fmt_br(mom_res["desvio_padrao"], 4), "Amostra": _fmt_br(stats["desvio_padrao"], 4)},
            {"Estatística": "Mínimo",       "População": _fmt_br(mom_res["minimo"], 2),        "Amostra": _fmt_br(stats["minimo"], 2)},
            {"Estatística": "Máximo",       "População": _fmt_br(mom_res["maximo"], 2),        "Amostra": _fmt_br(stats["maximo"], 2)},
        ])
        st.dataframe(_cmp_df, use_container_width=True, hide_index=True)

        _hist_col, _area_col = st.columns(2)
        with _hist_col:
            _fig_h = px.histogram(
                df_s["nota_media"].dropna(), nbins=50,
                title=f"Distribuição da nota média – {label}",
                labels={"value": "Nota Média", "count": "Participantes"},
                color_discrete_sequence=["#636EFA"],
            )
            _fig_h.add_vline(
                x=stats["media"], line_dash="dash", line_color="blue",
                annotation_text=f"x̄ = {_fmt_br(stats['media'], 1)}",
                annotation_position="top right",
            )
            _fig_h.add_vline(
                x=mom_res["media"], line_dash="dot", line_color="red",
                annotation_text=f"μ = {_fmt_br(mom_res['media'], 1)}",
                annotation_position="top left",
            )
            _fig_h.update_layout(height=380, xaxis_title="Nota Média", yaxis_title="Participantes")
            st.plotly_chart(_fig_h, use_container_width=True)

        with _area_col:
            _avail = {
                lbl: float(df_s[col].dropna().mean())
                for col, lbl in _SCORE_AREA_COLS.items()
                if col in df_s.columns
            }
            if _avail:
                _areas_df = pd.DataFrame({
                    "Área":           list(_avail.keys()),
                    "Média Amostral": list(_avail.values()),
                })
                _fig_a = px.bar(
                    _areas_df, x="Área", y="Média Amostral",
                    text=[_fmt_br(v, 1) for v in _areas_df["Média Amostral"]],
                    color_discrete_sequence=["#00CC96"],
                    title=f"Média por área – {label}",
                )
                _fig_a.update_traces(texttemplate="%{text}", textposition="outside")
                _fig_a.update_layout(
                    height=380, xaxis_title="Área", yaxis_title="Nota Média", showlegend=False
                )
                st.plotly_chart(_fig_a, use_container_width=True)

        # Class comparison: top municipalities
        if "municipio" in df_s.columns:
            with st.expander("📋 Comparação por município (top 20 na amostra)"):
                _mun = df_s["municipio"].value_counts().reset_index()
                _mun.columns = ["Município", "n amostra"]
                _mun["% amostra"] = (_mun["n amostra"] / _mun["n amostra"].sum() * 100).round(2)
                st.dataframe(_mun.head(20), use_container_width=True, hide_index=True)

    def _render_sample_tab_part(label: str, df_s: pd.DataFrame, descricao: str) -> None:
        """Renders one sampling method tab for ed_enem_2024_participantes."""
        st.markdown(descricao)
        if df_s.empty:
            st.warning("A amostra está vazia. Verifique os dados no banco.")
            return

        vals = df_s["idade"].dropna()
        stats = {
            "n":             len(vals),
            "media":         float(vals.mean()),
            "desvio_padrao": float(vals.std()),
            "minimo":        float(vals.min()),
            "maximo":        float(vals.max()),
        }

        _m1, _m2, _m3, _m4 = st.columns(4)
        _m1.metric("n (amostra)", _fmt_br(stats["n"]))
        _m2.metric(
            "Média de Idade (x̄)", _fmt_br(stats["media"], 1),
            f"{_fmt_br(stats['media'] - mom_part['media'], 2)} vs μ pop.",
        )
        _m3.metric(
            "Desvio Padrão (s)", _fmt_br(stats["desvio_padrao"], 2),
            f"{_fmt_br(stats['desvio_padrao'] - mom_part['desvio_padrao'], 2)} vs σ pop.",
        )
        _rel_err = abs(stats["media"] - mom_part["media"]) / mom_part["media"] * 100
        _m4.metric("Erro Relativo da Média", f"{_fmt_br(_rel_err, 4)}%")

        _cmp_df = pd.DataFrame([
            {"Estatística": "N / n",         "População": _fmt_br(mom_part["N"]),                "Amostra": _fmt_br(stats["n"])},
            {"Estatística": "Média de Idade","População": _fmt_br(mom_part["media"], 4),         "Amostra": _fmt_br(stats["media"], 4)},
            {"Estatística": "Desvio Padrão", "População": _fmt_br(mom_part["desvio_padrao"], 4), "Amostra": _fmt_br(stats["desvio_padrao"], 4)},
            {"Estatística": "Mínimo",        "População": _fmt_br(mom_part["minimo"], 0),        "Amostra": _fmt_br(stats["minimo"], 0)},
            {"Estatística": "Máximo",        "População": _fmt_br(mom_part["maximo"], 0),        "Amostra": _fmt_br(stats["maximo"], 0)},
        ])
        st.dataframe(_cmp_df, use_container_width=True, hide_index=True)

        _hist_col, _pie_col = st.columns(2)
        with _hist_col:
            _fig_h = px.histogram(
                df_s["idade"].dropna(), nbins=40,
                title=f"Distribuição de Idade – {label}",
                labels={"value": "Idade", "count": "Participantes"},
                color_discrete_sequence=["#636EFA"],
            )
            _fig_h.add_vline(
                x=stats["media"], line_dash="dash", line_color="blue",
                annotation_text=f"x̄ = {_fmt_br(stats['media'], 1)}",
                annotation_position="top right",
            )
            _fig_h.add_vline(
                x=mom_part["media"], line_dash="dot", line_color="red",
                annotation_text=f"μ = {_fmt_br(mom_part['media'], 1)}",
                annotation_position="top left",
            )
            _fig_h.update_layout(height=380, xaxis_title="Idade", yaxis_title="Participantes")
            st.plotly_chart(_fig_h, use_container_width=True)

        with _pie_col:
            if "cor_raca" in df_s.columns:
                _rc = df_s["cor_raca"].fillna("Não declarado").value_counts().reset_index()
                _rc.columns = ["Cor/Raça", "n"]
                _fig_r = px.pie(
                    _rc, names="Cor/Raça", values="n",
                    title=f"Cor/Raça na amostra – {label}", height=380,
                )
                st.plotly_chart(_fig_r, use_container_width=True)

        # Class comparison: cor/raça vs population
        st.markdown("**Comparação por Cor/Raça: Amostra × População**")
        _class_comparison_chart(
            df_s, "cor_raca", df_cor_raca_pop, "cor_raca", "pop_count",
            f"Proporção por Cor/Raça – {label} vs. População",
        )
        _class_comparison_table(df_s, "cor_raca", df_cor_raca_pop, "cor_raca", "pop_count")

        # Class comparison: sexo vs population (table only)
        if "sexo" in df_s.columns:
            st.markdown("**Comparação por Sexo: Amostra × População**")
            _class_comparison_table(df_s, "sexo", df_sexo_pop, "sexo", "pop_count")

    # ---------- outer tabs: one per table ----------

    _tab_part, _tab_res = st.tabs([
        "👥 Tabela: Participantes",
        "📝 Tabela: Resultados",
    ])

    with _tab_part:
        st.markdown(
            f"Amostras da tabela **`ed_enem_2024_participantes`** "
            f"(N = {_fmt_br(mom_part['N'])} inscritos). "
            "Variável de interesse: **idade**. "
            "Amostragem estratificada por **cor/raça** (`tp_cor_raca`)."
        )
        _ptab_aas, _ptab_est, _ptab_sis = st.tabs([
            "📊 AAS – Amostragem Aleatória Simples",
            "🗂️ Estratificada (por cor/raça)",
            "📐 Sistemática",
        ])
        with _ptab_aas:
            _render_sample_tab_part(
                "AAS (Participantes)",
                sampling["aas_part"],
                f"**Amostragem Aleatória Simples (AAS):** {_fmt_br(n_part)} participantes "
                f"selecionados de forma completamente aleatória dentre os {_fmt_br(mom_part['N'])} inscritos.",
            )
        with _ptab_est:
            _render_sample_tab_part(
                "Estratificada (Participantes)",
                sampling["estratificada_part"],
                f"**Amostragem Estratificada** por _cor/raça_ (`tp_cor_raca`): "
                f"a amostra de {_fmt_br(n_part)} participantes é distribuída "
                "proporcionalmente entre os estratos de cor/raça, "
                "garantindo representatividade racial.",
            )
        with _ptab_sis:
            _render_sample_tab_part(
                "Sistemática (Participantes)",
                sampling["sistematica_part"],
                f"**Amostragem Sistemática:** intervalo k = {_fmt_br(k_part)}. "
                "A população é ordenada por município; "
                f"seleciona-se cada k-ésimo inscrito, gerando uma amostra de até {_fmt_br(n_part)}.",
            )

    with _tab_res:
        st.markdown(
            f"Amostras da tabela **`ed_enem_2024_resultados`** "
            f"(N = {_fmt_br(mom_res['N'])} participantes com as 5 notas válidas). "
            "Variável de interesse: **nota média** _(CN + CH + LC + MT + Redação) / 5_. "
            "Amostragem estratificada por **município** (`co_municipio_prova`)."
        )
        _rtab_aas, _rtab_est, _rtab_sis = st.tabs([
            "📊 AAS – Amostragem Aleatória Simples",
            "🗂️ Estratificada (por município)",
            "📐 Sistemática",
        ])
        with _rtab_aas:
            _render_sample_tab_res(
                "AAS (Resultados)",
                sampling["aas_res"],
                f"**Amostragem Aleatória Simples (AAS):** {_fmt_br(n_res)} participantes "
                "selecionados de forma completamente aleatória da sub-população com notas válidas.",
            )
        with _rtab_est:
            _render_sample_tab_res(
                "Estratificada (Resultados)",
                sampling["estratificada_res"],
                f"**Amostragem Estratificada** por _município_ (`co_municipio_prova`): "
                f"a amostra de {_fmt_br(n_res)} participantes é distribuída "
                "proporcionalmente entre os estratos municipais, "
                "garantindo representatividade geográfica.",
            )
        with _rtab_sis:
            _render_sample_tab_res(
                "Sistemática (Resultados)",
                sampling["sistematica_res"],
                f"**Amostragem Sistemática:** intervalo k = {_fmt_br(k_res)}. "
                "A população é ordenada por município; "
                f"seleciona-se cada k-ésimo elemento, gerando uma amostra de até {_fmt_br(n_res)} participantes.",
            )
