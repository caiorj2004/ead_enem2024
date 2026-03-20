"""
app.py – Dashboard ENEM 2024 (Streamlit)

Páginas disponíveis no menu lateral:
  🏠 Visão Geral          → métricas resumidas e prévia dos dados
  📊 Variáveis Qualitativas → tabelas de frequência + gráficos de barra/pizza
  📈 Variáveis Quantitativas → histogramas e box plots das notas
  🔗 Análise de Correlação  → heatmap e scatter matrix das notas
"""

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    QUALITATIVE_VARS,
    QUANTITATIVE_VARS,
    SCORE_LABELS,
    apply_labels,
    correlation_matrix,
    descriptive_stats,
    frequency_table,
    mean_scores_by_group,
    normality_test,
)
from database import load_data

# Suppress only Plotly/pandas FutureWarnings that are known to be harmless
warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")


# ---------------------------------------------------------------------------
# Leitura das credenciais do banco de dados (st.secrets)
# ---------------------------------------------------------------------------

def _get_db_config() -> dict | None:
    """
    Lê as credenciais do banco PostgreSQL a partir de st.secrets["database"].

    Retorna None quando as credenciais não estiverem configuradas
    (modo template / demonstração).

    As chaves esperadas em secrets.toml são:
        host, port, dbname, user, password, table (opcional)
    """
    try:
        section = st.secrets["database"]
        # Retorna None se ainda forem os valores de placeholder
        if str(section.get("host", "")).upper().startswith("SEU_"):
            return None
        return dict(section)
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

@st.cache_data(show_spinner="Carregando dados do ENEM 2024…")
def get_data(db_config: dict | None) -> tuple[pd.DataFrame, bool]:
    df, is_demo = load_data(db_config)
    df = apply_labels(df)
    return df, is_demo


_db_config = _get_db_config()
df_full, _is_demo = get_data(_db_config)

# ---------------------------------------------------------------------------
# Sidebar – navegação + filtros globais
# ---------------------------------------------------------------------------

st.sidebar.title("📚 ENEM 2024")
st.sidebar.markdown("---")

PAGES = [
    "🏠 Visão Geral",
    "📊 Variáveis Qualitativas",
    "📈 Variáveis Quantitativas",
    "🔗 Análise de Correlação",
]
page = st.sidebar.radio("Navegação", PAGES)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros globais")

# Filtro por UF
all_ufs = sorted(df_full["SG_UF_RESIDENCIA"].dropna().unique().tolist())
sel_ufs = st.sidebar.multiselect("UF de Residência", all_ufs, default=all_ufs)

# Filtro por tipo de escola
escola_opts = df_full["ESCOLA_DESC"].dropna().unique().tolist()
sel_escola = st.sidebar.multiselect("Tipo de Escola", escola_opts, default=escola_opts)

# Filtro por sexo
sexo_opts = df_full["SEXO_DESC"].dropna().unique().tolist()
sel_sexo = st.sidebar.multiselect("Sexo", sexo_opts, default=sexo_opts)

# Aplica filtros
df = df_full[
    df_full["SG_UF_RESIDENCIA"].isin(sel_ufs)
    & df_full["ESCOLA_DESC"].isin(sel_escola)
    & df_full["SEXO_DESC"].isin(sel_sexo)
].copy()

st.sidebar.markdown(f"**Registros filtrados:** {len(df):,}")

# ---------------------------------------------------------------------------
# Banner de modo template (visível em todas as páginas)
# ---------------------------------------------------------------------------

if _is_demo:
    st.info(
        "⚙️ **Modo template ativo** – os dados exibidos são **sintéticos** e servem "
        "apenas para demonstração da estrutura do dashboard. "
        "Para conectar ao banco de dados real, configure as credenciais em "
        "*Streamlit Cloud → Settings → Secrets* seguindo o modelo em "
        "`.streamlit/secrets.toml`.",
        icon="🔧",
    )

# ---------------------------------------------------------------------------
# Paleta de cores padrão
# ---------------------------------------------------------------------------

PALETTE = px.colors.qualitative.Plotly

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def score_label(col: str) -> str:
    return SCORE_LABELS.get(col, col)


def card_metric(col, label, value, delta=None):
    with col:
        st.metric(label=label, value=value, delta=delta)


# ===========================================================================
# PÁGINA 1 – VISÃO GERAL
# ===========================================================================

if page == "🏠 Visão Geral":
    st.title("🏠 Visão Geral – ENEM 2024")
    st.markdown(
        "Este dashboard apresenta análises exploratórias dos microdados do **ENEM 2024**. "
        "Use os filtros no painel lateral para refinar a visualização."
    )
    st.markdown("---")

    # --- Métricas principais ---
    total = len(df)
    pct_fem = df["SEXO_DESC"].value_counts(normalize=True).get("Feminino", 0) * 100
    pct_pub = df["ESCOLA_DESC"].value_counts(normalize=True).get("Pública", 0) * 100
    media_geral = df[QUANTITATIVE_VARS].mean().mean()

    c1, c2, c3, c4 = st.columns(4)
    card_metric(c1, "Total de Participantes", f"{total:,}")
    card_metric(c2, "% Feminino", f"{pct_fem:.1f}%")
    card_metric(c3, "% Escola Pública", f"{pct_pub:.1f}%")
    card_metric(c4, "Média Geral (todas as provas)", f"{media_geral:.1f}")

    st.markdown("---")

    # --- Distribuição por UF (top 10) ---
    col_map, col_age = st.columns(2)

    with col_map:
        st.subheader("Participantes por UF")
        uf_counts = df["SG_UF_RESIDENCIA"].value_counts().reset_index()
        uf_counts.columns = ["UF", "Participantes"]
        fig_uf = px.bar(
            uf_counts.head(27),
            x="UF",
            y="Participantes",
            color="Participantes",
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_uf.update_layout(height=380, showlegend=False, xaxis_title="UF",
                              yaxis_title="Participantes",
                              coloraxis_showscale=False)
        st.plotly_chart(fig_uf, use_container_width=True)

    with col_age:
        st.subheader("Distribuição de Idade")
        fig_age = px.histogram(
            df,
            x="NU_IDADE",
            nbins=40,
            color_discrete_sequence=["#636EFA"],
            labels={"NU_IDADE": "Idade"},
        )
        fig_age.update_layout(height=380, bargap=0.05,
                               xaxis_title="Idade", yaxis_title="Frequência")
        st.plotly_chart(fig_age, use_container_width=True)

    # --- Médias das notas ---
    st.subheader("Médias por Área de Conhecimento")
    medias = df[QUANTITATIVE_VARS].mean().rename(SCORE_LABELS)
    fig_medias = px.bar(
        x=medias.index,
        y=medias.values,
        labels={"x": "Área", "y": "Média"},
        color=medias.values,
        color_continuous_scale="Viridis",
        text=[f"{v:.1f}" for v in medias.values],
    )
    fig_medias.update_traces(textposition="outside")
    fig_medias.update_layout(height=400, showlegend=False,
                              coloraxis_showscale=False)
    st.plotly_chart(fig_medias, use_container_width=True)

    # --- Prévia dos dados ---
    with st.expander("📋 Prévia dos dados (primeiras 200 linhas)"):
        st.dataframe(df.head(200), use_container_width=True)


# ===========================================================================
# PÁGINA 2 – VARIÁVEIS QUALITATIVAS
# ===========================================================================

elif page == "📊 Variáveis Qualitativas":
    st.title("📊 Variáveis Qualitativas")
    st.markdown(
        "Tabelas de distribuição de frequência e gráficos para as "
        "principais variáveis categóricas do ENEM 2024."
    )
    st.markdown("---")

    for var_info in QUALITATIVE_VARS:
        col_raw   = var_info["col"]
        col_label = var_info["label_col"]
        title     = var_info["title"]

        # Coluna de exibição: usa o label quando disponível
        display_col = col_label if col_label in df.columns else col_raw

        if display_col not in df.columns and col_raw not in df.columns:
            continue

        st.subheader(f"🔹 {title}")

        # Tabela de frequência
        freq_df = frequency_table(df, display_col)

        tab_table, tab_bar, tab_pie = st.tabs(["Tabela de Frequência", "Gráfico de Barras", "Gráfico de Pizza"])

        with tab_table:
            st.dataframe(freq_df, use_container_width=True, hide_index=True)

        with tab_bar:
            fig_bar = px.bar(
                freq_df,
                x="Categoria",
                y="Freq. Absoluta",
                text="Freq. Relativa (%)",
                color="Categoria",
                color_discrete_sequence=PALETTE,
                labels={"Freq. Absoluta": "Frequência Absoluta"},
                title=f"Distribuição – {title}",
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(showlegend=False, height=420,
                                  xaxis_title=title, yaxis_title="Frequência")
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab_pie:
            fig_pie = px.pie(
                freq_df,
                names="Categoria",
                values="Freq. Absoluta",
                color_discrete_sequence=PALETTE,
                title=f"Distribuição – {title}",
                hole=0.35,
            )
            fig_pie.update_traces(textinfo="percent+label")
            fig_pie.update_layout(height=420)
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")


# ===========================================================================
# PÁGINA 3 – VARIÁVEIS QUANTITATIVAS
# ===========================================================================

elif page == "📈 Variáveis Quantitativas":
    st.title("📈 Variáveis Quantitativas")
    st.markdown(
        "Histogramas, box plots e estatísticas descritivas das notas do ENEM 2024."
    )
    st.markdown("---")

    # --- Estatísticas descritivas ---
    st.subheader("Estatísticas Descritivas")
    stats_df = descriptive_stats(df)
    st.dataframe(stats_df, use_container_width=True)
    st.markdown("---")

    # --- Histogramas ---
    st.subheader("Histogramas")
    available_scores = [c for c in QUANTITATIVE_VARS if c in df.columns]
    score_options = {score_label(c): c for c in available_scores}

    sel_score_label = st.selectbox(
        "Selecione a área de conhecimento",
        list(score_options.keys()),
        key="hist_select",
    )
    sel_score_col = score_options[sel_score_label]

    col_hist, col_info = st.columns([3, 1])

    with col_hist:
        hist_data = df[sel_score_col].dropna()
        fig_hist = px.histogram(
            hist_data,
            nbins=50,
            color_discrete_sequence=["#636EFA"],
            labels={"value": "Nota", "count": "Frequência"},
            title=f"Histograma – {sel_score_label}",
        )
        # Linha de densidade (KDE)
        mean_val = hist_data.mean()
        fig_hist.add_vline(x=mean_val, line_dash="dash", line_color="red",
                           annotation_text=f"Média: {mean_val:.1f}",
                           annotation_position="top right")
        median_val = hist_data.median()
        fig_hist.add_vline(x=median_val, line_dash="dot", line_color="green",
                           annotation_text=f"Mediana: {median_val:.1f}",
                           annotation_position="top left")
        fig_hist.update_layout(height=420, xaxis_title="Nota", yaxis_title="Frequência")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_info:
        norm = normality_test(df, sel_score_col)
        st.markdown("**Teste de Normalidade**")
        st.markdown(f"- Teste: {norm['teste']}")
        st.markdown(f"- Estatística: {norm['estatística']:.4f}")
        st.markdown(f"- p-valor: {norm['p_valor']:.6f}")
        resultado = "✅ Normal" if norm["normal"] else "❌ Não Normal"
        st.markdown(f"- Resultado: {resultado}")

        st.markdown("**Percentis**")
        p_data = hist_data
        for pct in [10, 25, 50, 75, 90]:
            st.markdown(f"- P{pct}: {np.percentile(p_data, pct):.1f}")

    st.markdown("---")

    # --- Todos os histogramas lado a lado ---
    st.subheader("Todos os Histogramas")
    cols = st.columns(len(available_scores))
    for i, score_col in enumerate(available_scores):
        with cols[i]:
            data_col = df[score_col].dropna()
            fig_mini = px.histogram(
                data_col,
                nbins=40,
                title=score_label(score_col),
                color_discrete_sequence=[PALETTE[i % len(PALETTE)]],
                labels={"value": "Nota"},
            )
            fig_mini.update_layout(height=280, showlegend=False,
                                   xaxis_title="Nota", yaxis_title="Freq.",
                                   margin=dict(l=20, r=10, t=50, b=30))
            st.plotly_chart(fig_mini, use_container_width=True)

    st.markdown("---")

    # --- Box plots ---
    st.subheader("Box Plots das Notas")

    # Box plot geral (todas as notas)
    melt_df = df[available_scores].melt(var_name="Área", value_name="Nota").dropna()
    melt_df["Área"] = melt_df["Área"].map(SCORE_LABELS)

    fig_box = px.box(
        melt_df,
        x="Área",
        y="Nota",
        color="Área",
        color_discrete_sequence=PALETTE,
        points=False,
        title="Box Plot – Todas as Notas",
    )
    fig_box.update_layout(height=480, showlegend=False,
                           xaxis_title="Área", yaxis_title="Nota")
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # --- Box plot por grupo ---
    st.subheader("Box Plot por Grupo")
    group_options = {
        "Sexo": "SEXO_DESC",
        "Tipo de Escola": "ESCOLA_DESC",
        "Raça/Cor": "COR_RACA_DESC",
        "Situação de Conclusão": "ST_CONCLUSAO_DESC",
        "Treineiro": "TREINEIRO_DESC",
    }
    available_groups = {k: v for k, v in group_options.items() if v in df.columns}

    col_g1, col_g2 = st.columns(2)
    sel_group_label = col_g1.selectbox("Agrupar por", list(available_groups.keys()))
    sel_group_col   = available_groups[sel_group_label]
    sel_score_box   = col_g2.selectbox(
        "Nota", list(score_options.keys()), key="box_select"
    )
    sel_score_box_col = score_options[sel_score_box]

    fig_box_grp = px.box(
        df.dropna(subset=[sel_score_box_col, sel_group_col]),
        x=sel_group_col,
        y=sel_score_box_col,
        color=sel_group_col,
        color_discrete_sequence=PALETTE,
        points=False,
        title=f"{sel_score_box} por {sel_group_label}",
    )
    fig_box_grp.update_layout(height=480, showlegend=False,
                               xaxis_title=sel_group_label,
                               yaxis_title="Nota")
    st.plotly_chart(fig_box_grp, use_container_width=True)

    # Tabela de médias por grupo
    with st.expander(f"📋 Tabela de médias – {sel_score_box} por {sel_group_label}"):
        grp_table = mean_scores_by_group(df, sel_score_box_col, sel_group_col)
        st.dataframe(grp_table, use_container_width=True, hide_index=True)


# ===========================================================================
# PÁGINA 4 – ANÁLISE DE CORRELAÇÃO
# ===========================================================================

elif page == "🔗 Análise de Correlação":
    st.title("🔗 Análise de Correlação")
    st.markdown(
        "Heatmap e scatter matrix da correlação entre as notas do ENEM 2024."
    )
    st.markdown("---")

    available_scores = [c for c in QUANTITATIVE_VARS if c in df.columns]

    # Escolha do método
    method = st.selectbox(
        "Método de correlação",
        ["pearson", "spearman", "kendall"],
        format_func=lambda m: m.capitalize(),
    )

    corr_df = correlation_matrix(df, columns=available_scores, method=method)

    # --- Heatmap ---
    st.subheader("Matriz de Correlação (Heatmap)")

    corr_values = corr_df.values.tolist()
    labels = corr_df.columns.tolist()

    fig_heat = ff.create_annotated_heatmap(
        z=corr_values,
        x=labels,
        y=labels,
        colorscale="RdBu",
        reversescale=True,
        zmin=-1,
        zmax=1,
        annotation_text=[[f"{v:.3f}" for v in row] for row in corr_values],
        showscale=True,
    )
    fig_heat.update_layout(
        height=520,
        title=f"Correlação de {method.capitalize()} entre as notas",
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Tabela da matriz
    with st.expander("📋 Ver tabela da matriz de correlação"):
        st.dataframe(corr_df, use_container_width=True)

    st.markdown("---")

    # --- Scatter matrix ---
    st.subheader("Scatter Matrix (Pairplot)")
    st.markdown(
        "_Selecione as áreas que deseja incluir. "
        "O gráfico pode demorar alguns segundos para renderizar._"
    )

    score_options_multi = {score_label(c): c for c in available_scores}
    sel_pair_labels = st.multiselect(
        "Áreas de conhecimento",
        list(score_options_multi.keys()),
        default=list(score_options_multi.keys()),
    )

    if len(sel_pair_labels) >= 2:
        sel_pair_cols = [score_options_multi[lbl] for lbl in sel_pair_labels]
        color_by_options = {
            "Sexo": "SEXO_DESC",
            "Tipo de Escola": "ESCOLA_DESC",
            "Raça/Cor": "COR_RACA_DESC",
            "Nenhum": None,
        }
        available_color = {k: v for k, v in color_by_options.items()
                           if v is None or v in df.columns}
        color_choice = st.selectbox("Colorir por", list(available_color.keys()))
        color_col = available_color[color_choice]

        sample_df = df[sel_pair_cols + ([color_col] if color_col else [])].dropna().sample(
            min(3000, len(df)), random_state=42
        )
        renamed = {c: score_label(c) for c in sel_pair_cols}
        sample_df = sample_df.rename(columns=renamed)

        fig_scatter = px.scatter_matrix(
            sample_df,
            dimensions=list(renamed.values()),
            color=color_col if color_col else None,
            color_discrete_sequence=PALETTE,
            title="Scatter Matrix das Notas",
            opacity=0.4,
        )
        fig_scatter.update_traces(marker=dict(size=3))
        fig_scatter.update_layout(height=700)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Selecione pelo menos 2 áreas de conhecimento para exibir o gráfico.")

    st.markdown("---")

    # --- Correlação com variável de grupo ---
    st.subheader("Correlação Média por Grupo")
    group_options = {
        "Sexo": "SEXO_DESC",
        "Tipo de Escola": "ESCOLA_DESC",
        "Raça/Cor": "COR_RACA_DESC",
        "UF": "SG_UF_RESIDENCIA",
    }
    available_groups = {k: v for k, v in group_options.items() if v in df.columns}
    sel_grp = st.selectbox("Variável de agrupamento", list(available_groups.keys()),
                            key="corr_group")
    sel_grp_col = available_groups[sel_grp]

    score_label_sel = st.selectbox(
        "Nota para comparar por grupo",
        list(score_options_multi.keys()),
        key="corr_score",
    )
    score_col_sel = score_options_multi[score_label_sel]

    grp_means = mean_scores_by_group(df, score_col_sel, sel_grp_col)

    fig_grp = px.bar(
        grp_means,
        x="Categoria",
        y="Média",
        error_y=None,
        color="Média",
        color_continuous_scale="Tealrose",
        text="Média",
        title=f"Média de {score_label_sel} por {sel_grp}",
    )
    fig_grp.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_grp.update_layout(height=430, showlegend=False,
                           coloraxis_showscale=False,
                           xaxis_title=sel_grp, yaxis_title="Média")
    st.plotly_chart(fig_grp, use_container_width=True)
