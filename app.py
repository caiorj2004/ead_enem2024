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
import streamlit as st

from analysis import (
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
)
from database import load_data

warnings.filterwarnings("ignore", category=FutureWarning, module="plotly")


# ---------------------------------------------------------------------------
# Leitura das credenciais do banco de dados (st.secrets)
# ---------------------------------------------------------------------------

def _get_db_config() -> dict | None:
    """
    Lê as credenciais do banco PostgreSQL a partir de st.secrets["database"].
    Retorna None quando as credenciais não estiverem configuradas ou ainda
    contiverem os valores de placeholder (modo template).
    """
    try:
        section = st.secrets["database"]
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


_db_config        = _get_db_config()
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
all_ufs = sorted(df_full["uf"].dropna().unique().tolist())
sel_ufs = st.sidebar.multiselect("UF (Estado)", all_ufs, default=all_ufs)

# Filtro por tamanho do município (total_inscritos)
min_ins = int(df_full["total_inscritos"].min())
max_ins = int(df_full["total_inscritos"].max())
sel_min_ins = st.sidebar.slider(
    "Mínimo de inscritos por município",
    min_value=min_ins,
    max_value=max_ins,
    value=min_ins,
    step=max(1, (max_ins - min_ins) // 200),
)

# Aplica filtros
df = df_full[
    df_full["uf"].isin(sel_ufs)
    & (df_full["total_inscritos"] >= sel_min_ins)
].copy()

st.sidebar.markdown(f"**Municípios filtrados:** {len(df):,}")

if df.empty:
    st.warning(
        "⚠️ Nenhum município corresponde aos filtros selecionados. "
        "Selecione ao menos uma UF ou reduza o mínimo de inscritos."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Paleta de cores padrão
# ---------------------------------------------------------------------------

PALETTE = px.colors.qualitative.Plotly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_label(col: str) -> str:
    return SCORE_LABELS.get(col, PROPORTION_LABELS.get(col, col))


def card_metric(col, label, value, delta=None):
    with col:
        st.metric(label=label, value=value, delta=delta)


# ===========================================================================
# PÁGINA 1 – VISÃO GERAL
# ===========================================================================

if page == "🏠 Visão Geral":
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
    card_metric(c1, "Municípios analisados",        f"{n_municipios:,}")
    card_metric(c2, "Total de inscritos",            f"{total_inscritos:,}")
    card_metric(c3, "Média de idade (ponderada)",    f"{media_idade:.1f} anos")
    card_metric(c4, "Média geral nacional (ponderada)", f"{media_geral_br:.1f}")

    st.markdown("---")

    # --- Inscritos por UF ---
    col_uf, col_scores = st.columns(2)

    with col_uf:
        st.subheader("Total de Inscritos por UF")
        uf_agg = inscribed_by_uf(df).head(27)
        fig_uf = px.bar(
            uf_agg,
            x="uf",
            y="total_inscritos",
            color="total_inscritos",
            color_continuous_scale="Blues",
            text_auto=True,
            labels={"uf": "UF", "total_inscritos": "Inscritos"},
        )
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
                text=[f"{v:.1f}" for v in medias.values],
                labels={"x": "Área", "y": "Média"},
            )
            fig_medias.update_traces(textposition="outside")
            fig_medias.update_layout(height=400, showlegend=False,
                                     coloraxis_showscale=False,
                                     xaxis_title="Área", yaxis_title="Média")
            st.plotly_chart(fig_medias, use_container_width=True)

    # --- Distribuição do tamanho dos municípios ---
    st.subheader("Distribuição do Total de Inscritos por Município (escala log)")
    fig_hist_ins = px.histogram(
        df,
        x="total_inscritos",
        nbins=60,
        log_x=True,
        color_discrete_sequence=["#636EFA"],
        labels={"total_inscritos": "Total de Inscritos"},
    )
    fig_hist_ins.update_layout(height=360, xaxis_title="Total de Inscritos (log)",
                                yaxis_title="Número de Municípios")
    st.plotly_chart(fig_hist_ins, use_container_width=True)

    # --- Prévia dos dados ---
    with st.expander("📋 Prévia dos dados municipais (primeiras 200 linhas)"):
        st.dataframe(df.head(200), use_container_width=True)


# ===========================================================================
# PÁGINA 2 – VARIÁVEIS QUALITATIVAS
# ===========================================================================

elif page == "📊 Variáveis Qualitativas":
    st.title("📊 Variáveis Qualitativas")
    st.markdown(
        "Tabelas de distribuição de frequência e gráficos para a variável "
        "categórica principal (UF) e a composição demográfica dos municípios."
    )
    st.markdown("---")

    # ---- Seção 1: Distribuição por UF ----
    st.subheader("🔹 Distribuição de Municípios por UF")

    freq_df = frequency_table(df, "uf")
    tab_table, tab_bar = st.tabs(
        ["Tabela de Frequência", "Gráfico de Barras"]
    )

    with tab_table:
        st.markdown("Cada linha representa quantos municípios estão em cada estado.")
        st.dataframe(freq_df, use_container_width=True, hide_index=True)

    with tab_bar:
        fig_bar = px.bar(
            freq_df,
            x="Categoria",
            y="Freq. Absoluta",
            text="Freq. Relativa (%)",
            color="Categoria",
            color_discrete_sequence=PALETTE,
            labels={"Freq. Absoluta": "Nº de Municípios"},
            title="Número de Municípios por UF",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(showlegend=False, height=440,
                               xaxis_title="UF", yaxis_title="Municípios")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ---- Seção 2: Total de Inscritos por UF ----
    st.subheader("🔹 Total de Inscritos por UF")
    uf_ins = inscribed_by_uf(df)

    col_ins_bar, col_ins_tab = st.columns([2, 1])
    with col_ins_bar:
        fig_ins = px.bar(
            uf_ins,
            x="uf",
            y="total_inscritos",
            color="total_inscritos",
            color_continuous_scale="Teal",
            text_auto=True,
            title="Total de Inscritos por Estado",
            labels={"uf": "UF", "total_inscritos": "Inscritos"},
        )
        fig_ins.update_layout(height=420, showlegend=False,
                               coloraxis_showscale=False,
                               xaxis_title="UF", yaxis_title="Inscritos")
        st.plotly_chart(fig_ins, use_container_width=True)

    with col_ins_tab:
        st.markdown("**Tabela resumo**")
        st.dataframe(
            uf_ins.rename(columns={"uf": "UF", "municipios": "Municípios",
                                    "total_inscritos": "Inscritos"}),
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
    chosen_cols = [c for c in demo_cols[demo_choice] if c in df.columns]

    if chosen_cols:
        uf_demo = (
            df.groupby("uf")[chosen_cols].mean().round(2).reset_index()
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


# ===========================================================================
# PÁGINA 3 – VARIÁVEIS QUANTITATIVAS
# ===========================================================================

elif page == "📈 Variáveis Quantitativas":
    st.title("📈 Variáveis Quantitativas")
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
    st.dataframe(stats_df, use_container_width=True)
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
                           annotation_text=f"Média: {mean_val:.1f}",
                           annotation_position="top right")
        fig_hist.add_vline(x=median_val, line_dash="dot", line_color="green",
                           annotation_text=f"Mediana: {median_val:.1f}",
                           annotation_position="top left")
        fig_hist.update_layout(height=420, xaxis_title=sel_score_label,
                                yaxis_title="Municípios")
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
        for pct in [10, 25, 50, 75, 90]:
            st.markdown(f"- P{pct}: {np.percentile(hist_data, pct):.1f}")

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
        df.dropna(subset=[sel_score_box_col, "uf"]),
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
        grp_table = mean_scores_by_group(df, sel_score_box_col, "uf")
        st.dataframe(grp_table, use_container_width=True, hide_index=True)


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

    col_left, col_right = st.columns([3, 1])
    with col_right:
        include_props = st.checkbox("Incluir proporções demográficas", value=True)

    corr_cols = available_scores + (available_props if include_props else [])
    corr_df   = correlation_matrix(df, columns=corr_cols, method=method)

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
        annotation_text=[[f"{v:.2f}" for v in row] for row in corr_values],
        showscale=True,
    )
    fig_heat.update_layout(
        height=max(500, len(labels) * 45),
        title=f"Correlação de {method.capitalize()} entre notas e indicadores municipais",
        xaxis=dict(side="bottom"),
    )
    with col_left:
        st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("📋 Ver tabela da matriz de correlação"):
        st.dataframe(corr_df, use_container_width=True)

    st.markdown("---")

    # ---- Scatter matrix de notas ----
    st.subheader("Scatter Matrix das Notas")
    st.markdown(
        "_Distribuição par a par das médias municipais de notas. "
        "Colorido por UF. Usa amostra de até 2000 municípios para performance._"
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
            st.dataframe(pair_corr, use_container_width=True)
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

        scatter_df = df[[sel_x_col, sel_y_col, "uf", "municipio", "total_inscritos"]].dropna()
        fig_sc = px.scatter(
            scatter_df,
            x=sel_x_col,
            y=sel_y_col,
            color="uf",
            size="total_inscritos",
            size_max=20,
            hover_name="municipio",
            hover_data={"uf": True, "total_inscritos": True},
            color_discrete_sequence=PALETTE,
            opacity=0.6,
            title=f"{sel_y_lbl} × {sel_x_lbl} (por município)",
            labels={sel_x_col: sel_x_lbl, sel_y_col: sel_y_lbl},
        )
        # Add OLS trend line without requiring statsmodels
        x_vals  = scatter_df[sel_x_col].values
        y_vals  = scatter_df[sel_y_col].values
        poly_coef = np.polyfit(x_vals, y_vals, 1)
        x_range   = np.linspace(x_vals.min(), x_vals.max(), 200)
        fig_sc.add_scatter(
            x=x_range, y=np.polyval(poly_coef, x_range),
            mode="lines", line=dict(color="black", width=2, dash="dash"),
            name="Tendência (OLS)", showlegend=False,
        )
        fig_sc.update_layout(height=520)
        st.plotly_chart(fig_sc, use_container_width=True)

        # Correlação pontual
        corr_val = scatter_df[[sel_x_col, sel_y_col]].corr(method=method).iloc[0, 1]
        st.markdown(
            f"**Correlação de {method.capitalize()} entre "
            f"_{sel_x_lbl}_ e _{sel_y_lbl}_: `{corr_val:.4f}`**"
        )
    else:
        st.info("Colunas de proporção não disponíveis. Verifique se os dados foram carregados corretamente.")

    st.markdown("---")

    # ---- Médias ponderadas por UF ----
    st.subheader("Médias Ponderadas por UF")
    score_opt_grp = {score_label(c): c for c in available_scores}
    sel_grp_score = st.selectbox("Nota", list(score_opt_grp.keys()), key="corr_uf_score")
    sel_grp_col   = score_opt_grp[sel_grp_score]

    grp_means = mean_scores_by_group(df, sel_grp_col, "uf")

    fig_grp = px.bar(
        grp_means,
        x="Categoria",
        y="Média Ponderada",
        color="Média Ponderada",
        color_continuous_scale="Tealrose",
        text="Média Ponderada",
        title=f"Média Ponderada de {sel_grp_score} por UF",
    )
    fig_grp.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig_grp.update_layout(height=450, showlegend=False,
                           coloraxis_showscale=False,
                           xaxis_title="UF", yaxis_title="Nota Média Ponderada")
    st.plotly_chart(fig_grp, use_container_width=True)
