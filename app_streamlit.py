"""Interface Streamlit do projeto Steam Reviews.

Estrutura:

* **Sidebar** — filtros por gênero/modo, faixa de preço e período de
  lançamento; toggle para habilitar a seção do modelo.
* **Introdução** — contexto curto do projeto.
* **Visão geral** — KPIs do recorte atual.
* **Análise por dimensão** — gráficos univariados (taxa de recomendação
  por gênero/modo, preço, período + top tags).
* **Padrões escondidos** — top combinações de 2 características e
  visualização "Venn-like" entre flags.
* **Modelo baseline** (opcional) — métricas, top features e formulário
  de predição interativa.

Rodar com::

    streamlit run app_streamlit.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.eda import (
    COL_TARGET,
    plot_recommendation_rate_by_genre,
    plot_recommendation_rate_by_price_range,
    plot_recommendation_rate_by_release_period,
    plot_top_genres_by_review_count,
)
from src.modeling import (
    build_train_test_split,
    evaluate_model,
    top_positive_features,
    train_baseline_model,
)
from src.patterns import (
    DEFAULT_MIN_GAMES,
    plot_flag_pair_overlap,
    plot_top_combinations,
    top_combinations,
)
from src.preprocessing import COL_APP_ID, TAG_FLAGS


ROOT = Path(__file__).parent
PROCESSED_PATH = ROOT / "data" / "processed" / "steam_reviews_processed.csv"

# Ordens estáveis para os multiselects (mesma convenção usada em eda.py).
_PRICE_BAND_ORDER = ["Grátis", "Barato", "Médio", "Caro"]
_RELEASE_BAND_ORDER = [
    "Antes de 2010",
    "2010-2015",
    "2016-2020",
    "Depois de 2020",
]


# --- Caching ------------------------------------------------------------
@st.cache_data(show_spinner="Carregando dataset processado...")
def load_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_PATH)


@st.cache_resource(show_spinner="Treinando baseline (executado 1x por sessão)...")
def train_cached_baseline():
    """Treina o pipeline baseline na base COMPLETA (não respeita filtros).
    ``@st.cache_resource`` mantém o modelo em memória entre re-renders."""
    df = load_data()
    X_train, X_test, y_train, y_test = build_train_test_split(df)
    model = train_baseline_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    return model, metrics, list(X_train.columns)


# --- Entry point --------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Steam Reviews — O que faz um jogo ser bem avaliado?",
        layout="wide",
    )

    if not PROCESSED_PATH.exists():
        st.error(
            f"Arquivo não encontrado: `{PROCESSED_PATH}`.\n\n"
            "Rode o pipeline de pré-processamento antes:\n\n"
            "```\npython main_preprocessing.py\n```"
        )
        st.stop()

    df = load_data()
    filtered = render_sidebar(df)

    render_intro()
    st.divider()
    render_overview(filtered)
    st.divider()
    render_eda_section(filtered)
    st.divider()
    render_patterns_section(filtered)

    if st.session_state.get("show_model"):
        st.divider()
        render_model_section()


# --- Sidebar / filtros --------------------------------------------------
def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("Filtros")

    flag_cols = [c for c in TAG_FLAGS if c in df.columns]
    selected_flags = st.sidebar.multiselect(
        "Gêneros / modos de jogo",
        options=flag_cols,
        format_func=_flag_label,
        help=(
            "Mostra reviews de jogos com pelo menos UMA das características "
            "selecionadas (união). Deixar vazio = sem filtro."
        ),
    )

    price_options = _ordered_options(df, "price_band", _PRICE_BAND_ORDER)
    selected_prices = st.sidebar.multiselect(
        "Faixa de preço",
        options=price_options,
        default=price_options,
    )

    year_options = _ordered_options(df, "release_year_band", _RELEASE_BAND_ORDER)
    selected_years = st.sidebar.multiselect(
        "Período de lançamento",
        options=year_options,
        default=year_options,
    )

    # Aplica filtros encadeados; multiselect vazio = filtro não aplicado.
    mask = pd.Series(True, index=df.index)
    if selected_flags:
        mask &= (df[selected_flags] == 1).any(axis=1)
    if selected_prices and "price_band" in df.columns:
        mask &= df["price_band"].isin(selected_prices)
    if selected_years and "release_year_band" in df.columns:
        mask &= df["release_year_band"].isin(selected_years)
    filtered = df[mask].reset_index(drop=True)

    st.sidebar.divider()
    st.sidebar.markdown(f"**Reviews após filtro:** {len(filtered):,}")
    if 0 < len(filtered) < 100:
        st.sidebar.warning(
            "Filtros muito restritivos — gráficos podem ficar ruidosos."
        )

    st.sidebar.divider()
    st.sidebar.checkbox(
        "Mostrar seção do modelo",
        key="show_model",
        value=False,
        help="Habilita a seção do baseline ML (treina sob demanda na 1ª vez).",
    )

    return filtered


# --- Seções da página ---------------------------------------------------
def render_intro() -> None:
    st.title("O que faz um jogo ser bem avaliado na Steam?")
    st.markdown(
        """
        Este app explora a relação entre **metadados de jogos** (gênero,
        tags, preço, ano de lançamento, popularidade) e a **taxa de
        recomendação** das reviews na Steam.

        Use os **filtros na barra lateral** para focar em um subconjunto
        e observe como cada dimensão influencia a probabilidade de o
        jogo ser recomendado. A seção **Padrões escondidos** ranqueia
        combinações de duas características com a maior taxa de
        recomendação — equivalente a buscar padrões do tipo
        "cerveja/fralda" no catálogo da Steam.
        """
    )


def render_overview(df: pd.DataFrame) -> None:
    st.header("Visão geral")

    if df.empty:
        st.warning("Nenhum registro corresponde aos filtros atuais.")
        return

    n_reviews = len(df)
    n_games = df[COL_APP_ID].nunique() if COL_APP_ID in df.columns else None
    rec_rate = df[COL_TARGET].mean()

    top_flag_label = "—"
    flag_cols = [c for c in TAG_FLAGS if c in df.columns]
    if flag_cols:
        counts = {f: int((df[f] == 1).sum()) for f in flag_cols}
        if any(counts.values()):
            top_flag_label = _flag_label(max(counts, key=counts.get))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", f"{n_reviews:,}")
    c2.metric("Jogos únicos", f"{n_games:,}" if n_games is not None else "—")
    c3.metric("Taxa de recomendação", f"{rec_rate:.1%}")
    c4.metric("Gênero mais frequente", top_flag_label)


def render_eda_section(df: pd.DataFrame) -> None:
    st.header("Análise por dimensão")

    if len(df) < 50:
        st.warning(
            "Poucos registros após filtro — gráficos podem ficar instáveis."
        )
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Por gênero / modo", "Por preço", "Por período", "Top tags"]
    )
    with tab1:
        _safe_plot(plot_recommendation_rate_by_genre, df)
    with tab2:
        _safe_plot(plot_recommendation_rate_by_price_range, df)
    with tab3:
        _safe_plot(plot_recommendation_rate_by_release_period, df)
    with tab4:
        top_n = st.slider("Top N tags", 5, 30, 15, key="top_tags_n")
        _safe_plot(plot_top_genres_by_review_count, df, top_n=top_n)


def render_patterns_section(df: pd.DataFrame) -> None:
    st.header("Padrões escondidos")
    st.caption(
        "Combinações de 2 características com maior taxa de recomendação. "
        "O filtro de jogos mínimos evita rankings dominados por amostras pequenas."
    )

    if df.empty:
        return

    c1, c2 = st.columns(2)
    top_n = int(
        c1.number_input("Top N", min_value=5, max_value=30, value=10, step=1)
    )
    min_games = int(
        c2.number_input(
            "Mín. de jogos por combinação",
            min_value=1,
            max_value=500,
            value=DEFAULT_MIN_GAMES,
            step=10,
        )
    )

    top = top_combinations(df, top_n=top_n, min_games=min_games)
    if top.empty:
        st.warning(
            f"Nenhuma combinação com pelo menos {min_games} jogos únicos. "
            "Reduza o threshold ou afrouxe os filtros laterais."
        )
    else:
        baseline = df[COL_TARGET].mean()
        st.subheader("Top combinações")
        st.dataframe(
            top.style.format(
                {
                    "rate": "{:.1%}",
                    "lift": "{:.2f}x",
                    "n_reviews": "{:,}",
                    "n_games": "{:,}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        _safe_plot(plot_top_combinations, top, baseline=baseline)

    st.subheader("Sobreposição entre duas features (Venn-like)")
    flag_cols = [c for c in TAG_FLAGS if c in df.columns]
    if len(flag_cols) < 2:
        st.info("Necessárias ao menos 2 flags has_* para esta visualização.")
        return

    c1, c2, c3 = st.columns([2, 2, 1])
    flag_a = c1.selectbox(
        "Feature A", flag_cols, index=0, format_func=_flag_label
    )
    flag_b_options = [f for f in flag_cols if f != flag_a]
    flag_b = c2.selectbox(
        "Feature B", flag_b_options, index=0, format_func=_flag_label
    )
    show_neither = c3.checkbox("Incluir 'nem A nem B'", value=False)
    _safe_plot(
        plot_flag_pair_overlap,
        df,
        flag_a,
        flag_b,
        show_neither=show_neither,
    )


def render_model_section() -> None:
    st.header("Modelo baseline (LogisticRegression)")
    st.caption(
        "Treinado no dataset **completo** — não respeita os filtros laterais. "
        "Modelo linear simples e interpretável; não captura interações tipo "
        "'cerveja/fralda'. Para padrões combinados, ver seção anterior."
    )

    model, metrics, feature_names = train_cached_baseline()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("Precision", f"{metrics['precision']:.1%}")
    c3.metric("Recall", f"{metrics['recall']:.1%}")
    c4.metric("F1", f"{metrics['f1']:.1%}")

    st.subheader("Matriz de confusão")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=["real=0 (não rec.)", "real=1 (rec.)"],
        columns=["pred=0", "pred=1"],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Top 10 features associadas a MAIOR probabilidade de recomendação")
    top = top_positive_features(model, feature_names, top_n=10)
    st.dataframe(
        top.style.format({"coef": "{:+.3f}"}),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Predição interativa")
    st.caption(
        "Configure as características hipotéticas de um jogo e veja a "
        "probabilidade estimada pelo baseline."
    )
    _render_prediction_form(model, feature_names)


# --- Helpers ------------------------------------------------------------
def _render_prediction_form(model, feature_names: list[str]) -> None:
    df = load_data()  # cacheada

    with st.form("predict_form"):
        inputs: dict[str, float | int] = {}
        cols = st.columns(3)
        for i, feat in enumerate(feature_names):
            col = cols[i % 3]
            if feat in TAG_FLAGS:
                inputs[feat] = int(
                    col.checkbox(
                        _flag_label(feat),
                        value=False,
                        key=f"pred_{feat}",
                    )
                )
                continue

            series = df[feat].dropna()
            if series.empty:
                inputs[feat] = 0.0
                continue

            vmin, vmax = float(series.min()), float(series.max())
            vmed = float(series.median())
            if pd.api.types.is_integer_dtype(series):
                inputs[feat] = col.number_input(
                    feat,
                    min_value=int(vmin),
                    max_value=int(vmax),
                    value=int(vmed),
                    step=1,
                    key=f"pred_{feat}",
                )
            else:
                inputs[feat] = col.number_input(
                    feat,
                    min_value=vmin,
                    max_value=vmax,
                    value=vmed,
                    key=f"pred_{feat}",
                )

        submitted = st.form_submit_button("Calcular probabilidade")

    if not submitted:
        return

    X_one = pd.DataFrame([inputs], columns=feature_names)
    proba = model.predict_proba(X_one)[0]
    # Identifica explicitamente o índice da classe positiva.
    classes = list(model.classes_)
    idx_pos = classes.index(1) if 1 in classes else -1
    prob_rec = float(proba[idx_pos])
    st.success(f"Probabilidade de ser **recomendado**: **{prob_rec:.1%}**")


def _safe_plot(plot_fn, *args, **kwargs) -> None:
    """Renderiza a figura no Streamlit e libera memória após o display."""
    try:
        fig = plot_fn(*args, **kwargs)
    except ValueError as exc:
        st.info(str(exc))
        return
    st.pyplot(fig)
    plt.close(fig)


def _flag_label(col_name: str) -> str:
    return col_name.replace("has_", "").capitalize()


def _ordered_options(
    df: pd.DataFrame, col: str, order: list[str]
) -> list[str]:
    """Retorna os valores de ``col`` presentes em ``df`` mantendo a ordem
    canônica fornecida."""
    if col not in df.columns:
        return []
    present = set(df[col].dropna().unique())
    return [v for v in order if v in present]


if __name__ == "__main__":
    main()
