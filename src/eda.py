"""Funções de EDA para a base processada de reviews da Steam.

Cada função ``plot_*`` recebe um ``DataFrame`` já processado (saída do
``preprocessing``) e devolve uma ``matplotlib.figure.Figure``. O caller
decide se salva em disco (``main_eda``) ou se renderiza inline (Streamlit).
Isso mantém o módulo agnóstico ao destino e fácil de testar.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from src.preprocessing import COL_GENRES, COL_TAGS, TAG_FLAGS, _parse_tags


# Nome da coluna alvo no CSV processado (binária, 0/1).
COL_TARGET = "recommended"

# Ordem categórica das faixas. np.select gera strings sem ordering implícito,
# então definimos a sequência aqui para o eixo X ficar consistente.
PRICE_BAND_ORDER = ["Grátis", "Barato", "Médio", "Caro"]
RELEASE_BAND_ORDER = [
    "Antes de 2010",
    "2010-2015",
    "2016-2020",
    "Depois de 2020",
]

_BAR_COLOR = "#1f77b4"


# --- Plots públicos -----------------------------------------------------
def plot_recommendation_rate_by_genre(df: pd.DataFrame) -> Figure:
    """Taxa de recomendação para cada flag ``has_*`` (presença da tag/gênero).

    A taxa é calculada apenas entre os jogos onde a flag está ativa, e uma
    linha tracejada marca a média geral para referência visual.
    """
    flags = [c for c in TAG_FLAGS if c in df.columns]
    if not flags:
        raise ValueError("Nenhuma coluna has_* encontrada no DataFrame.")

    rows = []
    for flag in flags:
        mask = df[flag] == 1
        if mask.any():
            rows.append((flag, df.loc[mask, COL_TARGET].mean(), int(mask.sum())))
    summary = (
        pd.DataFrame(rows, columns=["genre", "rate", "n"])
        .sort_values("rate", ascending=False)
    )

    baseline = df[COL_TARGET].mean()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(summary["genre"], summary["rate"], color=_BAR_COLOR)
    ax.axhline(
        baseline,
        linestyle="--",
        color="gray",
        linewidth=1,
        label=f"Média geral ({baseline:.1%})",
    )
    _annotate_bars_v(ax, bars, summary["rate"], fmt="{:.1%}")
    _setup_axes(
        ax,
        title="Taxa de recomendação por gênero / modo de jogo",
        xlabel="Característica",
        ylabel="Taxa de recomendação",
    )
    ax.set_ylim(0, max(summary["rate"].max(), baseline) * 1.15)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_recommendation_rate_by_price_range(df: pd.DataFrame) -> Figure:
    """Taxa de recomendação por faixa de preço (Grátis / Barato / Médio / Caro)."""
    if "price_band" not in df.columns:
        raise ValueError("Coluna 'price_band' não encontrada.")
    return _rate_by_category(
        df,
        category_col="price_band",
        order=PRICE_BAND_ORDER,
        title="Taxa de recomendação por faixa de preço",
        xlabel="Faixa de preço",
    )


def plot_recommendation_rate_by_release_period(df: pd.DataFrame) -> Figure:
    """Taxa de recomendação por período de lançamento."""
    if "release_year_band" not in df.columns:
        raise ValueError("Coluna 'release_year_band' não encontrada.")
    return _rate_by_category(
        df,
        category_col="release_year_band",
        order=RELEASE_BAND_ORDER,
        title="Taxa de recomendação por período de lançamento",
        xlabel="Período de lançamento",
    )


def plot_top_genres_by_review_count(df: pd.DataFrame, top_n: int = 15) -> Figure:
    """Top ``top_n`` tags/gêneros pelo número total de reviews no dataset.

    Usa a coluna de tags se disponível (mais granular); cai para a de gêneros.
    O parsing reaproveita ``_parse_tags`` do preprocessing para garantir
    consistência com as flags ``has_*``.
    """
    tag_col = COL_TAGS if COL_TAGS in df.columns else COL_GENRES
    if tag_col not in df.columns:
        raise ValueError("Nenhuma coluna de tags/gêneros encontrada.")

    counter: Counter[str] = Counter()
    for raw in df[tag_col]:
        counter.update(_parse_tags(raw))
    if not counter:
        raise ValueError("Nenhuma tag pôde ser extraída do campo.")

    top = pd.Series(counter).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(top))))
    # Inverte para que a tag com mais reviews fique no topo do eixo Y.
    bars = ax.barh(top.index[::-1], top.values[::-1], color=_BAR_COLOR)
    _annotate_bars_h(ax, bars, top.values[::-1], fmt="{:,.0f}")
    _setup_axes(
        ax,
        title=f"Top {top_n} tags/gêneros por nº de reviews",
        xlabel="Nº de reviews",
        ylabel="Tag / Gênero",
        grid_axis="x",
    )
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    fig.tight_layout()
    return fig


# --- Helpers internos ---------------------------------------------------
def _rate_by_category(
    df: pd.DataFrame,
    *,
    category_col: str,
    order: Iterable[str],
    title: str,
    xlabel: str,
) -> Figure:
    """Compartilha o esqueleto do gráfico de barras categóricas para taxa
    de recomendação. ``order`` define a ordem do eixo X (categorias
    ausentes são silenciosamente puladas)."""
    grouped = df.groupby(category_col)[COL_TARGET].agg(["mean", "size"])
    order = [c for c in order if c in grouped.index]
    if not order:
        raise ValueError(f"Nenhuma categoria de '{category_col}' encontrada.")
    grouped = grouped.loc[order]

    baseline = df[COL_TARGET].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(grouped.index, grouped["mean"], color=_BAR_COLOR)
    ax.axhline(
        baseline,
        linestyle="--",
        color="gray",
        linewidth=1,
        label=f"Média geral ({baseline:.1%})",
    )
    _annotate_bars_v(ax, bars, grouped["mean"], fmt="{:.1%}")
    _setup_axes(ax, title=title, xlabel=xlabel, ylabel="Taxa de recomendação")
    ax.set_ylim(0, max(grouped["mean"].max(), baseline) * 1.15)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def _setup_axes(
    ax,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    grid_axis: str = "y",
) -> None:
    """Aplica estilo consistente (título em negrito, sem spines top/right,
    grade leve) para boa leitura também quando renderizado no Streamlit."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis=grid_axis, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _annotate_bars_v(ax, bars, values, fmt: str = "{}") -> None:
    """Escreve o valor de cada barra acima dela (gráfico vertical)."""
    for bar, val in zip(bars, values):
        if pd.isna(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _annotate_bars_h(ax, bars, values, fmt: str = "{}") -> None:
    """Escreve o valor de cada barra à direita dela (gráfico horizontal)."""
    for bar, val in zip(bars, values):
        if pd.isna(val):
            continue
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {fmt.format(val)}",
            ha="left",
            va="center",
            fontsize=9,
        )
