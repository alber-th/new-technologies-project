"""Descoberta de padrões "cerveja/fralda" no dataset processado da Steam.

Foca em combinações de **duas** características (flag×faixa categórica e
flag×flag) e ranqueia as que têm taxa de recomendação mais alta, filtrando
combinações com poucos jogos únicos para evitar conclusões ruidosas.

Funções devolvem ``DataFrame`` ou ``Figure``, ambos pensados para serem
reaproveitados na interface Streamlit sem alteração.
"""
from __future__ import annotations

import itertools
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from src.eda import COL_TARGET, _setup_axes
from src.preprocessing import COL_APP_ID, TAG_FLAGS


DEFAULT_MIN_GAMES = 50

_BAR_COLOR = "#1f77b4"
_HIGHLIGHT_COLOR = "#ff7f0e"
_BASELINE_COLOR = "gray"

# Colunas finais expostas pelos rankings de combos. Manter estável facilita
# o consumo no Streamlit e no relatório.
_COMBO_OUTPUT_COLS = ["combo", "rate", "n_reviews", "n_games", "lift"]


# --- API pública --------------------------------------------------------
def available_genre_flags(df: pd.DataFrame) -> list[str]:
    """Subconjunto de ``TAG_FLAGS`` realmente presente no DataFrame."""
    return [c for c in TAG_FLAGS if c in df.columns]


def combo_rate_flag_x_category(
    df: pd.DataFrame,
    flag_cols: Sequence[str],
    category_col: str,
    *,
    min_games: int = DEFAULT_MIN_GAMES,
    target_col: str = COL_TARGET,
) -> pd.DataFrame:
    """Cruza cada flag binária com uma coluna categórica.

    Para cada flag em ``flag_cols``, restringe ao subset onde a flag está
    ativa e agrupa por ``category_col``, calculando taxa de recomendação,
    nº de reviews, nº de jogos únicos e lift (taxa / baseline).

    Combinações com ``n_games`` abaixo de ``min_games`` são descartadas
    para não enviesar o ranking com amostras pequenas.
    """
    if category_col not in df.columns:
        raise ValueError(f"Coluna '{category_col}' não encontrada.")
    if COL_APP_ID not in df.columns:
        raise ValueError(f"Coluna '{COL_APP_ID}' (id do jogo) é obrigatória.")

    baseline = df[target_col].mean()
    rows: list[dict] = []
    for flag in flag_cols:
        if flag not in df.columns:
            continue
        subset = df[df[flag] == 1]
        if subset.empty:
            continue
        grouped = (
            subset.groupby(category_col, dropna=True)
            .agg(
                rate=(target_col, "mean"),
                n_reviews=(target_col, "size"),
                n_games=(COL_APP_ID, "nunique"),
            )
            .reset_index()
        )
        for _, r in grouped.iterrows():
            rows.append(
                {
                    "feature_a": flag,
                    "feature_b": r[category_col],
                    "rate": r["rate"],
                    "n_reviews": int(r["n_reviews"]),
                    "n_games": int(r["n_games"]),
                    "lift": _safe_lift(r["rate"], baseline),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out[out["n_games"] >= min_games].copy()
    out["combo"] = (
        out["feature_a"].str.replace("has_", "", regex=False)
        + " + "
        + out["feature_b"].astype(str)
    )
    return out.sort_values("rate", ascending=False).reset_index(drop=True)


def combo_rate_flag_pair(
    df: pd.DataFrame,
    flag_a: str,
    flag_b: str,
    *,
    min_games: int = 0,
    target_col: str = COL_TARGET,
) -> pd.DataFrame:
    """Cross de duas flags binárias.

    Devolve até 4 linhas (uma por quadrante: A∧B, A∧¬B, ¬A∧B, ¬A∧¬B) com
    taxa de recomendação, contagens, lift e um rótulo amigável da zona
    (``zone``). Útil tanto para alimentar o ranking quanto a visualização
    "Venn-like".
    """
    for col in (flag_a, flag_b):
        if col not in df.columns:
            raise ValueError(f"Flag '{col}' não encontrada.")
    if COL_APP_ID not in df.columns:
        raise ValueError(f"Coluna '{COL_APP_ID}' (id do jogo) é obrigatória.")

    baseline = df[target_col].mean()
    grouped = (
        df.groupby([flag_a, flag_b])
        .agg(
            rate=(target_col, "mean"),
            n_reviews=(target_col, "size"),
            n_games=(COL_APP_ID, "nunique"),
        )
        .reset_index()
    )
    grouped["lift"] = grouped["rate"].map(lambda v: _safe_lift(v, baseline))
    grouped["zone"] = grouped.apply(
        lambda r: _zone_label(flag_a, flag_b, int(r[flag_a]), int(r[flag_b])),
        axis=1,
    )
    grouped = grouped[grouped["n_games"] >= min_games].copy()
    return grouped.sort_values("rate", ascending=False).reset_index(drop=True)


def top_combinations(
    df: pd.DataFrame,
    *,
    top_n: int = 10,
    min_games: int = DEFAULT_MIN_GAMES,
    target_col: str = COL_TARGET,
) -> pd.DataFrame:
    """Ranking unificado das melhores combinações de 2 características.

    Junta três fontes:

    1. cada flag ``has_*`` × ``price_band``;
    2. cada flag ``has_*`` × ``release_year_band``;
    3. cada par distinto de flags ``has_*`` × ``has_*`` (apenas o
       quadrante "ambos ativos", que é o que mapeia um padrão real).

    Filtra por ``n_games >= min_games`` e devolve as top ``top_n`` por
    taxa de recomendação.
    """
    flags = available_genre_flags(df)
    parts: list[pd.DataFrame] = []

    for category_col in ("price_band", "release_year_band"):
        if category_col not in df.columns:
            continue
        part = combo_rate_flag_x_category(
            df,
            flags,
            category_col,
            min_games=min_games,
            target_col=target_col,
        )
        if not part.empty:
            parts.append(part[_COMBO_OUTPUT_COLS])

    for fa, fb in itertools.combinations(flags, 2):
        try:
            pair = combo_rate_flag_pair(
                df, fa, fb, min_games=min_games, target_col=target_col
            )
        except ValueError:
            continue
        # Só interessa o quadrante onde ambas as features estão ativas.
        both = pair[(pair[fa] == 1) & (pair[fb] == 1)]
        if both.empty:
            continue
        both = both.assign(combo=both["zone"])
        parts.append(both[_COMBO_OUTPUT_COLS])

    if not parts:
        return pd.DataFrame(columns=_COMBO_OUTPUT_COLS)

    out = pd.concat(parts, ignore_index=True)
    return (
        out.sort_values("rate", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# --- Visualizações ------------------------------------------------------
def plot_top_combinations(
    combos: pd.DataFrame,
    *,
    baseline: float | None = None,
    title: str = "Top combinações por taxa de recomendação",
) -> Figure:
    """Bar chart horizontal das top combinações. Cada barra é anotada com
    a taxa e o nº de jogos únicos; se ``baseline`` for fornecido, traça
    uma linha tracejada para referência visual."""
    if combos.empty:
        raise ValueError("DataFrame de combos vazio — nada para plotar.")

    combos = combos.sort_values("rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(5, 0.45 * len(combos))))
    bars = ax.barh(combos["combo"], combos["rate"], color=_BAR_COLOR)

    if baseline is not None:
        ax.axvline(
            baseline,
            linestyle="--",
            color=_BASELINE_COLOR,
            linewidth=1,
            label=f"Média geral ({baseline:.1%})",
        )
        ax.legend(loc="lower right")

    for bar, rate, n_games in zip(bars, combos["rate"], combos["n_games"]):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  {rate:.1%}  (n_jogos={int(n_games):,})",
            ha="left",
            va="center",
            fontsize=9,
        )

    _setup_axes(
        ax,
        title=title,
        xlabel="Taxa de recomendação",
        ylabel="Combinação",
        grid_axis="x",
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    upper = max(combos["rate"].max(), baseline or 0) * 1.25
    ax.set_xlim(0, upper)
    fig.tight_layout()
    return fig


def plot_flag_pair_overlap(
    df: pd.DataFrame,
    flag_a: str,
    flag_b: str,
    *,
    target_col: str = COL_TARGET,
    show_neither: bool = False,
) -> Figure:
    """Visualização "Venn-like" simplificada: barras por zona de
    sobreposição entre duas flags binárias.

    Mostra 3 barras por padrão (``A∧B``, ``só A``, ``só B``); ``só A`` e
    ``só B`` representam jogos que pertencem a um conjunto mas não ao
    outro. ``show_neither=True`` adiciona a zona "nem A nem B" (em geral
    pouco útil porque é grande demais). A zona "ambos" recebe cor de
    destaque, e cada barra é anotada com nº de jogos e taxa de
    recomendação local."""
    pair = combo_rate_flag_pair(df, flag_a, flag_b, min_games=0, target_col=target_col)
    if pair.empty:
        raise ValueError("Sem dados para o par de flags solicitado.")

    name_a = flag_a.replace("has_", "")
    name_b = flag_b.replace("has_", "")
    order = [
        f"{name_a} + {name_b}",
        f"apenas {name_a}",
        f"apenas {name_b}",
    ]
    if show_neither:
        order.append(f"nem {name_a} nem {name_b}")

    pair = pair.set_index("zone").reindex(order).dropna(subset=["n_games"])
    if pair.empty:
        raise ValueError("Zonas solicitadas não têm dados.")

    highlight = f"{name_a} + {name_b}"
    colors = [_HIGHLIGHT_COLOR if z == highlight else _BAR_COLOR for z in pair.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(pair.index, pair["n_games"], color=colors)
    for bar, n, rate in zip(bars, pair["n_games"], pair["rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(n):,}\nrec={rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _setup_axes(
        ax,
        title=f"Sobreposição '{name_a}' × '{name_b}' (jogos únicos)",
        xlabel="Zona",
        ylabel="Nº de jogos",
    )
    ax.set_ylim(0, pair["n_games"].max() * 1.20)
    fig.tight_layout()
    return fig


# --- Helpers internos ---------------------------------------------------
def _safe_lift(rate: float, baseline: float) -> float:
    """Lift = rate / baseline. Devolve ``NaN`` quando o baseline é inválido."""
    if baseline is None or baseline <= 0 or pd.isna(baseline):
        return float("nan")
    return float(rate) / float(baseline)


def _zone_label(flag_a: str, flag_b: str, val_a: int, val_b: int) -> str:
    name_a = flag_a.replace("has_", "")
    name_b = flag_b.replace("has_", "")
    if val_a and val_b:
        return f"{name_a} + {name_b}"
    if val_a and not val_b:
        return f"apenas {name_a}"
    if val_b and not val_a:
        return f"apenas {name_b}"
    return f"nem {name_a} nem {name_b}"
