"""Pipeline de EDA.

Carrega o dataset processado, gera os gráficos exploratórios definidos em
``src.eda`` salvando-os em ``docs/figures/`` e imprime estatísticas resumo
no console (taxa global, top flags por taxa de recomendação, melhores e
piores faixas de preço e período de lançamento).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.eda import (
    COL_TARGET,
    plot_recommendation_rate_by_genre,
    plot_recommendation_rate_by_price_range,
    plot_recommendation_rate_by_release_period,
    plot_top_genres_by_review_count,
)
from src.preprocessing import TAG_FLAGS


ROOT = Path(__file__).parent
PROCESSED_PATH = ROOT / "data" / "processed" / "steam_reviews_processed.csv"
FIGURES_DIR = ROOT / "docs" / "figures"

# (nome_do_arquivo, função_de_plot). Mantém a lista plana e fácil de estender.
PLOTS = [
    ("rec_rate_by_genre.png", plot_recommendation_rate_by_genre),
    ("rec_rate_by_price.png", plot_recommendation_rate_by_price_range),
    ("rec_rate_by_period.png", plot_recommendation_rate_by_release_period),
    ("top_genres_by_review_count.png", plot_top_genres_by_review_count),
]


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {PROCESSED_PATH}. "
            "Rode primeiro `python main_preprocessing.py`."
        )

    print(f"Carregando dataset processado: {PROCESSED_PATH}")
    df = pd.read_csv(PROCESSED_PATH)
    print(f"  -> {len(df):,} linhas, {df.shape[1]} colunas")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSalvando figuras em {FIGURES_DIR}")
    for filename, plot_fn in PLOTS:
        try:
            fig = plot_fn(df)
        except ValueError as exc:
            # Coluna esperada ausente — pula sem derrubar o pipeline.
            print(f"  skipped: {filename} ({exc})")
            continue
        path = FIGURES_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved:   {path}")

    print()
    _print_summary(df)


def _print_summary(df: pd.DataFrame) -> None:
    """Imprime estatísticas-chave para sanity-check rápido pós-EDA."""
    baseline = df[COL_TARGET].mean()
    print("=== Resumo estatístico ===")
    print(f"Taxa global de recomendação: {baseline:.1%}  (n={len(df):,})")

    # Top 5 flags has_* ranqueadas por taxa de recomendação.
    flag_cols = [c for c in TAG_FLAGS if c in df.columns]
    if flag_cols:
        rows = []
        for flag in flag_cols:
            mask = df[flag] == 1
            if mask.any():
                rows.append(
                    (flag, df.loc[mask, COL_TARGET].mean(), int(mask.sum()))
                )
        if rows:
            ranking = (
                pd.DataFrame(rows, columns=["flag", "rate", "n"])
                .sort_values("rate", ascending=False)
                .head(5)
            )
            print("\nTop 5 flags por taxa de recomendação:")
            for _, r in ranking.iterrows():
                print(f"  {r['flag']:<20s} rate={r['rate']:.1%}  n={r['n']:,}")

    _print_best_worst(df, "price_band", "faixa de preço")
    _print_best_worst(df, "release_year_band", "período de lançamento")


def _print_best_worst(df: pd.DataFrame, col: str, label: str) -> None:
    if col not in df.columns:
        return
    by_cat = df.groupby(col)[COL_TARGET].mean()
    if by_cat.empty:
        return
    best, worst = by_cat.idxmax(), by_cat.idxmin()
    print(f"\nMelhor {label}: {best} ({by_cat[best]:.1%})")
    print(f"Pior   {label}: {worst} ({by_cat[worst]:.1%})")


if __name__ == "__main__":
    main()
