"""Pipeline de descoberta de padrões.

Carrega o dataset processado, ranqueia as top combinações de 2
características associadas a maior taxa de recomendação e gera as
visualizações (top combos + sobreposições "Venn-like" para pares
selecionados de flags) em ``docs/figures/``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.eda import COL_TARGET
from src.patterns import (
    DEFAULT_MIN_GAMES,
    plot_flag_pair_overlap,
    plot_top_combinations,
    top_combinations,
)


ROOT = Path(__file__).parent
PROCESSED_PATH = ROOT / "data" / "processed" / "steam_reviews_processed.csv"
FIGURES_DIR = ROOT / "docs" / "figures"

# Pares de flags interessantes para o gráfico Venn-like. Ajuste à vontade
# — o pipeline pula silenciosamente os pares cujas colunas não existem.
VENN_PAIRS = [
    ("has_indie", "has_coop"),
    ("has_indie", "has_multiplayer"),
    ("has_action", "has_rpg"),
]

TOP_N = 10


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {PROCESSED_PATH}. "
            "Rode primeiro `python main_preprocessing.py`."
        )

    df = pd.read_csv(PROCESSED_PATH)
    baseline = df[COL_TARGET].mean()
    print(f"Carregados {len(df):,} reviews | baseline = {baseline:.1%}")
    print(f"Filtro mínimo: {DEFAULT_MIN_GAMES} jogos únicos por combinação")
    print()

    top = top_combinations(df, top_n=TOP_N, min_games=DEFAULT_MIN_GAMES)
    if top.empty:
        print("Nenhuma combinação passou no filtro mínimo de jogos.")
        return

    print(f"=== Top {TOP_N} combinações por taxa de recomendação ===")
    display = top.assign(
        rate=top["rate"].map(lambda v: f"{v:.1%}"),
        lift=top["lift"].map(lambda v: f"{v:.2f}x"),
        n_reviews=top["n_reviews"].map(lambda v: f"{v:,}"),
        n_games=top["n_games"].map(lambda v: f"{v:,}"),
    )
    print(display.to_string(index=False))
    print()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig = plot_top_combinations(top, baseline=baseline)
    path = FIGURES_DIR / "top_combinations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")

    for fa, fb in VENN_PAIRS:
        if fa not in df.columns or fb not in df.columns:
            print(f"  skipped overlap {fa}x{fb}: coluna ausente")
            continue
        try:
            fig = plot_flag_pair_overlap(df, fa, fb)
        except ValueError as exc:
            print(f"  skipped overlap {fa}x{fb}: {exc}")
            continue
        name = f"overlap_{fa.replace('has_', '')}_{fb.replace('has_', '')}.png"
        path = FIGURES_DIR / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {path}")


if __name__ == "__main__":
    main()
