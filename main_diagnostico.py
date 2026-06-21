"""Script de diagnóstico inicial do dataset combinado.

Carrega o conjunto reviews (``data/raw/game_rvw_csvs/``) + metadados
enriquecidos (``data/raw/games_metadata.csv``) via
``load_combined_data`` e imprime um diagnóstico (dimensões, dtypes,
nulos, amostra). Útil para validar o schema antes de rodar o
preprocessing.

**Pré-requisito**: ter rodado ``python main_enrich.py`` ao menos uma
vez para materializar ``games_metadata.csv``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loading import load_combined_data, show_basic_info


ROOT = Path(__file__).parent
REVIEWS_DIR = ROOT / "data" / "raw" / "game_rvw_csvs"
METADATA_PATH = ROOT / "data" / "raw" / "games_metadata.csv"


def main() -> None:
    # Garante que o print do DataFrame não trunque colunas no terminal.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    if not REVIEWS_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de reviews não encontrada: {REVIEWS_DIR}. "
            "Baixe e extraia o dataset do Kaggle em data/raw/."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de metadados não encontrado: {METADATA_PATH}. "
            "Rode `python main_enrich.py` antes de tudo."
        )

    print(f"Carregando reviews de {REVIEWS_DIR.name}/ + metadados...")
    df = load_combined_data(REVIEWS_DIR, METADATA_PATH)
    show_basic_info(df, sample_rows=10)


if __name__ == "__main__":
    main()
