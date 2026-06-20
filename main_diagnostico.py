"""Script de diagnóstico inicial do dataset bruto da Steam.

Carrega o CSV bruto em ``data/raw/`` e imprime informações básicas no console
(dimensões, tipos, nulos e amostra). Útil para validar que o arquivo foi
baixado e está sendo lido corretamente antes de iniciar o pré-processamento.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loading import load_raw_data, show_basic_info


RAW_DATA_PATH = Path(__file__).parent / "data" / "raw" / "steam_reviews.csv"


def main() -> None:
    # Garante que o print do DataFrame não trunque colunas no terminal.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {RAW_DATA_PATH}. "
            "Baixe o dataset do Kaggle e coloque-o em data/raw/ com este nome."
        )

    df = load_raw_data(str(RAW_DATA_PATH))
    show_basic_info(df, sample_rows=10)


if __name__ == "__main__":
    main()
