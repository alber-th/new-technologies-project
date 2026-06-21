"""Pipeline de pré-processamento.

Carrega o dataset combinado (reviews por jogo + metadados enriquecidos),
aplica limpeza e engenharia de features e persiste o resultado em
``data/processed/steam_reviews_processed.csv``. No final, imprime um
resumo das novas colunas para inspeção rápida.

**Pré-requisito**: ter rodado ``python main_enrich.py`` ao menos uma
vez para materializar ``data/raw/games_metadata.csv``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loading import load_combined_data
from src.preprocessing import TAG_FLAGS, clean_raw_data, engineer_features


ROOT = Path(__file__).parent
REVIEWS_DIR = ROOT / "data" / "raw" / "game_rvw_csvs"
METADATA_PATH = ROOT / "data" / "raw" / "games_metadata.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "steam_reviews_processed.csv"


def main() -> None:
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
            "Rode `python main_enrich.py` antes."
        )

    print(f"[1/4] Carregando reviews + metadados")
    df_raw = load_combined_data(REVIEWS_DIR, METADATA_PATH)
    print(f"      → {len(df_raw):,} linhas, {df_raw.shape[1]} colunas")

    print("[2/4] Aplicando limpeza (dedup, datas, target binário, nulos)")
    df_clean = clean_raw_data(df_raw)
    removed = len(df_raw) - len(df_clean)
    print(f"      → {len(df_clean):,} linhas restantes ({removed:,} removidas)")

    print("[3/4] Engenharia de features")
    df_final = engineer_features(df_clean)
    new_cols = [c for c in df_final.columns if c not in df_clean.columns]
    print(f"      → {len(new_cols)} novas colunas: {new_cols}")

    print(f"[4/4] Salvando em {PROCESSED_PATH}")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(PROCESSED_PATH, index=False)
    size_mb = PROCESSED_PATH.stat().st_size / (1024 * 1024)
    print(f"      → {size_mb:.1f} MB gravados")

    print()
    print("=== Resumo das novas colunas ===")
    for col in new_cols:
        print(f"\n[{col}]  dtype={df_final[col].dtype}")
        # Categóricas / flags: distribuição de valores.
        # Numéricas com muitos valores únicos: estatísticas resumidas.
        is_flag = col in TAG_FLAGS
        is_categorical = df_final[col].nunique(dropna=True) <= 10
        if is_flag or is_categorical:
            print(df_final[col].value_counts(dropna=False).to_string())
        else:
            print(df_final[col].describe().to_string())


if __name__ == "__main__":
    main()
