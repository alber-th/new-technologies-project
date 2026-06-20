"""Pipeline de pré-processamento.

Carrega o CSV bruto de ``data/raw/``, aplica limpeza e engenharia de
features e persiste o resultado em ``data/processed/``. No final, imprime
um resumo das novas colunas para inspeção rápida.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_loading import load_raw_data
from src.preprocessing import TAG_FLAGS, clean_raw_data, engineer_features


ROOT = Path(__file__).parent
RAW_PATH = ROOT / "data" / "raw" / "steam_reviews.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "steam_reviews_processed.csv"


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {RAW_PATH}. "
            "Baixe o dataset do Kaggle e coloque-o em data/raw/ com este nome."
        )

    print(f"[1/4] Carregando dados brutos de {RAW_PATH}")
    df_raw = load_raw_data(str(RAW_PATH))
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
    print(f"      → {PROCESSED_PATH.stat().st_size / 1024:.1f} KB gravados")

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
