"""Funções de carregamento e diagnóstico inicial do dataset bruto da Steam."""
from __future__ import annotations

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Carrega o dataset bruto a partir de um arquivo CSV em ``path``."""
    return pd.read_csv(path)


def show_basic_info(df: pd.DataFrame, sample_rows: int = 5) -> None:
    """Imprime no console um diagnóstico básico do DataFrame: dimensões,
    tipos de dados, contagem de nulos por coluna e uma amostra das primeiras
    linhas."""
    n_rows, n_cols = df.shape
    print(f"Linhas: {n_rows:,} | Colunas: {n_cols}")
    print()

    print("Tipos de dados por coluna:")
    print(df.dtypes.to_string())
    print()

    print("Contagem de nulos por coluna:")
    print(df.isna().sum().to_string())
    print()

    print(f"Amostra das primeiras {sample_rows} linhas:")
    print(df.head(sample_rows).to_string())
