"""Funções de carregamento e diagnóstico do dataset bruto.

Duas estratégias de carregamento convivem:

* ``load_raw_data`` — leitura simples de um único CSV (mantida para
  compatibilidade e uso pontual em diagnósticos).
* ``load_combined_data`` — leitura **da pasta de reviews por jogo** +
  **CSV de metadados gerado pelo enriquecimento via Steam Store API**.
  Concatena tudo em um único ``DataFrame`` e renomeia as colunas para
  o schema esperado por ``src/preprocessing.py``. É a função usada
  pelo pipeline analítico real desde a adoção do dataset
  ``smeeeow/steam-game-reviews`` (que entrega um CSV por jogo, sem
  metadados embutidos).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.enrichment import extract_app_id_from_filename


# Colunas das CSVs de review que mantemos no DataFrame combinado. Tudo
# que está fora dessa lista (notavelmente o texto livre da review e
# campos de moderação que não usamos) é descartado no momento da leitura
# para economizar memória — o dataset bruto tem ~1.5 GB e o grosso é
# texto livre.
_REVIEW_USECOLS_DEFAULT = [
    "recommendationid",
    "voted_up",
    "timestamp_created",
    "author.steamid",
    "author.playtime_forever",
    "author.num_reviews",
    "received_for_free",
    "written_during_early_access",
    "language",
]

# Renomeação para o schema canônico do preprocessing. As constantes
# ``COL_*`` em ``src/preprocessing.py`` precisam refletir os destinos
# desse mapeamento — qualquer mudança aqui deve ser propagada lá.
_RENAME_TO_CANONICAL = {
    "recommendationid": "review_id",
    "voted_up": "recommended",
    "timestamp_created": "review_date",
    "author.steamid": "author_id",
}


def load_raw_data(path: str) -> pd.DataFrame:
    """Carrega um CSV individual em um ``DataFrame``.

    Usado pelo ``main_diagnostico`` para inspecionar um arquivo
    específico. Para a pipeline analítica completa, ver
    ``load_combined_data``."""
    return pd.read_csv(path)


def load_combined_data(
    reviews_dir: Path | str,
    metadata_path: Path | str,
    *,
    usecols_reviews: list[str] | None = None,
    drop_review_text: bool = True,
) -> pd.DataFrame:
    """Concatena os CSVs de reviews (um por jogo) + metadados do
    enriquecimento e devolve um ``DataFrame`` com o schema esperado
    pelo preprocessing.

    Etapas:

    1. Itera por ``reviews_dir/*.csv``, extraindo ``app_id`` do nome de
       arquivo (``<appid>_<Nome>.csv``).
    2. Lê apenas as colunas necessárias (``usecols_reviews``) para
       conter o uso de memória do dataset original (~1.5 GB).
    3. Concatena os pedaços em um único ``DataFrame``.
    4. Renomeia as colunas do schema bruto para o schema canônico
       (``voted_up`` → ``recommended``, etc.).
    5. Converte ``timestamp_created`` (unix epoch) para ``datetime``.
    6. Faz ``merge(on='app_id', how='left')`` com o CSV de metadados
       — jogos sem metadado retornam ``NaN`` nessas colunas e o
       preprocessing lida com isso preenchendo ``Unknown``.

    Parâmetros relevantes:

    * ``drop_review_text``: ``True`` por padrão. Mantemos ``False`` só
      em análises pontuais que precisem do texto livre.
    """
    reviews_dir = Path(reviews_dir)
    metadata_path = Path(metadata_path)

    csv_paths = sorted(reviews_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"Nenhum CSV em {reviews_dir}. "
            "Baixe e extraia o dataset do Kaggle em data/raw/."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Arquivo de metadados não encontrado: {metadata_path}. "
            "Rode `python main_enrich.py` primeiro para gerá-lo."
        )

    cols = list(usecols_reviews or _REVIEW_USECOLS_DEFAULT)
    if not drop_review_text and "review" not in cols:
        cols.append("review")
    cols_set = set(cols)

    parts: list[pd.DataFrame] = []
    for path in csv_paths:
        app_id = extract_app_id_from_filename(path)
        if app_id is None:
            continue
        # ``usecols=lambda`` ignora colunas ausentes silenciosamente.
        df_part = pd.read_csv(path, usecols=lambda c: c in cols_set)
        df_part["app_id"] = app_id
        parts.append(df_part)

    if not parts:
        raise RuntimeError(
            f"Nenhum CSV pôde ser carregado de {reviews_dir}."
        )

    reviews = pd.concat(parts, ignore_index=True)
    reviews = reviews.rename(columns=_RENAME_TO_CANONICAL)

    if "review_date" in reviews.columns:
        # timestamp_created é um unix epoch (segundos); converte para
        # datetime para alinhar com o que ``preprocessing.clean_raw_data``
        # espera ao chamar ``pd.to_datetime``.
        reviews["review_date"] = pd.to_datetime(
            reviews["review_date"], unit="s", errors="coerce"
        )

    metadata = pd.read_csv(metadata_path)
    combined = reviews.merge(metadata, on="app_id", how="left")
    return combined


def show_basic_info(df: pd.DataFrame, sample_rows: int = 5) -> None:
    """Imprime no console um diagnóstico básico do ``DataFrame``:
    dimensões, ``dtypes``, contagem de nulos por coluna e uma amostra
    das primeiras linhas."""
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
