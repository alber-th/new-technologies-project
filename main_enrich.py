"""Pipeline de enriquecimento de metadados via Steam Store API.

Lê os arquivos da pasta ``data/raw/game_rvw_csvs/`` (um CSV por jogo,
nomeado ``<appid>_<Nome>.csv``), extrai os ``app_id`` únicos e faz uma
chamada à Steam Store API para cada um, materializando o resultado em
``data/raw/games_metadata.csv``.

O resultado é a "ponte" que falta entre o dataset Kaggle (somente
reviews) e o pipeline analítico (que pressupõe gênero, tags, preço,
data de lançamento e estúdio).

**Como rodar:**

.. code-block:: bash

    python main_enrich.py

A primeira execução demora ~5 minutos para os ~192 jogos do dataset
(serializadas com 1.5s entre chamadas para respeitar limites práticos
da API). Execuções subsequentes são instantâneas se o CSV já existir —
o módulo é idempotente e só busca os ``app_id`` que ainda não estão lá.
"""
from __future__ import annotations

from pathlib import Path

from src.enrichment import enrich_games, extract_app_id_from_filename


ROOT = Path(__file__).parent
REVIEWS_DIR = ROOT / "data" / "raw" / "game_rvw_csvs"
METADATA_PATH = ROOT / "data" / "raw" / "games_metadata.csv"


def main() -> None:
    if not REVIEWS_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de reviews não encontrada: {REVIEWS_DIR}. "
            "Baixe e extraia o dataset do Kaggle em data/raw/ antes de rodar."
        )

    # Extrai app_ids únicos dos nomes de arquivo — set evita IDs duplicados
    # e ignora silenciosamente arquivos auxiliares (README, etc.).
    app_ids = sorted(
        {
            aid
            for f in REVIEWS_DIR.glob("*.csv")
            if (aid := extract_app_id_from_filename(f)) is not None
        }
    )
    if not app_ids:
        raise RuntimeError(
            f"Nenhum app_id pôde ser extraído de {REVIEWS_DIR}. "
            "Esperado o padrão '<appid>_<Nome>.csv'."
        )
    print(f"Encontrados {len(app_ids)} jogos em {REVIEWS_DIR.name}/")

    if METADATA_PATH.exists():
        print(
            f"Arquivo de metadados existente será reutilizado e completado: "
            f"{METADATA_PATH.name}"
        )

    def progress(i: int, total: int, app_id: int, status: str) -> None:
        print(f"  [{i:>3}/{total}] app_id={app_id:<10} {status}")

    df = enrich_games(
        app_ids,
        METADATA_PATH,
        resume=True,
        on_progress=progress,
    )

    n_with_data = int(df["app_name"].notna().sum())
    print()
    print(f"Salvo: {METADATA_PATH}")
    print(f"  → {len(df):,} linhas total, {n_with_data:,} com metadados válidos")
    print(f"  → {len(df) - n_with_data:,} sem dados (jogos removidos da loja)")


if __name__ == "__main__":
    main()
