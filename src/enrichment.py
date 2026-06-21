"""Enriquecimento de metadados de jogos via Steam Store API.

**Contexto.** O dataset bruto do Kaggle (``smeeeow/steam-game-reviews``)
fornece **apenas as reviews** organizadas em um CSV por jogo
(``data/raw/game_rvw_csvs/<appid>_<NomeDoJogo>.csv``). Não há colunas
de gênero, tags, preço, data de lançamento ou estúdio — exatamente as
dimensões que a análise principal precisa para responder à pergunta de
negócio. Este módulo busca esses metadados diretamente na **Steam Store
API pública** e materializa um CSV alinhado por ``app_id``, que depois
é mesclado às reviews pelo ``data_loading.load_combined_data``.

**API utilizada.** O endpoint público não-autenticado é:

    https://store.steampowered.com/api/appdetails?appids=<id>&filters=...

A resposta vem como JSON aninhado com a forma ``{"<id>": {"success":
bool, "data": {...}}}``. Os campos relevantes para o nosso schema são
``name``, ``genres``, ``categories``, ``release_date``, ``price_overview``,
``developers`` e ``publishers``.

**Rate limit.** A Steam não documenta oficialmente, mas convenções da
comunidade apontam algo em torno de 200 requisições por 5 minutos. Por
isso o módulo serializa as chamadas e aceita um ``sleep_seconds``
configurável (default ``1.5``s) entre requisições — bem abaixo desse
limite e suficiente para finalizar os ~192 jogos do dataset em ~5 min.

**Idempotência.** ``enrich_games`` aceita ``resume=True``: se o CSV de
saída já existir, ele preserva as linhas já presentes e refaz a chamada
apenas para os ``app_id`` faltantes. Permite retomar a coleta após uma
interrupção sem perder o que já foi baixado.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import requests


STEAM_API_URL = "https://store.steampowered.com/api/appdetails"
DEFAULT_SLEEP_S = 1.5
DEFAULT_TIMEOUT_S = 10
# Filtros explícitos minimizam o payload: pulamos descrição longa,
# screenshots e outros campos pesados que não usamos.
DEFAULT_FILTERS = "basic,genres,categories,price_overview,release_date"

# Nomes dos arquivos do dataset Kaggle seguem o padrão
# "<appid>_<NomeDoJogo>.csv" (ex.: "10_CounterStrike.csv").
_APP_ID_RE = re.compile(r"^(\d+)_")


def extract_app_id_from_filename(path: Path | str) -> int | None:
    """Extrai o ``app_id`` numérico do prefixo do nome de arquivo.

    Devolve ``None`` se o padrão não bater (arquivos auxiliares como
    README ou JSON ficam de fora silenciosamente)."""
    name = Path(path).name
    match = _APP_ID_RE.match(name)
    return int(match.group(1)) if match else None


def fetch_app_details(
    app_id: int,
    session: requests.Session,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any] | None:
    """Faz uma chamada à Steam Store API para um único ``app_id``.

    Retorna o dicionário aninhado ``data`` da resposta da API quando o
    jogo está disponível, ou ``None`` em três cenários:

    * O jogo foi removido da loja e ``success == false``.
    * A resposta veio sem ``data`` (jogos restritos por idade, regiões,
      DLCs sem ficha pública).
    * A API devolveu JSON vazio.

    Erros de rede (``ConnectionError``, ``Timeout``, etc.) são propagados
    para o caller decidir entre retry, log ou skip — manter essa
    responsabilidade fora desta função preserva a função "pura"."""
    response = session.get(
        STEAM_API_URL,
        params={"appids": app_id, "filters": DEFAULT_FILTERS},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json() or {}
    entry = payload.get(str(app_id)) or {}
    if not entry.get("success"):
        return None
    return entry.get("data")


def parse_app_details(data: dict[str, Any] | None, app_id: int) -> dict[str, Any]:
    """Achata o JSON da Steam em uma linha plana com o schema canônico
    esperado por ``src.preprocessing`` (``app_id``, ``app_name``,
    ``genres``, ``tags``, ``release_date``, ``price``, ``developer``,
    ``publisher``).

    Quando ``data`` é ``None``, devolve uma linha com os campos vazios
    mas mantendo o ``app_id`` — o jogo continua válido para os reviews,
    apenas sem metadados (essas linhas viram ``Unknown`` no
    preprocessing)."""
    if data is None:
        return {
            "app_id": app_id,
            "app_name": None,
            "genres": None,
            "tags": None,
            "release_date": None,
            "price": None,
            "developer": None,
            "publisher": None,
        }

    genres = ", ".join(
        (g.get("description") or "").strip()
        for g in (data.get("genres") or [])
        if g.get("description")
    )
    categories = ", ".join(
        (c.get("description") or "").strip()
        for c in (data.get("categories") or [])
        if c.get("description")
    )

    # Preço: 0 para jogos free-to-play; senão pega o ``final`` em cents
    # do ``price_overview`` e converte para a unidade da moeda.
    price_value: float | None
    if data.get("is_free"):
        price_value = 0.0
    elif (po := data.get("price_overview")) is not None:
        price_value = (po.get("final") or 0) / 100
    else:
        price_value = None

    return {
        "app_id": app_id,
        "app_name": data.get("name"),
        "genres": genres or None,
        "tags": categories or None,
        "release_date": (data.get("release_date") or {}).get("date"),
        "price": price_value,
        "developer": ", ".join(data.get("developers") or []) or None,
        "publisher": ", ".join(data.get("publishers") or []) or None,
    }


def enrich_games(
    app_ids: Iterable[int],
    output_csv: Path | str,
    *,
    sleep_seconds: float = DEFAULT_SLEEP_S,
    resume: bool = True,
    on_progress: Callable[[int, int, int, str], None] | None = None,
) -> pd.DataFrame:
    """Orquestra a coleta de metadados para uma lista de ``app_id`` e
    grava o resultado em ``output_csv``.

    * Se ``resume=True`` e o CSV já existir, preserva as linhas
      existentes e busca apenas os IDs faltantes (idempotente).
    * ``on_progress(i, total, app_id, status)`` é chamado após cada
      requisição, útil para feedback no terminal.
    * Erros de rede individuais não derrubam a coleta — o ``app_id``
      correspondente entra com metadados vazios e o status é registrado
      no callback.

    Devolve o ``DataFrame`` final (existentes ∪ recém-coletados)."""
    output_csv = Path(output_csv)
    app_ids = list(app_ids)

    already = pd.DataFrame()
    pending = app_ids
    if resume and output_csv.exists():
        already = pd.read_csv(output_csv)
        if "app_id" in already.columns:
            done = set(already["app_id"].astype(int))
            pending = [i for i in app_ids if i not in done]

    rows: list[dict[str, Any]] = []
    with requests.Session() as session:
        session.headers["User-Agent"] = (
            "new-technologies-project / steam-reviews-eda"
        )
        for idx, app_id in enumerate(pending, 1):
            try:
                data = fetch_app_details(app_id, session)
                row = parse_app_details(data, app_id)
                status = "ok" if data is not None else "no_data"
            except requests.RequestException as exc:
                row = parse_app_details(None, app_id)
                status = f"error: {type(exc).__name__}"
            rows.append(row)
            if on_progress is not None:
                on_progress(idx, len(pending), app_id, status)
            time.sleep(sleep_seconds)

    new_df = pd.DataFrame(rows)
    final = (
        pd.concat([already, new_df], ignore_index=True)
        if not already.empty
        else new_df
    )
    # Em caso de retomada parcial, mantém a versão mais recente do app_id.
    final = final.drop_duplicates(subset=["app_id"], keep="last")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_csv, index=False)
    return final
