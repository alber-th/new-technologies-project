"""Limpeza e engenharia de features do dataset bruto de reviews da Steam.

Cobre duas etapas:

* ``clean_raw_data`` — remove duplicatas, parseia datas, binariza o campo de
  recomendação e trata nulos.
* ``engineer_features`` — deriva faixas de preço e de ano de lançamento,
  flags binárias de tags/gêneros e uma medida simples de popularidade do jogo.

Nomes de colunas esperados são centralizados nas constantes abaixo. Se o CSV
baixado usar nomenclatura diferente, basta ajustá-las aqui — as funções são
robustas a colunas ausentes (apenas pulam as etapas que dependem delas).
"""
from __future__ import annotations

import ast

import numpy as np
import pandas as pd


# --- Nomes esperados de colunas no CSV bruto ----------------------------
COL_REVIEW_ID = "review_id"
COL_APP_ID = "app_id"
COL_APP_NAME = "app_name"
COL_AUTHOR = "author_id"
COL_REVIEW_DATE = "review_date"
COL_RECOMMENDED = "recommended"
COL_RELEASE_DATE = "release_date"
COL_PRICE = "price"
COL_TAGS = "tags"
COL_GENRES = "genres"
COL_DEVELOPER = "developer"
COL_PUBLISHER = "publisher"

# Mapa "flag binária" → palavras-chave que indicam presença da característica
# na string de tags/gêneros. Use formas minúsculas; o parser normaliza tudo.
TAG_FLAGS: dict[str, tuple[str, ...]] = {
    "has_multiplayer": ("multiplayer",),
    "has_singleplayer": ("single-player", "singleplayer"),
    "has_rpg": ("rpg", "role-playing"),
    "has_action": ("action",),
    "has_coop": ("co-op", "coop", "cooperative"),
    "has_indie": ("indie",),
}


# --- Limpeza ------------------------------------------------------------
def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as limpezas básicas necessárias antes da análise.

    Etapas:

    1. Remove duplicatas. Se houver ``review_id``, ele é a chave; caso
       contrário, usa a combinação ``app_id + author + review_date`` como
       chave aproximada.
    2. Converte colunas de data para ``datetime`` com ``errors='coerce'``,
       transformando valores inválidos em ``NaT`` em vez de quebrar a
       pipeline.
    3. Binariza a recomendação (``1`` = recomendado, ``0`` = não recomendado).
    4. Descarta linhas sem recomendação ou sem identificador de jogo, pois
       elas não contribuem para responder a pergunta de negócio.
    5. Preenche campos textuais ausentes com ``"Unknown"`` para preservar a
       linha sem distorcer a contagem de jogos.
    """
    df = df.copy()

    # 1) Deduplicação. Prioriza review_id por ser único por natureza.
    if COL_REVIEW_ID in df.columns:
        df = df.drop_duplicates(subset=[COL_REVIEW_ID])
    else:
        dedup_cols = [
            c for c in (COL_APP_ID, COL_AUTHOR, COL_REVIEW_DATE) if c in df.columns
        ]
        if dedup_cols:
            df = df.drop_duplicates(subset=dedup_cols)

    # 2) Datas → datetime.
    for col in (COL_REVIEW_DATE, COL_RELEASE_DATE):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 3) Recomendação → binária.
    if COL_RECOMMENDED in df.columns:
        df[COL_RECOMMENDED] = _to_binary_recommendation(df[COL_RECOMMENDED])

    # 4) Descarta linhas inúteis (sem target ou sem jogo).
    required = [c for c in (COL_RECOMMENDED, COL_APP_ID) if c in df.columns]
    if required:
        df = df.dropna(subset=required)

    # 5) Preenche textuais ausentes com "Unknown".
    for col in (COL_DEVELOPER, COL_PUBLISHER, COL_TAGS, COL_GENRES):
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df.reset_index(drop=True)


# --- Engenharia de features ---------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria colunas derivadas úteis para a análise.

    Adiciona, quando os campos de origem existem:

    * ``price_numeric`` — preço convertido para ``float``.
    * ``price_band`` — faixa categórica de preço (Grátis / Barato / Médio / Caro).
    * ``release_year`` — ano de lançamento extraído.
    * ``release_year_band`` — faixa categórica de ano de lançamento.
    * ``has_*`` — flags binárias derivadas de tags/gêneros (ver ``TAG_FLAGS``).
    * ``num_reviews_game`` — número de reviews por jogo no dataset (proxy
      simples de popularidade).
    """
    df = df.copy()

    # Faixa de preço: regras absolutas em vez de quantis para gerar bandas
    # interpretáveis no relatório ("até R$ 20", "até R$ 60", etc.). Jogos
    # gratuitos viram uma banda à parte porque o padrão de avaliação tende
    # a ser bem diferente (volume alto, qualidade variada).
    if COL_PRICE in df.columns:
        price = _parse_price(df[COL_PRICE])
        df["price_numeric"] = price
        df["price_band"] = np.select(
            [price <= 0, price <= 20, price <= 60],
            ["Grátis", "Barato", "Médio"],
            default="Caro",
        )
        # np.select cai no default quando o valor é NaN (todas as
        # comparações retornam False). Mantém NaN explicitamente.
        df.loc[price.isna(), "price_band"] = np.nan

    # Faixa de ano de lançamento — recortes alinhados com gerações de
    # plataformas e o crescimento da Steam (boom indie pós-2010).
    if COL_RELEASE_DATE in df.columns:
        year = df[COL_RELEASE_DATE].dt.year
        df["release_year"] = year
        df["release_year_band"] = np.select(
            [year < 2010, year <= 2015, year <= 2020],
            ["Antes de 2010", "2010-2015", "2016-2020"],
            default="Depois de 2020",
        )
        df.loc[year.isna(), "release_year_band"] = np.nan

    # Flags binárias de tags/gêneros. Usa o campo mais granular disponível.
    tag_source = _select_tag_source(df)
    if tag_source is not None:
        tag_sets = tag_source.map(_parse_tags)
        for flag, keywords in TAG_FLAGS.items():
            df[flag] = tag_sets.map(
                lambda tags, kw=keywords: int(any(k in tags for k in kw))
            ).astype("int8")

    # Popularidade do jogo: nº total de reviews no dataset para aquele app_id.
    if COL_APP_ID in df.columns:
        df["num_reviews_game"] = (
            df.groupby(COL_APP_ID)[COL_APP_ID].transform("size").astype("int64")
        )

    return df


# --- Helpers internos ---------------------------------------------------
def _to_binary_recommendation(series: pd.Series) -> pd.Series:
    """Mapeia o campo de recomendação para {0, 1} suportando formatos comuns:
    ``bool``, strings (``Recommended`` / ``Not Recommended`` / ``Positive`` /
    ``Negative``) e códigos numéricos (1 / -1)."""
    if series.dtype == bool:
        return series.astype("Int8")

    str_map = {
        "recommended": 1,
        "not recommended": 0,
        "positive": 1,
        "negative": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "-1": 0,
    }
    return (
        series.astype(str).str.strip().str.lower().map(str_map).astype("Int8")
    )


def _parse_price(series: pd.Series) -> pd.Series:
    """Converte preço para ``float``. Aceita números, strings com símbolos
    de moeda e categoriza ``free`` / ``gratis`` como ``0``. Valores não
    interpretáveis viram ``NaN`` (não ``0``) para não enviesar a análise."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    s = series.astype(str).str.strip().str.lower()
    s = s.replace(
        {"free": "0", "gratis": "0", "grátis": "0", "free to play": "0"}
    )
    s = s.str.replace(r"[^\d.,]", "", regex=True)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _parse_tags(raw) -> set[str]:
    """Converte o campo de tags em um conjunto de strings em lowercase.
    Aceita listas/tuplas Python, strings JSON-like (``"['Action','RPG']"``)
    e CSV (``"Action, RPG"``)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return set()
    if isinstance(raw, (list, tuple, set)):
        return {str(t).strip().lower() for t in raw}

    text = str(raw).strip()
    if not text or text.lower() == "unknown":
        return set()

    if text.startswith(("[", "(")):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return {str(t).strip().lower() for t in parsed}
        except (SyntaxError, ValueError):
            pass

    return {
        p.strip().lower()
        for p in text.replace(";", ",").split(",")
        if p.strip()
    }


def _select_tag_source(df: pd.DataFrame) -> pd.Series | None:
    """Escolhe a melhor coluna disponível para extrair tags. Prefere
    ``tags`` sobre ``genres`` por ser mais granular."""
    for col in (COL_TAGS, COL_GENRES):
        if col in df.columns:
            return df[col]
    return None
