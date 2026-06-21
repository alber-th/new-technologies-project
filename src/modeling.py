"""Modelo baseline para prever se um jogo será recomendado, a partir dos
metadados disponíveis na base processada.

Por design, o modelo é simples e interpretável: ``LogisticRegression`` com
imputação mediana, padronização e ``class_weight='balanced'``. O objetivo
é gerar um ponto de comparação inicial e fornecer uma leitura clara sobre
quais features estão associadas a maior probabilidade de recomendação —
não maximizar performance de previsão.

Limitações conhecidas (documentar no relatório):

* Modelo linear nos preditores; não captura interações tipo
  "cerveja/fralda" sem feature engineering adicional.
* Sem busca de hiperparâmetros nem validação cruzada.
* ``num_reviews_game`` é muito assimétrico — uma transformação log poderia
  ajudar e fica como melhoria futura.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eda import COL_TARGET


RANDOM_STATE = 42
TEST_SIZE = 0.2

# Colunas a NÃO usar como feature: identificadores, texto livre, datas,
# alvo, e versões categóricas redundantes com a numérica equivalente.
_EXCLUDE_COLS: frozenset[str] = frozenset({
    COL_TARGET,
    "review_id",
    "app_id",
    "app_name",
    "author_id",
    "review_date",
    "release_date",
    "tags",
    "genres",
    "developer",
    "publisher",
    "price",               # redundante com price_numeric (este é o limpo)
    "price_band",          # redundante com price_numeric
    "release_year_band",   # redundante com release_year
    "language",            # alta cardinalidade, exige encoding dedicado
})


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """Seleciona as colunas numéricas/binárias adequadas como features.

    Retorno é ordenado alfabeticamente para garantir que o vetor de
    features seja determinístico entre execuções (importante para
    reprodutibilidade da importância).
    """
    candidate = [c for c in df.columns if c not in _EXCLUDE_COLS]
    numeric = [c for c in candidate if pd.api.types.is_numeric_dtype(df[c])]
    return sorted(numeric)


def build_train_test_split(
    df: pd.DataFrame,
    *,
    target_col: str = COL_TARGET,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa em treino/teste mantendo a proporção do alvo (``stratify``).

    Mantém ``X`` como ``DataFrame`` (preserva nomes de coluna) para que a
    análise de importância de features fique legível depois.
    """
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada.")

    feature_cols = select_feature_columns(df)
    if not feature_cols:
        raise ValueError("Nenhuma coluna numérica disponível como feature.")

    # Defesa contra CSV com alvo NaN (não deveria ocorrer pós-preprocessing).
    df = df.dropna(subset=[target_col])

    X = df[feature_cols].copy()
    y = df[target_col].astype(int)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Treina o pipeline baseline: imputação mediana → padronização →
    regressão logística com pesos balanceados.

    Cada passo tem razão de ser:

    * **Imputação mediana** — features como ``price_numeric`` e
      ``release_year`` podem ter ``NaN``.
    * **Padronização** — regressão logística é sensível à escala; também
      torna os coeficientes comparáveis entre features.
    * **``class_weight='balanced'``** — a base de reviews da Steam tem
      forte predominância de positivos; sem balanceamento, o modelo
      tende a prever "1" sempre e atinge alta accuracy sem aprender nada
      útil.
    """
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Aplica o modelo no conjunto de teste e devolve as métricas.

    ``precision``, ``recall`` e ``f1`` são da classe positiva
    (``1 = recomendado``), por ser a classe de interesse da pergunta de
    negócio. ``zero_division=0`` evita warning caso o modelo degenere
    para prever sempre a mesma classe.
    """
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def top_positive_features(
    model: Pipeline,
    feature_names: list[str],
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    """Top N features associadas a MAIOR probabilidade de recomendação.

    Usa os coeficientes da regressão logística, já em escala padronizada
    (graças ao ``StandardScaler`` no pipeline), portanto comparáveis em
    magnitude entre features.
    """
    classifier = model.named_steps["classifier"]
    coefs = classifier.coef_[0]
    out = pd.DataFrame({"feature": feature_names, "coef": coefs})
    return (
        out.sort_values("coef", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
