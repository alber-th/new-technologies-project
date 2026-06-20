"""Pipeline de modelagem baseline.

Carrega o dataset processado, separa em treino/teste, treina a regressão
logística baseline, imprime métricas no console e mostra as top 10
features associadas a maior probabilidade de recomendação.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.modeling import (
    RANDOM_STATE,
    build_train_test_split,
    evaluate_model,
    top_positive_features,
    train_baseline_model,
)


ROOT = Path(__file__).parent
PROCESSED_PATH = ROOT / "data" / "processed" / "steam_reviews_processed.csv"


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {PROCESSED_PATH}. "
            "Rode primeiro `python main_preprocessing.py`."
        )

    df = pd.read_csv(PROCESSED_PATH)
    print(f"Carregados {len(df):,} registros, {df.shape[1]} colunas")
    print(f"Random state fixo: {RANDOM_STATE} (reprodutível)\n")

    print("[1/3] Split treino/teste (80/20, stratify pelo alvo)")
    X_train, X_test, y_train, y_test = build_train_test_split(df)
    print(f"      treino: {len(X_train):,} | teste: {len(X_test):,}")
    print(f"      features ({len(X_train.columns)}): {list(X_train.columns)}")
    print(f"      taxa de 1 no treino: {y_train.mean():.1%}")

    print("\n[2/3] Treinando baseline (LogReg + class_weight=balanced)")
    model = train_baseline_model(X_train, y_train)

    print("\n[3/3] Avaliação no conjunto de teste")
    metrics = evaluate_model(model, X_test, y_test)
    _print_metrics(metrics)

    print("\n=== Top 10 features associadas a MAIOR probabilidade de recomendação ===")
    top = top_positive_features(model, list(X_train.columns), top_n=10)
    formatted = top.assign(coef=top["coef"].map(lambda v: f"{v:+.3f}"))
    print(formatted.to_string(index=False))

    print()
    print("Nota: este é um BASELINE simples (regressão logística linear,")
    print("sem tuning de hiperparâmetros nem CV). Limitações estão")
    print("documentadas em src/modeling.py e devem entrar no relatório.")


def _print_metrics(metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    print(f"      accuracy : {metrics['accuracy']:.3f}")
    print(f"      precision: {metrics['precision']:.3f}  (classe positiva = recomendado)")
    print(f"      recall   : {metrics['recall']:.3f}")
    print(f"      f1       : {metrics['f1']:.3f}")
    print(f"      matriz de confusão:")
    print(f"                   pred=0    pred=1")
    print(f"        real=0   {cm[0, 0]:>8,}  {cm[0, 1]:>8,}")
    print(f"        real=1   {cm[1, 0]:>8,}  {cm[1, 1]:>8,}")


if __name__ == "__main__":
    main()
