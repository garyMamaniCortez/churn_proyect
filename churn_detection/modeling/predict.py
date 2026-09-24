"""Score currently open ("censurado") membership cycles with the trained churn model.

These are exactly the rows `churn_dataset.py` excluded from training: clients
whose current cycle hasn't been resolved yet (either still active, or lapsed
but still inside the grace window). They're precisely who a retention team
would want a risk score for -- this module is the "propose a way to use the
results" piece of the project, not a second training path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from churn_detection.config import MODELS_DIR, PROCESSED_DATA_DIR
from churn_detection.modeling.train import (
    FEATURE_COLUMNS,
    CalibratedChurnModel,
    KerasBinaryClassifier,
    SigmoidCalibrator,
    _Log1pSkewedColumns,
)

app = typer.Typer(help="Score open (unresolved) membership cycles with the trained churn model.")


def _risk_bins(probabilidad_churn: pd.Series) -> tuple[list[float], list[str]]:
    """Bin edges and labels for risk level, based on the TERCILES of this
    batch's own distribution of estimated probabilities -- "alto" is the
    riskiest third of the clients being scored right now, not a fixed point
    on the 0-1 probability scale. A probability of 0.67 does not become
    "riesgo alto" just because 0-1 was cut into three equal parts; it
    becomes "riesgo alto" only if it is actually among the highest
    predicted probabilities in the batch currently being scored. This means
    the cutoffs are only meaningful for comparing clients within the same
    run, which matches the deployment proposal already described in the
    project's report: the retention team adjusts how far down the ranked
    list it works based on its own contact capacity, not against a fixed
    probability value.

    Falls back to two levels (bajo/alto split at the median) if the batch's
    probabilities are too concentrated for three distinct tercile edges
    (e.g. a very small or near-uniform batch) -- `pd.cut` requires strictly
    increasing edges, and silently coercing ties there would misclassify
    rows rather than fail loudly.
    """
    low_cut, high_cut = probabilidad_churn.quantile([1 / 3, 2 / 3])
    if low_cut < high_cut:
        return [0.0, low_cut, high_cut, 1.0], ["bajo", "medio", "alto"]

    logger.warning(
        "Probabilidades del lote demasiado concentradas para terciles "
        f"distintos (P33={low_cut:.4f}, P66={high_cut:.4f}); usando dos "
        "niveles de riesgo en torno a la mediana en vez de tres."
    )
    median_cut = probabilidad_churn.median()
    return [0.0, median_cut, 1.0], ["bajo", "alto"]


def _register_train_classes_under_main() -> None:
    """`models/churn_model.joblib` is produced by running train.py as a
    script (`python -m churn_detection.modeling.train`, what dvc.yaml/
    Makefile do), which makes Python treat that execution of train.py as
    the "__main__" module -- so every custom class it defines
    (CalibratedChurnModel, SigmoidCalibrator, KerasBinaryClassifier,
    _Log1pSkewedColumns) gets pickled as if it lived in "__main__" rather
    than at its real dotted path. This process's own "__main__" is this
    predict.py script, which never defined those classes, so a plain
    `joblib.load` fails with `AttributeError: Can't get attribute
    'CalibratedChurnModel' on <module '__main__' ...>`. Attaching the real
    classes (imported normally above, so they ARE the correct objects, just
    exposed under an extra name) to this process's `sys.modules["__main__"]`
    before loading lets pickle's lookup succeed.
    """
    main_module = sys.modules["__main__"]
    for cls in (_Log1pSkewedColumns, KerasBinaryClassifier, SigmoidCalibrator, CalibratedChurnModel):
        setattr(main_module, cls.__name__, cls)


@app.command()
def score_open_cycles(
    input_path: Path = PROCESSED_DATA_DIR / "churn_ciclos.csv",
    model_path: Path = MODELS_DIR / "churn_model.joblib",
    output_path: Path = PROCESSED_DATA_DIR / "predicciones_churn.csv",
) -> None:
    df = pd.read_csv(input_path)
    open_cycles = df[df["estado_ciclo"] == "censurado"].copy()
    if open_cycles.empty:
        logger.warning("No hay ciclos 'censurado' para puntuar.")
        return

    _register_train_classes_under_main()
    pipeline = joblib.load(model_path)
    open_cycles["probabilidad_churn"] = pipeline.predict_proba(open_cycles[FEATURE_COLUMNS])[:, 1]
    bins, labels = _risk_bins(open_cycles["probabilidad_churn"])
    open_cycles["riesgo"] = pd.cut(
        open_cycles["probabilidad_churn"], bins=bins, labels=labels, include_lowest=True
    )
    logger.info(f"Cortes de riesgo de esta corrida (terciles del lote puntuado): {bins}")

    output_cols = [
        "persona_id",
        "inscripcion_id",
        "fecha_vencimiento",
        "probabilidad_churn",
        "riesgo",
    ]
    result = open_cycles[output_cols].sort_values("probabilidad_churn", ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.success(f"Wrote {len(result)} risk scores to {output_path}")
    logger.info(f"Distribución de riesgo:\n{result['riesgo'].value_counts()}")


if __name__ == "__main__":
    app()
