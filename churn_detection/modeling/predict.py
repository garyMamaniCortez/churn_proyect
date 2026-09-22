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
    open_cycles["riesgo"] = pd.cut(
        open_cycles["probabilidad_churn"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["bajo", "medio", "alto"],
        include_lowest=True,
    )

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
