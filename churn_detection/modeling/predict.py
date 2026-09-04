"""Score currently open ("censurado") membership cycles with the trained churn model.

These are exactly the rows `churn_dataset.py` excluded from training: clients
whose current cycle hasn't been resolved yet (either still active, or lapsed
but still inside the grace window). They're precisely who a retention team
would want a risk score for -- this module is the "propose a way to use the
results" piece of the project, not a second training path.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import typer

from churn_detection.config import MODELS_DIR, PROCESSED_DATA_DIR
from churn_detection.modeling.train import FEATURE_COLUMNS

app = typer.Typer(help="Score open (unresolved) membership cycles with the trained churn model.")


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
