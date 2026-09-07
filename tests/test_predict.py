import joblib
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from churn_detection.modeling.predict import app
from churn_detection.modeling.train import FEATURE_COLUMNS, LogisticRegressionCandidate


@pytest.fixture
def trained_model_path(tmp_path):
    rng = np.random.default_rng(3)
    n = 40
    df = pd.DataFrame({col: rng.uniform(0, 10, n) for col in FEATURE_COLUMNS})
    labels = rng.integers(0, 2, n)
    pipeline = LogisticRegressionCandidate().build_pipeline()
    pipeline.fit(df, labels)

    model_path = tmp_path / "model.joblib"
    joblib.dump(pipeline, model_path)
    return model_path


@pytest.fixture
def churn_ciclos_path(tmp_path):
    rng = np.random.default_rng(4)
    n = 10
    data = {col: rng.uniform(0, 10, n) for col in FEATURE_COLUMNS}
    data["persona_id"] = range(1, n + 1)
    data["inscripcion_id"] = range(100, 100 + n)
    data["fecha_vencimiento"] = ["2026-08-01"] * n
    data["estado_ciclo"] = ["censurado"] * (n - 2) + ["churned", "renovado"]
    df = pd.DataFrame(data)
    path = tmp_path / "churn_ciclos.csv"
    df.to_csv(path, index=False)
    return path


def test_score_open_cycles_only_scores_censored_rows(trained_model_path, churn_ciclos_path, tmp_path):
    output_path = tmp_path / "predicciones.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--input-path",
            str(churn_ciclos_path),
            "--model-path",
            str(trained_model_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    out = pd.read_csv(output_path)
    assert len(out) == 8  # 10 rows minus 1 "churned" minus 1 "renovado"
    assert set(out.columns) == {
        "persona_id",
        "inscripcion_id",
        "fecha_vencimiento",
        "probabilidad_churn",
        "riesgo",
    }
    assert out["probabilidad_churn"].between(0, 1).all()
    assert out["riesgo"].isin(["bajo", "medio", "alto"]).all()
