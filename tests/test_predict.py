import joblib
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from churn_detection.modeling.predict import _risk_bins, app
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
    # Los cortes ya no son 0.33/0.66 fijos: "alto" debe ser el tercio
    # superior del propio lote puntuado, no un punto fijo del intervalo 0-1.
    high_cut = out["probabilidad_churn"].quantile(2 / 3)
    assert out.loc[out["riesgo"] == "alto", "probabilidad_churn"].min() >= high_cut - 1e-9


def test_risk_bins_uses_batch_terciles_not_fixed_thirds():
    probabilidad_churn = pd.Series(np.linspace(0.0, 1.0, 30))

    bins, labels = _risk_bins(probabilidad_churn)

    assert labels == ["bajo", "medio", "alto"]
    expected_low, expected_high = probabilidad_churn.quantile([1 / 3, 2 / 3])
    assert bins == [0.0, expected_low, expected_high, 1.0]
    # A uniform batch would only match the old fixed 0.33/0.66 split by
    # coincidence -- assert it does NOT, to catch a regression back to it.
    assert (expected_low, expected_high) != (0.33, 0.66)


def test_risk_bins_falls_back_to_two_levels_for_a_degenerate_batch():
    # Every client scored identically (e.g. a tiny or near-uniform batch):
    # the terciles collapse to the same value, and pd.cut would raise on
    # non-strictly-increasing edges if used directly with them.
    probabilidad_churn = pd.Series([0.5] * 10)

    bins, labels = _risk_bins(probabilidad_churn)

    assert labels == ["bajo", "alto"]
    assert bins == [0.0, 0.5, 1.0]
    result = pd.cut(probabilidad_churn, bins=bins, labels=labels, include_lowest=True)
    assert not result.isna().any()


def test_score_open_cycles_loads_a_calibrated_model_pickled_under_main(
    churn_ciclos_path, tmp_path
):
    # Regression test: models/churn_model.joblib is produced by running
    # train.py as `python -m churn_detection.modeling.train`, which makes
    # Python treat that run of train.py as the "__main__" module -- so
    # CalibratedChurnModel gets pickled as if it lived in "__main__" rather
    # than at its real dotted path. This process's own "__main__" never
    # defined that class, so a plain joblib.load raises AttributeError
    # unless _register_train_classes_under_main() bridges the gap.
    # Reproduced here by dumping while the class briefly claims to live in
    # "__main__" (matching train.py's run), then loading normally (matching
    # predict.py's separate run).
    import joblib
    from sklearn.isotonic import IsotonicRegression

    from churn_detection.modeling.train import CalibratedChurnModel

    rng = np.random.default_rng(6)
    n = 40
    X = pd.DataFrame({col: rng.uniform(0, 10, n) for col in FEATURE_COLUMNS})
    y = rng.integers(0, 2, n)
    pipeline = LogisticRegressionCandidate().build_pipeline()
    pipeline.fit(X, y)
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(pipeline.predict_proba(X)[:, 1], y)
    model = CalibratedChurnModel(pipeline=pipeline, calibrator=calibrator)

    original_module = CalibratedChurnModel.__module__
    CalibratedChurnModel.__module__ = "__main__"
    model_path = tmp_path / "model_pickled_under_main.joblib"
    try:
        joblib.dump(model, model_path)
    finally:
        CalibratedChurnModel.__module__ = original_module

    output_path = tmp_path / "predicciones.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "--input-path",
            str(churn_ciclos_path),
            "--model-path",
            str(model_path),
            "--output-path",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    out = pd.read_csv(output_path)
    assert out["probabilidad_churn"].between(0, 1).all()
