import numpy as np
import pandas as pd
import pytest

from churn_detection.modeling.train import (
    CALIBRATION_METHODS,
    CV_SCORING,
    FEATURE_COLUMNS,
    CalibratedChurnModel,
    ChronologicalSplitter,
    HistGradientBoostingCandidate,
    HyperparameterTuner,
    KerasBinaryClassifier,
    ModelEvaluator,
    NeuralNetworkCandidate,
    LogisticRegressionCandidate,
    SigmoidCalibrator,
    ThresholdSelector,
    cross_validate_candidate,
    plot_reliability_diagram,
    reliability_table,
    select_calibrator,
    select_threshold,
)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(7)
    n_clients = 60
    base_date = pd.Timestamp("2025-10-08")
    rows = []
    cycle_id = 1
    for persona_id in range(1, n_clients + 1):
        n_cycles = rng.integers(1, 4)
        for _ in range(n_cycles):
            label = rng.integers(0, 2)
            rows.append(
                {
                    "persona_id": persona_id,
                    "inscripcion_id": cycle_id,
                    "fecha_vencimiento": base_date + pd.Timedelta(days=cycle_id),
                    "churn_label": float(label),
                    "tenure_dias": rng.integers(10, 300),
                    "porcentaje_uso_membresia": rng.uniform(0, 1),
                    "recencia_dias": rng.integers(0, 200),
                    "frecuencia_visitas_semanal": rng.uniform(0, 7),
                    "hora_promedio_checkin": rng.uniform(6, 22),
                    "pct_visitas_fin_de_semana": rng.uniform(0, 0.3),
                    "monto_total_gastado": rng.uniform(0, 2000),
                    "n_inscripciones_total": rng.integers(1, 10),
                    "ratio_actividad_reciente": rng.uniform(0, 3),
                    "cv_gap_visitas": rng.uniform(0, 2),
                    "monto_gastado_ultimos_90d": rng.uniform(0, 800),
                }
            )
            cycle_id += 1
    # A few unlabeled (censored) rows, which must never end up in a split.
    for persona_id in range(1, 6):
        rows.append(
            {
                "persona_id": persona_id,
                "inscripcion_id": cycle_id,
                "fecha_vencimiento": base_date + pd.Timedelta(days=cycle_id),
                "churn_label": np.nan,
                "tenure_dias": 50,
                "porcentaje_uso_membresia": 0.5,
                "recencia_dias": 5,
                "frecuencia_visitas_semanal": 2.0,
                "hora_promedio_checkin": 15,
                "pct_visitas_fin_de_semana": 0.1,
                "monto_total_gastado": 200,
                "n_inscripciones_total": 2,
                "ratio_actividad_reciente": 1.0,
                "cv_gap_visitas": 1.0,
                "monto_gastado_ultimos_90d": 100,
            }
        )
        cycle_id += 1
    return pd.DataFrame(rows)


def test_split_excludes_unlabeled_rows(sample_df):
    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    assert not split.y_train.isna().any()
    assert not split.y_test.isna().any()
    assert len(split.X_train) + len(split.X_test) == sample_df["churn_label"].notna().sum()


def test_split_is_chronological_not_just_grouped(sample_df):
    # The whole point of ChronologicalSplitter: every train row resolved
    # (fecha_vencimiento) at or before the cutoff, every test row strictly
    # after it -- train must never contain a cycle later than a test cycle.
    split = ChronologicalSplitter().split(sample_df, [*FEATURE_COLUMNS, "fecha_vencimiento"])
    train_dates = pd.to_datetime(split.X_train["fecha_vencimiento"])
    test_dates = pd.to_datetime(split.X_test["fecha_vencimiento"])
    assert train_dates.max() <= split.cutoff_date
    assert test_dates.min() > split.cutoff_date
    assert train_dates.max() < test_dates.min()


def test_split_allows_the_same_persona_on_both_sides(sample_df):
    # Unlike the old group-only split, a client's early cycle in train and a
    # later cycle in test is expected here -- persona_id is never a feature,
    # so this isn't leakage, it's the real sequence of events for a
    # returning client.
    split = ChronologicalSplitter().split(sample_df, [*FEATURE_COLUMNS, "persona_id"])
    train_ids = set(split.X_train["persona_id"])
    test_ids = set(split.X_test["persona_id"])
    assert train_ids & test_ids


@pytest.mark.parametrize(
    "candidate_factory",
    [LogisticRegressionCandidate, HistGradientBoostingCandidate, NeuralNetworkCandidate],
)
def test_every_candidate_fits_and_predicts_probabilities(sample_df, candidate_factory):
    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    candidate = candidate_factory()
    pipeline = candidate.build_pipeline()

    pipeline.fit(split.X_train, split.y_train)
    proba = pipeline.predict_proba(split.X_test)[:, 1]

    assert len(proba) == len(split.X_test)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_hist_gradient_boosting_handles_nan_features_without_imputation(sample_df):
    df = sample_df.copy()
    df.loc[df.index[0], "recencia_dias"] = np.nan  # simulate real missingness
    split = ChronologicalSplitter().split(df, FEATURE_COLUMNS)

    candidate = HistGradientBoostingCandidate()
    pipeline = candidate.build_pipeline()
    pipeline.fit(split.X_train, split.y_train)  # must not raise on NaN
    pipeline.predict_proba(split.X_test)


def test_neural_network_log1p_transform_only_touches_skewed_columns():
    from churn_detection.modeling.train import SKEWED_FEATURES

    candidate = NeuralNetworkCandidate()
    pipeline = candidate.build_pipeline()
    log1p_step = pipeline.named_steps["log1p_skewed"]

    X = pd.DataFrame([[10.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS).to_numpy()
    transformed = log1p_step.transform(X)

    for i, col in enumerate(FEATURE_COLUMNS):
        expected = np.log1p(10.0) if col in SKEWED_FEATURES else 10.0
        assert transformed[0, i] == pytest.approx(expected)


def test_neural_network_pipeline_is_picklable(sample_df, tmp_path):
    # KerasBinaryClassifier wraps a raw Keras model, which is NOT picklable by
    # default -- this is exactly the class of bug that broke the logistic
    # regression candidate earlier (a local closure that time). Custom
    # __getstate__/__setstate__ on KerasBinaryClassifier is what makes this pass.
    import joblib

    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    candidate = NeuralNetworkCandidate()
    pipeline = candidate.build_pipeline()
    pipeline.fit(split.X_train, split.y_train)

    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)  # must not raise
    reloaded = joblib.load(path)
    proba = reloaded.predict_proba(split.X_test)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_evaluator_returns_expected_metric_keys_within_bounds(sample_df):
    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    pipeline = LogisticRegressionCandidate().build_pipeline()
    pipeline.fit(split.X_train, split.y_train)
    y_pred = pipeline.predict(split.X_test)
    y_proba = pipeline.predict_proba(split.X_test)[:, 1]

    metrics = ModelEvaluator.evaluate(split.y_test, y_pred, y_proba)

    expected_keys = {
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
    }
    assert set(metrics.keys()) == expected_keys
    # Discrimination metrics and brier_score are bounded in [0, 1]; log_loss
    # is a proper scoring rule bounded below by 0 but not above.
    for key, value in metrics.items():
        if key == "log_loss":
            assert value >= 0.0
        else:
            assert 0.0 <= value <= 1.0


def test_plot_confusion_matrix_writes_a_file(sample_df, tmp_path):
    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    pipeline = LogisticRegressionCandidate().build_pipeline()
    pipeline.fit(split.X_train, split.y_train)

    from churn_detection.modeling.train import plot_confusion_matrix

    output_path = tmp_path / "cm.png"
    result = plot_confusion_matrix(
        pipeline, split.X_test, split.y_test, "logistic_regression", output_path
    )

    assert result == output_path
    assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_feature_importance_writes_a_file_for_linear_coefficients(sample_df, tmp_path):
    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    pipeline = LogisticRegressionCandidate().build_pipeline()
    pipeline.fit(split.X_train, split.y_train)

    from churn_detection.modeling.train import plot_feature_importance

    output_path = tmp_path / "fi.png"
    result = plot_feature_importance(
        pipeline, split.X_test, split.y_test, "logistic_regression", output_path
    )

    assert result == output_path
    assert output_path.exists() and output_path.stat().st_size > 0


def test_plot_feature_importance_falls_back_to_permutation_importance(sample_df, tmp_path):
    # Neither HistGradientBoostingClassifier nor the neural network expose
    # feature_importances_ or coef_ -- this is exactly the case that broke
    # the DVC pipeline before the permutation-importance fallback was added.
    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    pipeline = HistGradientBoostingCandidate().build_pipeline()
    pipeline.fit(split.X_train, split.y_train)

    from churn_detection.modeling.train import plot_feature_importance

    output_path = tmp_path / "fi_permutation.png"
    result = plot_feature_importance(
        pipeline, split.X_test, split.y_test, "hist_gradient_boosting", output_path
    )

    assert result == output_path
    assert output_path.exists() and output_path.stat().st_size > 0


def test_split_exposes_groups_train_matching_training_personas(sample_df):
    split = ChronologicalSplitter().split(sample_df, [*FEATURE_COLUMNS, "persona_id"])
    assert len(split.groups_train) == len(split.X_train)
    assert set(split.groups_train) == set(split.X_train["persona_id"])


@pytest.mark.parametrize(
    "candidate_factory",
    [LogisticRegressionCandidate, HistGradientBoostingCandidate, NeuralNetworkCandidate],
)
def test_every_candidate_declares_a_non_empty_prefixed_param_grid(candidate_factory):
    grid = candidate_factory().param_grid()
    assert len(grid) > 0
    for key in grid:
        assert key.startswith("model__")


def test_tuner_with_empty_grid_just_fits_the_pipeline(sample_df):
    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    pipeline = HistGradientBoostingCandidate().build_pipeline()

    tuner = HyperparameterTuner(n_splits=3)
    fitted, best_params, cv_metrics = tuner.tune(
        pipeline, {}, split.X_train, split.y_train, split.groups_train
    )

    assert best_params == {}
    assert cv_metrics == {}
    proba = fitted.predict_proba(split.X_test)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_tuner_picks_best_params_from_the_provided_grid(sample_df):
    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    candidate = LogisticRegressionCandidate()

    tuner = HyperparameterTuner(n_splits=3)
    fitted, best_params, cv_metrics = tuner.tune(
        candidate.build_pipeline(),
        candidate.param_grid(),
        split.X_train,
        split.y_train,
        split.groups_train,
    )

    assert best_params["model__C"] in candidate.param_grid()["model__C"]
    assert set(cv_metrics) == {f"cv_{stat}_{m}" for m in CV_SCORING for stat in ("mean", "std")}
    proba = fitted.predict_proba(split.X_test)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_keras_binary_classifier_is_recognized_as_a_classifier_by_sklearn():
    # Regression test: mixin/base-class declaration order matters for sklearn's
    # cooperative __sklearn_tags__() resolution. `class Foo(BaseEstimator,
    # ClassifierMixin)` silently fails to pick up ClassifierMixin's tags (MRO
    # resolves BaseEstimator's __sklearn_tags__ first, which doesn't forward to
    # the mixin), so `is_classifier()` returns False and GridSearchCV's scorer
    # rejects the whole pipeline as "a regressor" with response_method=
    # predict_proba. The fix is declaration order: ClassifierMixin first.
    from sklearn.base import is_classifier

    assert is_classifier(KerasBinaryClassifier())


def test_tuner_works_end_to_end_for_the_neural_network_candidate(sample_df):
    # Integration-level companion to the test above: GridSearchCV with
    # scoring="log_loss" must not raise for the neural network candidate.
    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    candidate = NeuralNetworkCandidate()

    tuner = HyperparameterTuner(n_splits=3)
    fitted, best_params, cv_metrics = tuner.tune(
        candidate.build_pipeline(),
        candidate.param_grid(),
        split.X_train,
        split.y_train,
        split.groups_train,
    )

    assert best_params  # a real (non-empty) choice was made, not skipped
    assert cv_metrics
    proba = fitted.predict_proba(split.X_test)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()


def test_cross_validate_candidate_returns_mean_and_std_for_every_metric(sample_df):
    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    candidate = LogisticRegressionCandidate()

    summary = cross_validate_candidate(
        candidate, split.X_train, split.y_train, split.groups_train, n_splits=3
    )

    assert set(summary) == {f"cv_{stat}_{m}" for m in CV_SCORING for stat in ("mean", "std")}


def test_calibrated_model_returns_valid_probabilities_and_is_picklable(sample_df, tmp_path):
    import joblib
    from sklearn.isotonic import IsotonicRegression
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    pipeline = LogisticRegressionCandidate().build_pipeline()
    pipeline.fit(split.X_train, split.y_train)

    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(
        LogisticRegressionCandidate().build_pipeline(),
        split.X_train,
        split.y_train,
        groups=split.groups_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip").fit(oof_proba, split.y_train)
    calibrated = CalibratedChurnModel(pipeline=pipeline, calibrator=calibrator)

    proba = calibrated.predict_proba(split.X_test)[:, 1]
    assert (proba >= 0).all() and (proba <= 1).all()
    preds = calibrated.predict(split.X_test)
    assert set(np.unique(preds)) <= {0, 1}

    path = tmp_path / "calibrated.joblib"
    joblib.dump(calibrated, path)  # must not raise
    reloaded = joblib.load(path)
    reloaded_proba = reloaded.predict_proba(split.X_test)[:, 1]
    np.testing.assert_allclose(reloaded_proba, proba)


def test_calibrated_model_uses_its_own_threshold_not_a_hardcoded_half():
    class _ConstantPipeline:
        def predict_proba(self, X):
            proba = np.full(len(X), 0.4)
            return np.column_stack([1 - proba, proba])

    calibrator = SigmoidCalibrator()
    calibrator.predict = lambda raw_proba: np.asarray(raw_proba)

    model_low_threshold = CalibratedChurnModel(
        pipeline=_ConstantPipeline(), calibrator=calibrator, threshold=0.3
    )
    model_high_threshold = CalibratedChurnModel(
        pipeline=_ConstantPipeline(), calibrator=calibrator, threshold=0.5
    )
    X = pd.DataFrame({"a": [1, 2, 3]})

    assert model_low_threshold.predict(X).tolist() == [1, 1, 1]
    assert model_high_threshold.predict(X).tolist() == [0, 0, 0]


def test_threshold_selector_picks_the_f1_maximizing_cutoff():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])

    threshold = ThresholdSelector().select(y_true, y_proba)

    from sklearn.metrics import f1_score

    candidate_thresholds = np.unique(y_proba)
    best_f1 = max(
        f1_score(y_true, (y_proba >= t).astype(int)) for t in candidate_thresholds
    )
    chosen_f1 = f1_score(y_true, (y_proba >= threshold).astype(int))
    assert chosen_f1 == pytest.approx(best_f1)


def test_select_threshold_averages_over_repeated_group_aware_folds(sample_df):
    # select_threshold must not just delegate to a single ThresholdSelector
    # sweep over the whole sample: that was the earlier, less stable version
    # of this function. It should average several partial estimates instead,
    # so the result is close to, but not necessarily identical to, the
    # single-sweep answer.
    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    pipeline = LogisticRegressionCandidate().build_pipeline()
    pipeline.fit(split.X_train, split.y_train)
    oof_proba = pipeline.predict_proba(split.X_train)[:, 1]

    threshold = select_threshold(
        split.y_train, oof_proba, split.groups_train, n_splits=3, n_repeats=2
    )

    assert 0.0 <= threshold <= 1.0
    assert not np.isnan(threshold)


def test_reliability_table_has_matching_raw_and_calibrated_columns(sample_df):
    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    y_proba_raw = np.linspace(0.0, 1.0, len(split.y_test))
    y_proba_calibrated = y_proba_raw.copy()

    table = reliability_table(split.y_test, y_proba_raw, y_proba_calibrated, n_bins=5)

    assert list(table.columns) == [
        "decil",
        "prob_media_predicha_sin_calibrar",
        "tasa_churn_observada_sin_calibrar",
        "prob_media_predicha_calibrada",
        "tasa_churn_observada_calibrada",
    ]
    assert len(table) > 0


def test_plot_reliability_diagram_writes_a_file(sample_df, tmp_path):
    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    y_proba_raw = np.linspace(0.0, 1.0, len(split.y_test))
    y_proba_calibrated = y_proba_raw.copy()

    output_path = tmp_path / "calibracion.png"
    result = plot_reliability_diagram(
        split.y_test, y_proba_raw, y_proba_calibrated, "logistic_regression", output_path
    )

    assert result == output_path
    assert output_path.exists() and output_path.stat().st_size > 0


def test_sigmoid_calibrator_returns_valid_probabilities():
    rng = np.random.default_rng(11)
    raw_proba = rng.uniform(0, 1, 200)
    y = rng.integers(0, 2, 200)

    calibrator = SigmoidCalibrator().fit(raw_proba, y)
    calibrated = calibrator.predict(raw_proba)

    assert (calibrated >= 0).all() and (calibrated <= 1).all()


def test_select_calibrator_picks_the_lower_cv_log_loss_method(sample_df):
    split = ChronologicalSplitter().split(sample_df, FEATURE_COLUMNS)
    pipeline = LogisticRegressionCandidate().build_pipeline()
    pipeline.fit(split.X_train, split.y_train)
    oof_proba = pipeline.predict_proba(split.X_train)[:, 1]

    method, calibrator, cv_log_loss = select_calibrator(
        oof_proba, split.y_train, split.groups_train, n_splits=3, n_repeats=2
    )

    assert set(cv_log_loss) == set(CALIBRATION_METHODS)
    assert method == min(cv_log_loss, key=cv_log_loss.get)
    calibrated = np.clip(calibrator.predict(oof_proba), 0.0, 1.0)
    assert (calibrated >= 0).all() and (calibrated <= 1).all()
