"""Train and compare churn-prediction models on churn_ciclos.csv.

Design notes (SOLID)
---------------------
- `ChronologicalSplitter` has one job: split into train/test BY TIME, not by
  a random (even if group-aware) shuffle. The goal is to demonstrate FUTURE
  performance, so train may only contain cycles that resolved (`fecha_
  vencimiento`) at or before a cutoff date, and test only cycles that
  resolved strictly after it -- exactly mirroring deployment, where the
  model only ever has past cycles to learn from and is scored on cycles
  that finish later. A client can still appear on both sides (their early
  cycle in train, a later one in test): that is not leakage, it is the real
  sequence of events for a returning client, and `persona_id` is never a
  feature.
- `WalkForwardGroupSplitter` extends that same time discipline one level
  down, INSIDE the training split. Model selection, hyperparameter tuning,
  probability calibration and threshold selection all cross-validate with
  this splitter instead of a random (even if group-aware and stratified)
  K-fold: fold i trains on everything up to calendar month M_i and
  validates on month M_{i+1}, purging any client seen in the validation
  month from that fold's training rows. A random K-fold would happily
  validate an early month with a model partly trained on later months --
  exactly the kind of leakage `ChronologicalSplitter` already prevents for
  the outer test split, left unaddressed one level down. It matters
  concretely because `RELIABLE_ACCESS_TRACKING_SINCE` (churn_detection/
  config.py) is 2026-02-01: every check-in-derived feature is NaN before
  that date from a real system outage, not from an actual absence of
  visits, so a fold mixing pre/post-February cycles between train and
  validation is being scored under a condition that could never happen in
  production, where the model only ever looks backward from today.
  `min_train_size` additionally skips any fold whose training window is
  still too small to give a stable estimate -- this project's very first
  windows (a single, mostly check-in-blind month) are exactly that, and a
  handful of near-empty folds dragging the average around says more about
  fold size than about the candidate being evaluated.
- `ChurnModelCandidate` (ABC): each concrete candidate only knows how to build
  its own preprocessing + estimator `Pipeline`. Adding a new candidate model
  is one small class, not a change to the training loop (Open/Closed).
- `ModelEvaluator`: single responsibility, turns predictions into a metrics
  dict. Used identically for every candidate so comparisons are apples-to-apples.
  It includes both discrimination metrics (ROC-AUC, PR-AUC, recall, ...) and
  calibration metrics (Brier score, log loss): discrimination says whether the
  model ranks churners above non-churners, calibration says whether a
  predicted probability of 0.80 actually corresponds to an observed churn
  rate near 80% -- the two are independent properties, and only the second
  one is evidence for the stated goal of estimating a churn PROBABILITY.
- Model selection (which of the 3 candidates wins, and which hyperparameters
  win within that candidate) is decided ONLY from K-fold cross-validation
  scores computed on the training split. The held-out test split is fit and
  scored for every candidate too, but purely for descriptive reporting
  (ROC curves, the comparison table) -- never to pick a winner, since doing
  that would contaminate the one dataset left for an honest final estimate.
- `ThresholdSelector` has one job too: pick the probability cutoff that
  maximizes F1 on out-of-fold predictions, instead of leaving every
  candidate stuck at a hardcoded 0.5. It only runs once, on the calibrated
  probabilities of the already selected and tuned model, so it changes how
  a probability becomes a hard label without touching model selection.
- MLflow: every candidate's params, metrics, and fitted pipeline are logged
  under one experiment, so a training run is reproducible/comparable later,
  not just "whichever model happened to win this time".
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

FEATURE_COLUMNS = [
    "tenure_dias",
    "porcentaje_uso_membresia",
    "recencia_dias",
    "frecuencia_visitas_semanal",
    "hora_promedio_checkin",
    "pct_visitas_fin_de_semana",
    "monto_total_gastado",
    "n_inscripciones_total",
    "ratio_actividad_reciente",
    "cv_gap_visitas",
    "monto_gastado_ultimos_90d",
]

# Right-skewed features flagged during EDA (reports/figures/10_*.png). Only
# gradient-descent-trained models need this: logistic regression and the
# neural network have scale-sensitive inputs (unscaled/skewed inputs make
# gradient descent converge poorly), and their StandardScaler step computes
# mean/std, which long right tails distort. Hist Gradient Boosting splits on
# thresholds, so it's invariant to any monotonic transform of a feature --
# applying log1p to it would be a no-op on its predictions, not a fix.
SKEWED_FEATURES = [
    "tenure_dias",
    "recencia_dias",
    "monto_total_gastado",
    "n_inscripciones_total",
    "ratio_actividad_reciente",
    "cv_gap_visitas",
    "monto_gastado_ultimos_90d",
]


@dataclass
class ChurnSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    dates_train: pd.Series
    cutoff_date: pd.Timestamp


class ChronologicalSplitter:
    """Train/test split BY TIME: train only contains cycles whose outcome was
    decided at or before a cutoff date, test only contains cycles decided
    strictly after it.

    `fecha_vencimiento` is the point-in-time cutoff every feature in
    churn_dataset.py is already computed as of (see that module's "No
    leakage" note), so it is also the right column to order cycles
    chronologically for this split: a row's `fecha_vencimiento` is when its
    outcome (renewed/churned) becomes knowable.

    A prior version of this splitter (`GroupAwareSplitter`) grouped by
    `persona_id` so a client's cycles never straddled train/test, but ignored
    time entirely -- train could (and did) contain cycles that resolved
    AFTER some test cycles, which overstates how the model would perform on
    genuinely future data. This splitter fixes that directly: every row in
    `X_train` has `fecha_vencimiento <= cutoff_date`, every row in `X_test`
    has `fecha_vencimiento > cutoff_date`, by construction.

    `persona_id` is still exposed as `groups_train`, not to constrain this
    outer split (a client's early cycle can legitimately land in train while
    a later one lands in test -- that is not leakage, `persona_id` is never a
    feature) but so the K-fold cross-validation used downstream for model
    selection and hyperparameter tuning can still avoid splitting one
    client's cycles across two folds.
    """

    def __init__(self, test_size: float = 0.25, date_column: str = "fecha_vencimiento") -> None:
        self._test_size = test_size
        self._date_column = date_column

    def split(self, df: pd.DataFrame, feature_columns: list[str]) -> ChurnSplit:
        labeled = df.dropna(subset=["churn_label"]).copy()
        labeled[self._date_column] = pd.to_datetime(labeled[self._date_column])
        labeled = labeled.sort_values(self._date_column).reset_index(drop=True)

        cutoff_idx = int(round(len(labeled) * (1 - self._test_size)))
        cutoff_idx = min(max(cutoff_idx, 1), len(labeled) - 1)
        cutoff_date = labeled.loc[cutoff_idx - 1, self._date_column]

        train = labeled[labeled[self._date_column] <= cutoff_date]
        test = labeled[labeled[self._date_column] > cutoff_date]

        return ChurnSplit(
            X_train=train[feature_columns],
            X_test=test[feature_columns],
            y_train=train["churn_label"].astype(int),
            y_test=test["churn_label"].astype(int),
            groups_train=train["persona_id"],
            dates_train=train[self._date_column],
            cutoff_date=cutoff_date,
        )


class WalkForwardGroupSplitter:
    """Time-respecting cross-validator for everything that happens INSIDE the
    training split: model selection, hyperparameter tuning, calibration and
    threshold selection. See the module docstring for why this replaces a
    random (even if group-aware, stratified) K-fold there.

    Fold i trains on every row whose date falls at or before calendar month
    M_i, and validates on the immediately following month, M_{i+1} -- an
    expanding window, never a rolling one, so later folds simply have more
    training history, exactly like a real retraining schedule would.

    Any client who appears in a fold's validation month is purged entirely
    from that fold's training rows (not just their cycle in that month --
    every row of theirs up to that point), for the same reason this project
    already groups by client everywhere else: a client's own earlier cycle
    could otherwise leak information about their later, held-out one within
    the same fold.

    Validation windows with fewer than `min_val_size` rows are skipped --
    too few observations for the fold's metric to mean much -- but their
    rows still accumulate into the training set of whichever later fold
    does qualify. Folds whose TRAINING window (after purging) has fewer than
    `min_train_size` rows are skipped too, and this time the excluded rows
    are simply never scored at all: a fold trained on a few hundred rows
    from a single early month, where several features are entirely missing
    (see the module docstring's note on RELIABLE_ACCESS_TRACKING_SINCE),
    produces an unstable per-fold estimate for any candidate regardless of
    how good it actually is, and it is more honest to exclude that fold
    than to let it swing the average around. The very first month is never
    a validation window either way: there is no earlier data a model could
    have trained on to be validated against it, exactly mirroring that a
    real model deployed at that point would have had nothing to learn from
    yet.
    """

    def __init__(
        self, dates: pd.Series, min_val_size: int = 100, min_train_size: int = 100
    ) -> None:
        self._dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
        self._min_val_size = min_val_size
        self._min_train_size = min_train_size

    def months(self) -> list[pd.Period]:
        """Calendar months present in `dates`, sorted -- the first is never a
        validation window (see class docstring), the rest are candidates for
        one, subject to `min_val_size`/`min_train_size`."""
        return sorted(self._dates.dt.to_period("M").unique())

    def split(self, X=None, y=None, groups=None):
        months = self._dates.dt.to_period("M")
        groups_arr = pd.Series(groups).reset_index(drop=True) if groups is not None else None
        windows = self.months()
        for i in range(len(windows) - 1):
            train_cutoff_month, val_month = windows[i], windows[i + 1]
            val_mask = (months == val_month).to_numpy()
            if val_mask.sum() < self._min_val_size:
                continue
            train_mask = (months <= train_cutoff_month).to_numpy()
            if groups_arr is not None:
                val_clients = set(groups_arr[val_mask])
                train_mask = train_mask & ~groups_arr.isin(val_clients).to_numpy()
            if train_mask.sum() < self._min_train_size:
                continue
            yield np.flatnonzero(train_mask), np.flatnonzero(val_mask)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return sum(1 for _ in self.split(X, y, groups))


def walk_forward_oof_predict_proba(
    estimator, X: pd.DataFrame, y: pd.Series, groups: pd.Series, cv: WalkForwardGroupSplitter
) -> tuple[np.ndarray, np.ndarray]:
    """Like `sklearn.model_selection.cross_val_predict(method="predict_proba")`,
    but tolerant of a CV that does not cover every row -- which
    `WalkForwardGroupSplitter` never does by construction, since its first
    month (and any fold below `min_train_size`/`min_val_size`) is never a
    validation window (see its docstring). sklearn's own `cross_val_predict`
    requires full coverage and raises otherwise.

    Returns `(oof_proba, covered)`: `oof_proba` has one entry per row of `X`,
    `nan` for rows that never landed in a validation fold; `covered` flags
    which rows are real out-of-fold predictions. Callers must restrict to
    `covered` rows before using `oof_proba` for anything -- an uncovered row
    was never actually validated against a model that didn't train on it.
    """
    n = len(X)
    oof_proba = np.full(n, np.nan)
    covered = np.zeros(n, dtype=bool)
    for train_idx, val_idx in cv.split(X, y, groups=groups):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        covered[val_idx] = True
    return oof_proba, covered


class ChurnModelCandidate(ABC):
    """Contract every candidate model implements: build its own pipeline and
    declare its own hyperparameter search space."""

    name: str
    # GridSearchCV parallelism for this candidate's own hyperparameter search.
    # -1 (all cores) by default; a candidate overrides this only if it has a
    # concrete reason not to parallelize (see NeuralNetworkCandidate).
    n_jobs: int = -1

    @abstractmethod
    def build_pipeline(self) -> Pipeline: ...

    def param_grid(self) -> dict | list[dict]:
        """Grid of pipeline-step-prefixed hyperparameters to search over (e.g.
        `{"model__C": [0.1, 1, 10]}`). Empty by default -- a candidate with
        nothing worth tuning just skips the search (Open/Closed: adding a
        candidate never requires touching the tuning code)."""
        return {}


# Mismas 7 métricas que ModelEvaluator.evaluate produce sobre el test set,
# expresadas como scorers de sklearn para que GridSearchCV/cross_validate las
# calculen en cada fold del CV. Los nombres de la izquierda (las keys) son
# intencionalmente los mismos que las keys de ModelEvaluator.evaluate, para
# poder comparar "cv_mean_X" / "cv_std_X" contra "X" (test) fila por fila sin
# renombrar nada. brier_score/log_loss son métricas de CALIBRACIÓN, indican
# qué tan bien la probabilidad predicha refleja la tasa real observada, y no
# de discriminación. Se agregan porque el objetivo del proyecto es estimar
# una probabilidad de abandono, y ROC-AUC, PR-AUC y recall no dicen nada
# sobre eso, ya que son invariantes a cualquier recalibración monótona de la
# probabilidad. Accuracy se quitó del set de métricas: la tasa de abandono
# cambia entre train y test, según lo descrito en ChronologicalSplitter, así
# que Accuracy puede verse bien o mal solo por ese cambio de proporción, sin
# que eso diga nada sobre si el modelo mejoró o empeoró. Precision, Recall y
# F1 ya describen el comportamiento del modelo sin ese problema.
CV_SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "brier_score": "neg_brier_score",
    "log_loss": "neg_log_loss",
}

# brier_score/log_loss are losses (lower is better) but their sklearn scorers
# ("neg_brier_score"/"neg_log_loss") report them negated, so that internally
# "greater is better" holds uniformly across every metric in CV_SCORING --
# that is what lets GridSearchCV's `refit=<any CV_SCORING key>` and a plain
# `max()` across candidates work without special-casing which metrics are
# losses. `_humanize_cv_summary` flips the sign back only when BUILDING A
# REPORT (CSV/mlflow), so it reads on the same lower-is-better scale as
# ModelEvaluator.evaluate.
_LOSS_METRICS = {"brier_score", "log_loss"}


def _humanize_cv_summary(cv_summary: dict[str, float]) -> dict[str, float]:
    out = dict(cv_summary)
    for metric_name in _LOSS_METRICS:
        key = f"cv_mean_{metric_name}"
        if key in out:
            out[key] = -out[key]
    return out


class HyperparameterTuner:
    """Grid search cross-validated with `cv` (see `WalkForwardGroupSplitter`).

    Same concern as `WalkForwardGroupSplitter`'s client purging, one level
    down: if a client's cycles could land in both the training and
    validation portion of a fold, a hyperparameter choice could look good
    only because the model partly memorized that specific client, not
    because it generalizes.

    The winning hyperparameter combination is chosen entirely from these CV
    (validation) scores -- `X`/`y`/`groups` here must be the TRAINING split
    only, never the held-out test set.
    """

    def __init__(self, cv: WalkForwardGroupSplitter, scoring: str = "log_loss") -> None:
        self._cv = cv
        self._scoring = scoring

    def tune(
        self, pipeline: Pipeline, param_grid: dict, X, y, groups, n_jobs: int = -1
    ) -> tuple[Pipeline, dict, dict]:
        """Returns (best_estimator, best_params, cv_metrics).

        `cv_metrics` holds the mean AND standard deviation, across the CV
        folds, of *every* metric in `CV_SCORING` -- not just the one
        (`self._scoring`) used to rank/refit -- for the winning
        hyperparameter combination only
        (`GridSearchCV.cv_results_["mean_test_<metric>"]` /
        `["std_test_<metric>"]` at `best_index_`). This is what tells you
        whether the best score was a stable improvement across folds or a
        lucky split -- a high mean with a high std is a red flag that raw
        ranking alone hides -- and it lets you sanity-check the winner on
        every metric, not just the one it was optimized for. Returned as-is
        in sklearn's "greater is better" convention (see `_LOSS_METRICS`);
        callers that report these numbers should pass them through
        `_humanize_cv_summary` first.
        """
        if not param_grid:
            pipeline.fit(X, y)
            return pipeline, {}, {}

        search = GridSearchCV(
            pipeline,
            param_grid,
            scoring=CV_SCORING,
            refit=self._scoring,
            cv=self._cv,
            n_jobs=n_jobs,
        )
        search.fit(X, y, groups=groups)
        best_idx = search.best_index_
        cv_metrics = {}
        for metric_name in CV_SCORING:
            cv_metrics[f"cv_mean_{metric_name}"] = float(
                search.cv_results_[f"mean_test_{metric_name}"][best_idx]
            )
            cv_metrics[f"cv_std_{metric_name}"] = float(
                search.cv_results_[f"std_test_{metric_name}"][best_idx]
            )
        return search.best_estimator_, search.best_params_, cv_metrics


class _Log1pSkewedColumns:
    """Picklable callable for FunctionTransformer: log1p only the skewed columns.

    Must be a top-level class, not a local closure inside build_pipeline() --
    a closure can't be pickled, which would silently break `joblib.dump()`
    (used to save `models/churn_model.joblib`) and MLflow's model logging.
    Caught by test_logistic_regression_pipeline_is_picklable.
    """

    def __init__(self, skewed_idx: list[int]) -> None:
        self._skewed_idx = skewed_idx

    def __call__(self, X):
        X = np.asarray(X, dtype=float).copy()
        X[:, self._skewed_idx] = np.log1p(np.clip(X[:, self._skewed_idx], 0, None))
        return X


class KerasBinaryClassifier(ClassifierMixin, BaseEstimator):
    """Minimal sklearn-compatible wrapper around a small Keras feed-forward
    network, so it can sit inside the same `Pipeline` API as every other
    candidate (`.fit`, `.predict`, `.predict_proba`).

    Written by hand instead of pulling in scikeras, to keep one fewer
    dependency and full control over serialization: a raw Keras model is NOT
    picklable by joblib. `__getstate__`/`__setstate__` below convert it to/from
    bytes using Keras's own save/load format -- the same class of bug as
    `_Log1pSkewedColumns` above, caught here by
    test_neural_network_pipeline_is_picklable before it could silently break
    `models/churn_model.joblib`.
    """

    def __init__(
        self,
        hidden_units: tuple[int, ...] = (32, 16),
        epochs: int = 40,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model = None
        self.classes_ = None

    def _build_model(self, n_features: int):
        import tensorflow as tf

        tf.random.set_seed(self.random_state)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(n_features,)))
        for units in self.hidden_units:
            model.add(tf.keras.layers.Dense(units, activation="relu"))
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["AUC"],
        )
        return model

    def fit(self, X, y):
        import tensorflow as tf

        # Seed every source of randomness involved (Python/NumPy for data
        # shuffling and weight init helpers, TF for the graph itself) so a
        # run with the same hyperparameters is actually reproducible.
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self.classes_ = np.array([0, 1])
        self._model = self._build_model(X.shape[1])

        # A fixed epoch count with no monitoring risks over/underfitting
        # depending on the learning rate picked by the grid search. Early
        # stopping on a held-out slice of the training data (val_loss, not a
        # named metric, so this doesn't depend on Keras's internal metric
        # naming) fixes that without needing to hand-tune epochs per config.
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", mode="min", patience=5, restore_best_weights=True
            )
        ]
        self._model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=0,
        )
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype="float32")
        p1 = self._model.predict(X, verbose=0).reshape(-1)
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        model = state.pop("_model", None)
        state["_model_bytes"] = None
        if model is not None:
            import os
            import tempfile

            fd, path = tempfile.mkstemp(suffix=".keras")
            os.close(fd)
            try:
                model.save(path)
                with open(path, "rb") as f:
                    state["_model_bytes"] = f.read()
            finally:
                os.remove(path)
        return state

    def __setstate__(self, state: dict) -> None:
        model_bytes = state.pop("_model_bytes", None)
        self.__dict__.update(state)
        self._model = None
        if model_bytes is not None:
            import os
            import tempfile

            import tensorflow as tf

            fd, path = tempfile.mkstemp(suffix=".keras")
            os.close(fd)
            try:
                with open(path, "wb") as f:
                    f.write(model_bytes)
                self._model = tf.keras.models.load_model(path)
            finally:
                os.remove(path)


class NeuralNetworkCandidate(ChurnModelCandidate):
    """Small feed-forward neural network (TensorFlow/Keras). Needs the same
    preprocessing as a linear model would: it's scale-sensitive (gradient
    descent on unscaled/skewed inputs converges poorly) and can't take NaN."""

    name = "neural_network"
    # GridSearchCV's default joblib backend forks/pickles the estimator to
    # worker processes; this candidate wraps a live TensorFlow model, so
    # multiprocessing + TF is a known source of subtle hangs/crashes. Not
    # worth the risk for a search space this small -- the other candidates
    # keep their full parallelism (n_jobs=-1 from the base class).
    n_jobs = 1

    def build_pipeline(self) -> Pipeline:
        skewed_idx = [FEATURE_COLUMNS.index(c) for c in SKEWED_FEATURES]
        return Pipeline(
            [
                # keep_empty_features=True: an early WalkForwardGroupSplitter
                # fold can still have a column with zero non-missing values
                # even after min_train_size filtering. SimpleImputer's default
                # silently DROPS an all-NaN column instead of imputing it,
                # which shifts every later column's position and breaks
                # _Log1pSkewedColumns, whose indices are fixed positions in
                # FEATURE_COLUMNS. Keeping the column (filled with 0, the
                # documented fallback when there is no median to compute)
                # keeps the column count and order stable across every fold.
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("log1p_skewed", FunctionTransformer(_Log1pSkewedColumns(skewed_idx))),
                ("scaler", StandardScaler()),
                ("model", KerasBinaryClassifier()),
            ]
        )

    def param_grid(self) -> dict:
        # Deliberately small: each combination refits the network from scratch
        # for every CV fold, and unlike the other two candidates this one has
        # no cheap way to skip redundant work. hidden_units controls capacity,
        # learning_rate controls how well gradient descent actually converges
        # within `epochs` -- the two knobs most likely to matter here.
        return {
            "model__hidden_units": [(32, 16), (16, 8)],
            "model__learning_rate": [1e-3, 1e-2],
        }


class LogisticRegressionCandidate(ChurnModelCandidate):
    """Interpretable linear baseline. Needs imputation, the same log1p transform
    on the right-skewed features used by the neural network (see SKEWED_FEATURES),
    and scaling -- in that order. Serves as the simple reference model that the
    more complex candidates (Gradient Boosting, neural network) need to beat to
    justify their extra complexity."""

    name = "logistic_regression"

    def build_pipeline(self) -> Pipeline:
        skewed_idx = [FEATURE_COLUMNS.index(c) for c in SKEWED_FEATURES]
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("log1p_skewed", FunctionTransformer(_Log1pSkewedColumns(skewed_idx))),
                ("scaler", StandardScaler()),
                # saga soporta l1/l2/elasticnet; liblinear (el default) solo
                # soporta l2 y l1, no elasticnet, y no escala tan bien como
                # saga cuando el grid crece.
                ("model", LogisticRegression(max_iter=2000, random_state=42, solver="saga")),
            ]
        )

    def param_grid(self) -> dict:
        # El tuning anterior eligio C=0.01, cerca del extremo mas regularizado
        # de la grilla vieja (el minimo era 0.001) -- senal de que el optimo
        # real podia estar mas alla del borde explorado. Se extiende el rango
        # hacia C aun mas chicos (mas regularizacion) y se agregan puntos
        # intermedios alrededor de la zona donde gano la corrida anterior,
        # en vez de solo mover el limite.
        c_values = [
            0.0001,
            0.0005,
            0.001,
            0.0025,
            0.005,
            0.0075,
            0.01,
            0.025,
            0.05,
            0.1,
            0.5,
            1.0,
            5.0,
            10.0,
            50.0,
        ]
        return [
            {
                "model__penalty": ["l2"],
                "model__C": c_values,
            },
            {
                "model__penalty": ["l1"],
                "model__C": c_values,
            },
            {
                "model__penalty": ["elasticnet"],
                "model__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                "model__l1_ratio": [0.15, 0.5, 0.85],
            },
        ]


class HistGradientBoostingCandidate(ChurnModelCandidate):
    """Handles NaN natively -- no imputation step at all.

    Our missingness is not random noise (see churn_dataset.py: it mostly means
    "no reliably-tracked check-in history"), so a model that can use the
    *pattern* of missingness as a signal, instead of having it papered over
    by a median, is a genuinely better fit here, not just a convenience.
    """

    name = "hist_gradient_boosting"

    def build_pipeline(self) -> Pipeline:
        return Pipeline([("model", HistGradientBoostingClassifier(random_state=42))])

    def param_grid(self) -> dict:
        return {
            "model__max_iter": [100, 200, 300],
            "model__max_depth": [None, 5, 10],
            "model__learning_rate": [0.05, 0.1, 0.2],
        }


class ModelEvaluator:
    """Turns (y_true, y_pred, y_proba) into a standard classification metrics dict.

    precision/recall/f1/roc_auc/pr_auc measure DISCRIMINATION. They answer
    whether predicted probabilities rank churners above non-churners. All
    five are invariant to any monotonic rescaling of y_proba, so a fixed
    shift or replacing every probability with its square root would leave
    them unchanged, and none of them can tell you whether a probability of
    0.80 corresponds to an observed churn rate anywhere near 80%.
    brier_score/log_loss measure CALIBRATION instead. Both are proper
    scoring rules that get strictly worse the further y_proba drifts from
    the true probability, which is what actually answers that question. See
    `plot_reliability_diagram` for the visual version of the same check.

    Accuracy is deliberately left out. The churn rate shifts between train
    and test under `ChronologicalSplitter`, so Accuracy can move just
    because that proportion changed, not because the model got better or
    worse. Precision, Recall and F1 already describe the model's behavior
    without that distortion.
    """

    @staticmethod
    def evaluate(y_true: pd.Series, y_pred, y_proba) -> dict[str, float]:
        return {
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "pr_auc": average_precision_score(y_true, y_proba),
            "brier_score": brier_score_loss(y_true, y_proba),
            "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        }


def cross_validate_candidate(
    candidate: ChurnModelCandidate,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    cv: WalkForwardGroupSplitter,
) -> dict[str, float]:
    """Runs `cv` (see `WalkForwardGroupSplitter`) for one candidate on the
    TRAINING split, and summarizes every `CV_SCORING` metric.

    This is what Phase 1 of `train_models` uses to decide which of the 3
    candidates wins -- never a single fit-once-score-on-test comparison,
    which would double-use the test set (once, implicitly, to pick a winner,
    and again to report that winner's "final" performance).

    Returned in sklearn's "greater is better" convention (see
    `_LOSS_METRICS`); pass through `_humanize_cv_summary` before reporting.
    """
    scores = cross_validate(
        candidate.build_pipeline(),
        X_train,
        y_train,
        groups=groups_train,
        cv=cv,
        scoring=CV_SCORING,
        n_jobs=candidate.n_jobs,
    )
    summary = {}
    for metric_name in CV_SCORING:
        raw = np.asarray(scores[f"test_{metric_name}"], dtype=float)
        summary[f"cv_mean_{metric_name}"] = float(raw.mean())
        summary[f"cv_std_{metric_name}"] = float(raw.std())
    return summary


class SigmoidCalibrator:
    """Platt-scaling calibrator: a 1-feature logistic regression fit on the
    model's raw probability. Only 2 parameters (slope + intercept in
    log-odds space), so it can only stretch/shift the probability curve as a
    whole -- far less flexible than `IsotonicRegression`, which can fit an
    arbitrary monotonic step function.

    That extra flexibility is exactly what makes isotonic regression prone to
    overfitting the tails of the probability distribution when few
    out-of-fold observations land there (see `select_calibrator`'s docstring
    for the concrete failure this was written to fix): a handful of points
    at the high-risk end can pull an isotonic step to a wild extreme, while a
    single fitted logistic curve cannot swing nearly as far from what the
    bulk of the data supports.
    """

    def __init__(self) -> None:
        self._model = LogisticRegression()

    def fit(self, raw_proba, y):
        self._model.fit(np.asarray(raw_proba).reshape(-1, 1), y)
        return self

    def predict(self, raw_proba):
        return self._model.predict_proba(np.asarray(raw_proba).reshape(-1, 1))[:, 1]


# Calibration methods considered by `select_calibrator`. Both expose the same
# minimal (fit(raw_proba, y) -> self, predict(raw_proba) -> calibrated proba)
# interface, so adding a third method later is a one-line addition here.
CALIBRATION_METHODS: dict[str, Callable[[], object]] = {
    "isotonic": lambda: IsotonicRegression(out_of_bounds="clip"),
    "sigmoid": lambda: SigmoidCalibrator(),
}


def _cv_calibration_log_loss(
    calibrator_factory,
    oof_proba,
    y_train,
    cv_folds: list[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Mean Log Loss of `calibrator_factory()`, cross-validated over
    `cv_folds` on the model's own out-of-fold probabilities: fit on one
    fold's split, score on the other. This cross-validates the CALIBRATION
    STEP itself, for the same reason the model and its hyperparameters are
    picked from CV scores earlier in this module -- scoring a calibrator on
    the same probabilities it was fit on would always favor the most
    flexible method (isotonic), because it can bend the calibration curve
    until it matches that exact sample, whether or not that curve
    generalizes.

    Uses Log Loss, not Brier Score, as the comparison metric -- an earlier
    version used Brier Score here and it was not sensitive enough: isotonic
    kept winning the CV comparison on this project's data (~0.1803 vs.
    ~0.1809 for sigmoid) even though it clearly overfit the riskiest decile
    on the held-out test set (predicted ~0.97 there when the observed rate
    was ~0.80 -- see `select_calibrator`). That decile is a small slice of
    the data, so a squared-error average like Brier Score barely moves; Log
    Loss's logarithmic penalty for a confident, wrong prediction is large
    enough to make that same slice tip the comparison the other way -- consistent
    with why Log Loss, not ROC-AUC, is already this project's model/
    hyperparameter selection metric (section 6.4).

    `cv_folds` is a materialized list of (train_idx, val_idx) pairs, built by
    the caller from `WalkForwardGroupSplitter` -- unlike a shuffled random
    K-fold, a walk-forward split has no randomness to repeat over: the same
    calendar months always produce the same folds.
    """
    oof_proba = np.asarray(oof_proba, dtype=float)
    y_arr = np.asarray(y_train)
    scores = []
    for train_idx, val_idx in cv_folds:
        calibrator = calibrator_factory().fit(oof_proba[train_idx], y_arr[train_idx])
        calibrated = np.clip(calibrator.predict(oof_proba[val_idx]), 1e-6, 1 - 1e-6)
        scores.append(log_loss(y_arr[val_idx], calibrated, labels=[0, 1]))
    return float(np.mean(scores))


def select_calibrator(
    oof_proba, y_train, cv_folds: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[str, object, dict[str, float]]:
    """Picks whichever method in `CALIBRATION_METHODS` gets the lowest
    cross-validated Log Loss (over `cv_folds`) on the model's own
    out-of-fold probabilities, then refits that method on all of them.

    Written after isotonic calibration, applied unconditionally, was found to
    make Brier Score/Log Loss on the held-out test set slightly WORSE for
    this project's winning model, not better -- the reliability table showed
    it fit a very confident, wrong step at the riskiest decile (very few
    out-of-fold observations land there, so isotonic's flexibility let it
    overfit that handful of points instead of generalizing). Comparing
    methods by their own cross-validated score, instead of always applying
    isotonic, catches exactly that failure instead of shipping it -- but only
    once that comparison uses a metric sensitive enough to see it (see
    `_cv_calibration_log_loss`'s docstring for why Brier Score was not
    sensitive enough here).

    Returns (method_name, fitted_calibrator, cv_log_loss_by_method).
    """
    cv_log_loss = {
        name: _cv_calibration_log_loss(factory, oof_proba, y_train, cv_folds)
        for name, factory in CALIBRATION_METHODS.items()
    }
    best_name = min(cv_log_loss, key=cv_log_loss.get)
    fitted = CALIBRATION_METHODS[best_name]().fit(np.asarray(oof_proba, dtype=float), np.asarray(y_train))
    return best_name, fitted, cv_log_loss


class ThresholdSelector:
    """Picks the probability threshold that maximizes F1 on labeled
    predictions, instead of always cutting at 0.5.

    F1 depends entirely on where a continuous probability gets cut into a
    hard label. 0.5 is a convenient default, not a value with any special
    claim to being the best cutoff for a given model. This class sweeps
    every threshold implied by the data, using `precision_recall_curve`
    instead of guessing candidate values by hand, and keeps the one with
    the highest F1.

    Single responsibility: given labels and probabilities, return one
    threshold. It does not decide how those probabilities were produced,
    nor whether one sweep is stable enough to trust on its own. Stability
    across different fold assignments is `select_threshold`'s job, the same
    split of responsibilities `select_calibrator` already uses for the
    calibration method itself.
    """

    def select(self, y_true, y_proba) -> float:
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        denominator = precision + recall
        f1_scores = np.divide(
            2 * precision * recall,
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        # precision_recall_curve appends one extra point, precision=1 and
        # recall=0 at an implicit threshold of infinity, that has no matching
        # entry in `thresholds`. Drop it so the two arrays line up.
        best_idx = int(np.argmax(f1_scores[:-1]))
        return float(thresholds[best_idx])


def _cv_best_thresholds(
    selector: ThresholdSelector,
    y_true,
    y_proba,
    cv_folds: list[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    """Collects one F1-optimal threshold per fold in `cv_folds`, using only
    that fold's training portion of `y_proba`.

    A single sweep over the full out-of-fold array already avoids the
    in-sample overconfidence problem, but it still commits to whichever
    threshold looks best under one particular arrangement of clients into
    folds. `select_calibrator` runs into the same issue comparing isotonic
    against sigmoid, and handles it the same way: many partial estimates,
    each computed on one fold's training portion of the out-of-fold data,
    average out the noise that any single 100% estimate would carry.
    """
    y_arr = np.asarray(y_true)
    proba_arr = np.asarray(y_proba, dtype=float)
    return [selector.select(y_arr[train_idx], proba_arr[train_idx]) for train_idx, _ in cv_folds]


def select_threshold(y_true, y_proba, cv_folds: list[tuple[np.ndarray, np.ndarray]]) -> float:
    """Returns a decision threshold averaged over one F1-optimal estimate per
    fold in `cv_folds`, instead of a single sweep over all of `y_proba`.

    Reuses the already computed out-of-fold probabilities, the same ones
    `select_calibrator` uses, rather than refitting the underlying model
    again, since the instability being addressed here comes from how the
    data gets split, not from how the model itself was trained.
    """
    thresholds = _cv_best_thresholds(ThresholdSelector(), y_true, y_proba, cv_folds)
    return float(np.mean(thresholds))


class CalibratedChurnModel(ClassifierMixin, BaseEstimator):
    """Wraps a fitted `pipeline` with a post-hoc calibrator (see
    `select_calibrator`) so that `predict_proba` returns probabilities that
    mean what they say. A 0.80 should correspond to an observed churn rate
    near 80%, not merely "more likely to churn than something scored 0.79".
    ROC-AUC and PR-AUC only guarantee the latter, see `ModelEvaluator`.

    `calibrator` must be fit on OUT-OF-FOLD probabilities from `pipeline`'s
    own training data, using `walk_forward_oof_predict_proba` with the same
    walk-forward CV used everywhere else in this module, never on
    `pipeline`'s in-sample predictions. Those are systematically overconfident, which would just
    teach the calibrator to reproduce the model's own uncalibrated output.
    This is the same reason hyperparameters are picked from CV scores
    rather than training-set scores. A model's opinion of itself, measured
    on data it already saw, is not evidence.

    `threshold` is the cutoff `predict` applies to the calibrated
    probability, chosen by `ThresholdSelector` instead of hardcoded at 0.5.
    """

    def __init__(self, pipeline, calibrator, threshold: float = 0.5) -> None:
        self.pipeline = pipeline
        self.calibrator = calibrator
        self.threshold = threshold
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        raw_proba = self.pipeline.predict_proba(X)[:, 1]
        calibrated = np.clip(self.calibrator.predict(raw_proba), 0.0, 1.0)
        return np.column_stack([1 - calibrated, calibrated])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def reliability_table(
    y_true: pd.Series, y_proba_raw: np.ndarray, y_proba_calibrated: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Decile-by-decile table of mean predicted probability vs. observed churn
    rate, before and after calibration -- the numeric answer to "does a
    predicted probability of 0.80 mean an observed churn rate near 80%?".

    Each column pair (raw / calibrated) is binned independently by quantile,
    so a row compares the same RANK (e.g. "riskiest decile") on both sides,
    not necessarily the same probability range.
    """
    from sklearn.calibration import calibration_curve

    frac_raw, mean_raw = calibration_curve(y_true, y_proba_raw, n_bins=n_bins, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(
        y_true, y_proba_calibrated, n_bins=n_bins, strategy="quantile"
    )
    n = min(len(frac_raw), len(frac_cal))
    return pd.DataFrame(
        {
            "decil": range(1, n + 1),
            "prob_media_predicha_sin_calibrar": mean_raw[:n],
            "tasa_churn_observada_sin_calibrar": frac_raw[:n],
            "prob_media_predicha_calibrada": mean_cal[:n],
            "tasa_churn_observada_calibrada": frac_cal[:n],
        }
    )


def plot_reliability_diagram(
    y_true: pd.Series,
    y_proba_raw: np.ndarray,
    y_proba_calibrated: np.ndarray,
    model_name: str,
    output_path: Path,
    n_bins: int = 10,
    calibration_method: str = "calibrado",
) -> Path:
    """Reliability diagram: mean predicted probability vs. observed churn rate
    per bin, before and after calibration (see `select_calibrator` for how
    the method -- isotonic or sigmoid -- is chosen).

    This is the visual answer to "if your goal is to estimate a churn
    PROBABILITY, how do you show that 0.80 really means about 80% risk?" --
    a point on the dashed diagonal means "predicted == observed"; ROC-AUC,
    PR-AUC and recall cannot show this because all three are unchanged by any
    monotonic recalibration of the predicted probabilities.
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    frac_raw, mean_raw = calibration_curve(y_true, y_proba_raw, n_bins=n_bins, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(
        y_true, y_proba_calibrated, n_bins=n_bins, strategy="quantile"
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Calibración perfecta")
    ax.plot(mean_raw, frac_raw, marker="o", label="Sin calibrar")
    ax.plot(mean_cal, frac_cal, marker="o", label=f"Calibrado ({calibration_method})")
    ax.set_xlabel("Probabilidad media predicha")
    ax.set_ylabel("Tasa de churn observada")
    ax.set_title(f"Diagrama de confiabilidad — {model_name} (test set)")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_matrix(pipeline, X_test, y_test, model_name: str, output_path: Path) -> Path:
    """Saves a confusion matrix for `pipeline` on the held-out test set."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["renovado", "churned"]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Matriz de confusión — {model_name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_importance(pipeline, X_test, y_test, model_name: str, output_path: Path) -> Path:
    """Saves a feature-importance bar chart for `pipeline`.

    Uses, in order of preference: `feature_importances_` (tree impurity-based,
    e.g. a tree-ensemble model), `coef_` (linear models, e.g. logistic
    regression), and permutation importance as a universal fallback for
    anything else (e.g. HistGradientBoosting or the neural network, which
    expose neither of the above) -- so this never silently produces nothing
    regardless of which candidate wins the comparison.
    """
    import matplotlib.pyplot as plt

    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        xlabel = "Importancia (impureza)"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        xlabel = "|coeficiente|"
    else:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            pipeline, X_test, y_test, n_repeats=10, random_state=42, scoring="roc_auc"
        )
        importances = result.importances_mean
        xlabel = "Importancia por permutación (caída de ROC-AUC)"

    order = np.argsort(importances)[::-1]
    names = [FEATURE_COLUMNS[i] for i in order]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names[::-1], importances[order][::-1], color="#4C72B0")
    ax.set_xlabel(xlabel)
    ax.set_title(f"Importancia de features — {model_name}")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# --- CLI ---

if __name__ == "__main__":
    import typer

    from churn_detection.config import (
        FIGURES_DIR,
        MLFLOW_EXPERIMENT_NAME,
        MLFLOW_TRACKING_URI,
        MODELS_DIR,
        PROCESSED_DATA_DIR,
    )

    app = typer.Typer(help="Train and compare churn models, tracked with MLflow.")

    @app.command()
    def train_models(
        input_path: Path = PROCESSED_DATA_DIR / "churn_ciclos.csv",
        model_output_path: Path = MODELS_DIR / "churn_model.joblib",
        comparison_path: Path = PROCESSED_DATA_DIR / "comparacion_modelos_churn.csv",
        tuning_metrics_path: Path = PROCESSED_DATA_DIR / "metricas_tuning_ganador.csv",
        calibration_path: Path = PROCESSED_DATA_DIR / "calibracion_modelo_ganador.csv",
        figure_path: Path = FIGURES_DIR / "07_curvas_roc.png",
        calibration_figure_path: Path = FIGURES_DIR / "14_calibracion.png",
        selection_metric: str = "log_loss",
        min_val_size: int = 100,
        min_train_size: int = 700,
        test_size: float = 0.20,
    ) -> None:
        import joblib
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mlflow
        import mlflow.sklearn
        from loguru import logger
        from sklearn.metrics import roc_curve

        df = pd.read_csv(input_path)
        split = ChronologicalSplitter(test_size=test_size).split(df, FEATURE_COLUMNS)
        train_rate = split.y_train.mean()
        test_rate = split.y_test.mean()
        logger.info(
            f"Corte cronológico en fecha_vencimiento={split.cutoff_date.date()}. "
            f"Train: {len(split.X_train)} ciclos (churn={train_rate:.1%}), todos "
            f"resueltos <= corte. Test: {len(split.X_test)} ciclos "
            f"(churn={test_rate:.1%}), todos resueltos después del corte."
        )

        # Un solo WalkForwardGroupSplitter para las 4 etapas que necesitan
        # validación cruzada dentro del train (selección de candidato,
        # tuning, calibración y umbral) -- ver su docstring y la nota del
        # módulo sobre por qué reemplaza a StratifiedGroupKFold ahí.
        # min_train_size descarta folds cuyo train sea tan chico (o tan
        # anterior a RELIABLE_ACCESS_TRACKING_SINCE) que la estimación por
        # fold sea inestable, en vez de dejar que unos pocos folds diminutos
        # muevan el promedio.
        wf_cv = WalkForwardGroupSplitter(
            dates=split.dates_train, min_val_size=min_val_size, min_train_size=min_train_size
        )
        wf_windows = wf_cv.months()
        wf_n_splits = wf_cv.get_n_splits(groups=split.groups_train)
        logger.info(
            f"Validación cruzada walk-forward: {wf_n_splits} folds "
            f"(min_val_size={min_val_size}, min_train_size={min_train_size}), "
            f"meses disponibles en train: {[str(m) for m in wf_windows]}."
        )

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        candidates: list[ChurnModelCandidate] = [
            LogisticRegressionCandidate(),
            HistGradientBoostingCandidate(),
            NeuralNetworkCandidate(),
        ]

        # --- Fase 1: elegir candidato ganador por validación cruzada -------
        # Cada uno de los 3 candidatos se valida con wf_cv SOLO sobre el
        # train set. El ganador se decide con `cv_mean_{selection_metric}`
        # -- nunca con el test set, para no contaminar la única evaluación
        # que queda para reportar desempeño final honesto. La curva ROC
        # comparativa también se dibuja con probabilidades out-of-fold de
        # esa misma validación cruzada walk-forward, no con el test set; la
        # ventana inicial de train nunca es validación (ver
        # walk_forward_oof_predict_proba), así que la curva se dibuja solo
        # con las filas que sí tuvieron una predicción fuera de muestra
        # real. Cada candidato además se ajusta una vez sobre todo el train
        # set y se evalúa en test aquí, pero solo para la tabla descriptiva
        # ("test_..." en comparacion_modelos_churn.csv y MLflow) -- no
        # participa en la elección ni en la figura.
        results = []
        cv_summaries_raw: dict[str, dict[str, float]] = {}
        fig, ax = plt.subplots(figsize=(7, 6))

        for candidate in candidates:
            with mlflow.start_run(run_name=f"{candidate.name}_entrenamiento"):
                logger.info(f"{candidate.name}: validación cruzada walk-forward...")
                cv_summary_raw = cross_validate_candidate(
                    candidate, split.X_train, split.y_train, split.groups_train, wf_cv
                )
                cv_summary = _humanize_cv_summary(cv_summary_raw)

                oof_proba, covered = walk_forward_oof_predict_proba(
                    candidate.build_pipeline(),
                    split.X_train,
                    split.y_train,
                    split.groups_train,
                    wf_cv,
                )

                pipeline = candidate.build_pipeline()
                pipeline.fit(split.X_train, split.y_train)
                y_pred = pipeline.predict(split.X_test)
                y_proba = pipeline.predict_proba(split.X_test)[:, 1]
                test_metrics = ModelEvaluator.evaluate(split.y_test, y_pred, y_proba)

                mlflow.log_params(
                    {
                        "model": candidate.name,
                        "stage": "entrenamiento",
                        "n_train": len(split.X_train),
                        "n_test": len(split.X_test),
                        "n_features": len(FEATURE_COLUMNS),
                        "wf_n_splits": wf_n_splits,
                        "min_val_size": min_val_size,
                        "min_train_size": min_train_size,
                    }
                )
                mlflow.log_metrics(cv_summary)
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
                mlflow.sklearn.log_model(
                    pipeline, artifact_path="model", serialization_format="pickle"
                )

                row = {"modelo": candidate.name, **cv_summary}
                row.update({f"test_{k}": v for k, v in test_metrics.items()})
                results.append(row)
                logger.success(
                    f"{candidate.name} (validación cruzada): "
                    f"cv_mean_{selection_metric}={cv_summary[f'cv_mean_{selection_metric}']:.4f} "
                    f"(test, solo informativo: {test_metrics})"
                )
                cv_summaries_raw[candidate.name] = cv_summary_raw

            fpr, tpr, _ = roc_curve(split.y_train[covered], oof_proba[covered])
            ax.plot(
                fpr, tpr, label=f"{candidate.name} (AUC={cv_summary['cv_mean_roc_auc']:.3f})"
            )

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title("Curvas ROC por modelo — validación cruzada walk-forward sobre el train")
        ax.legend()
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"Wrote {figure_path}")

        comparison = pd.DataFrame(results).sort_values(
            f"cv_mean_{selection_metric}",
            ascending=selection_metric in _LOSS_METRICS,
        )
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(comparison_path, index=False)
        logger.info(f"Comparación de modelos (validación cruzada):\n{comparison}")

        best_name = max(
            cv_summaries_raw, key=lambda name: cv_summaries_raw[name][f"cv_mean_{selection_metric}"]
        )
        winner = next(c for c in candidates if c.name == best_name)
        logger.info(f"Modelo ganador por validación cruzada ({selection_metric}): {best_name}")

        # --- Fase 2: tuning de hiperparámetros, elegido también por CV -----
        tuner = HyperparameterTuner(cv=wf_cv, scoring=selection_metric)
        grid = winner.param_grid()
        grid_list = grid if isinstance(grid, list) else [grid] if grid else []
        n_combos = sum(
            int(np.prod([len(v) for v in sub_grid.values()])) if sub_grid else 0
            for sub_grid in grid_list
        ) or 1
        logger.info(
            f"{best_name}: probando {n_combos} combinaciones de hiperparámetros "
            f"con validación cruzada walk-forward ({wf_n_splits} folds), "
            f"n_jobs={winner.n_jobs}..."
        )

        with mlflow.start_run(run_name=f"{best_name}_tuning"):
            best_pipeline, best_params, cv_metrics_raw = tuner.tune(
                winner.build_pipeline(),
                grid,
                split.X_train,
                split.y_train,
                split.groups_train,
                n_jobs=winner.n_jobs,
            )
            cv_metrics = _humanize_cv_summary(cv_metrics_raw)

            y_pred_raw = best_pipeline.predict(split.X_test)
            y_proba_raw = best_pipeline.predict_proba(split.X_test)[:, 1]
            tuned_metrics_raw = ModelEvaluator.evaluate(split.y_test, y_pred_raw, y_proba_raw)

            mlflow.log_params(
                {
                    "model": best_name,
                    "stage": "tuning",
                    "n_train": len(split.X_train),
                    "n_test": len(split.X_test),
                    "n_features": len(FEATURE_COLUMNS),
                    "wf_n_splits": wf_n_splits,
                    "min_val_size": min_val_size,
                    "min_train_size": min_train_size,
                    "n_combinaciones": n_combos,
                    "scoring": selection_metric,
                    **best_params,
                }
            )
            # Métricas de la mejor combinación de hiperparámetros encontrada:
            # media y desviación estándar, a través de los folds del
            # GridSearchCV, de cada métrica en CV_SCORING (elegida por
            # validación, ver HyperparameterTuner).
            mlflow.log_metrics(cv_metrics)
            # Métricas del modelo ya afinado (sin calibrar todavía) en el
            # test set held-out -- reporte final, no usado para elegir nada.
            mlflow.log_metrics({f"test_sin_calibrar_{k}": v for k, v in tuned_metrics_raw.items()})
            mlflow.sklearn.log_model(
                best_pipeline, artifact_path="model", serialization_format="pickle"
            )

            # Misma idea que la tabla "comparison" de la Fase 1, pero acá
            # una fila por métrica: media y desviación del CV al lado del
            # valor final en el test set, para el modelo ganador afinado.
            tuning_metrics = pd.DataFrame(
                {
                    "metrica": list(CV_SCORING.keys()),
                    "cv_mean": [cv_metrics[f"cv_mean_{m}"] for m in CV_SCORING],
                    "cv_std": [cv_metrics[f"cv_std_{m}"] for m in CV_SCORING],
                    "test": [tuned_metrics_raw[m] for m in CV_SCORING],
                }
            )
            tuning_metrics_path.parent.mkdir(parents=True, exist_ok=True)
            tuning_metrics.to_csv(tuning_metrics_path, index=False)
            logger.info(
                f"Métricas de tuning ({best_name}) — CV (media/desviación) vs. "
                f"test (sin calibrar):\n{tuning_metrics}"
            )

            logger.success(f"{best_name} (tuning): mejores params={best_params}")

        # --- Fase 3: calibración y umbral, ambos elegidos por CV ---
        # El tuning anterior eligió el mejor modelo y sus hiperparámetros por
        # discriminación y calibración conjuntas, ya que log_loss y
        # brier_score son proper scoring rules sensibles a ambas, pero eso
        # no garantiza que las probabilidades ya salgan bien calibradas ni
        # que el umbral por defecto de 0.5 sea el mejor punto de corte para
        # F1. Este paso verifica y corrige ambas cosas antes de guardar el
        # modelo final, que es justamente lo que se usa para estimar una
        # probabilidad de abandono y clasificar el riesgo en producción, ver
        # churn_detection/modeling/predict.py.
        logger.info(f"{best_name}: generando probabilidades out-of-fold walk-forward para calibración...")
        oof_proba, covered = walk_forward_oof_predict_proba(
            clone(best_pipeline), split.X_train, split.y_train, split.groups_train, wf_cv
        )
        # La ventana inicial de train nunca fue validación (no hay datos
        # anteriores con los que entrenar un fold que la valide -- ver
        # walk_forward_oof_predict_proba), así que calibración y umbral se
        # calculan únicamente sobre las filas que sí tuvieron una
        # probabilidad fuera de muestra real, nunca sobre todo split.X_train.
        y_covered = split.y_train[covered]
        groups_covered = split.groups_train[covered]
        dates_covered = split.dates_train[covered]
        oof_covered = oof_proba[covered]
        logger.info(
            f"{best_name}: {covered.sum()}/{len(covered)} ciclos de train tuvieron "
            "probabilidad out-of-fold walk-forward (el resto es la ventana inicial, "
            "sin datos anteriores con los que validarla)."
        )

        # Folds walk-forward de nivel interno, sobre el subconjunto cubierto,
        # para cross-validar el método de calibración y el umbral -- mismo
        # principio que wf_cv, aplicado un nivel más abajo porque
        # select_calibrator/select_threshold trabajan sobre oof_covered, no
        # sobre split.X_train completo. Usa min_val_size también como piso
        # del train, no min_train_size: lo que se ajusta acá es un
        # calibrador de 1-2 parámetros (isotónica o sigmoid) o un barrido de
        # umbral, no el pipeline de 11 variables -- exigirle el mismo mínimo
        # de filas que a wf_cv deja muy pocos folds para una comparación que
        # necesita varios para ser estable (ver select_calibrator).
        calib_cv = WalkForwardGroupSplitter(
            dates=dates_covered, min_val_size=min_val_size, min_train_size=min_val_size
        )
        calib_folds = list(calib_cv.split(groups=groups_covered))
        logger.info(f"{best_name}: {len(calib_folds)} folds walk-forward para calibración y umbral.")

        # El método de calibración también se elige por validación cruzada
        # (Log Loss, sobre las propias probabilidades out-of-fold), no se
        # aplica isotónica sin más: con pocas observaciones out-of-fold en
        # los deciles más extremos, isotónica puede sobreajustar un escalón
        # muy pronunciado que no generaliza (ver docstring de
        # select_calibrator, incluyendo por qué Log Loss y no Brier Score).
        # sigmoid (Platt scaling) es más rígido y menos propenso a ese
        # sobreajuste.
        calibration_method, calibrator, calibration_cv_log_loss = select_calibrator(
            oof_covered, y_covered, calib_folds
        )
        logger.info(
            f"{best_name}: método de calibración elegido por validación cruzada "
            f"(Log Loss): {calibration_method} {calibration_cv_log_loss}"
        )

        # El umbral de decisión también se elige por validación cruzada, no
        # se deja fijo en 0.5 ni se calcula con un solo barrido sobre toda
        # la muestra out-of-fold. Se aplica el calibrador a esas
        # probabilidades out-of-fold para obtener la probabilidad calibrada
        # que el modelo final realmente entregaría en cada ciclo de
        # entrenamiento, y select_threshold promedia el punto de corte que
        # maximiza F1 sobre los mismos folds walk-forward, en lugar de
        # confiar en un solo barrido.
        oof_proba_calibrated_covered = np.clip(calibrator.predict(oof_covered), 0.0, 1.0)
        threshold = select_threshold(y_covered, oof_proba_calibrated_covered, calib_folds)
        logger.info(
            f"{best_name}: umbral de decisión elegido por validación cruzada walk-forward (F1): "
            f"{threshold:.4f}"
        )

        # Métricas promedio de validación cruzada del modelo YA calibrado:
        # se reutilizan calib_folds (los mismos folds walk-forward que
        # generaron oof_covered), evaluando en cada fold con las
        # probabilidades calibradas fuera de muestra y el umbral final. Es
        # el mismo resumen (media y desviación estándar entre folds) que
        # cross_validate_candidate y HyperparameterTuner ya reportan para
        # elegir modelo e hiperparámetros, pero aplicado al pipeline
        # calibrado completo en vez de al pipeline crudo.
        oof_pred_calibrated_covered = (oof_proba_calibrated_covered >= threshold).astype(int)
        cv_calibrated_folds: dict[str, list[float]] = {metric: [] for metric in CV_SCORING}
        for _, val_idx in calib_folds:
            fold_metrics = ModelEvaluator.evaluate(
                y_covered.iloc[val_idx],
                oof_pred_calibrated_covered[val_idx],
                oof_proba_calibrated_covered[val_idx],
            )
            for metric_name, value in fold_metrics.items():
                cv_calibrated_folds[metric_name].append(value)
        cv_calibrated_report = "\n".join(
            f"  {metric_name}: {np.mean(values):.4f} (+/- {np.std(values):.4f})"
            for metric_name, values in cv_calibrated_folds.items()
        )
        logger.success(
            f"Métricas promedio de validación cruzada del modelo calibrado "
            f"({best_name}, umbral {threshold:.4f}, calibrado con {calibration_method}):\n"
            f"{cv_calibrated_report}"
        )

        calibrated_model = CalibratedChurnModel(
            pipeline=best_pipeline, calibrator=calibrator, threshold=threshold
        )

        y_proba_calibrated = calibrated_model.predict_proba(split.X_test)[:, 1]
        y_pred_calibrated = calibrated_model.predict(split.X_test)
        tuned_metrics_calibrated = ModelEvaluator.evaluate(
            split.y_test, y_pred_calibrated, y_proba_calibrated
        )
        logger.info(
            f"Calibración y umbral en test. Antes (umbral 0.5, sin calibrar): \n"
            f"precision={tuned_metrics_raw['precision']:.4f}, "
            f"recall={tuned_metrics_raw['recall']:.4f}, "
            f"log_loss={tuned_metrics_raw['log_loss']:.4f}, "
            f"brier={tuned_metrics_raw['brier_score']:.4f}, "
            f"f1={tuned_metrics_raw['f1']:.4f}. \n"
            f"Después (umbral {threshold:.4f}, calibrado con {calibration_method}): \n"
            f"precision={tuned_metrics_calibrated['precision']:.4f}, "
            f"recall={tuned_metrics_calibrated['recall']:.4f}, "
            f"log_loss={tuned_metrics_calibrated['log_loss']:.4f}, "
            f"brier={tuned_metrics_calibrated['brier_score']:.4f}, "
            f"f1={tuned_metrics_calibrated['f1']:.4f}. \n"
            "roc_auc y pr_auc no deberían cambiar por la calibración, ya que ambos métodos "
            "son monótonos, pero sí pueden cambiar por el nuevo umbral en las métricas que "
            "dependen de una clasificación dura."
        )

        with mlflow.start_run(run_name=f"{best_name}_calibracion"):
            mlflow.log_params(
                {
                    "model": best_name,
                    "stage": "calibracion",
                    "metodo": calibration_method,
                    "threshold": threshold,
                }
            )
            mlflow.log_metrics({f"cv_log_loss_{k}": v for k, v in calibration_cv_log_loss.items()})
            mlflow.log_metrics({f"sin_calibrar_{k}": v for k, v in tuned_metrics_raw.items()})
            mlflow.log_metrics({f"calibrado_{k}": v for k, v in tuned_metrics_calibrated.items()})
            mlflow.log_metrics(
                {
                    f"cv_calibrado_mean_{k}": float(np.mean(v))
                    for k, v in cv_calibrated_folds.items()
                }
            )
            mlflow.log_metrics(
                {
                    f"cv_calibrado_std_{k}": float(np.std(v))
                    for k, v in cv_calibrated_folds.items()
                }
            )
            mlflow.sklearn.log_model(
                calibrated_model, artifact_path="model", serialization_format="pickle"
            )

        calibration_table = reliability_table(split.y_test, y_proba_raw, y_proba_calibrated)
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        calibration_table.to_csv(calibration_path, index=False)
        logger.info(f"Tabla de confiabilidad (test set):\n{calibration_table}")

        cal_fig_path = plot_reliability_diagram(
            split.y_test,
            y_proba_raw,
            y_proba_calibrated,
            best_name,
            calibration_figure_path,
            calibration_method=calibration_method,
        )
        logger.success(f"Wrote {cal_fig_path}")

        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(calibrated_model, model_output_path)
        logger.success(
            f"Mejor modelo por validación cruzada ({selection_metric}): {best_name}, "
            f"calibrado. Guardado en {model_output_path}"
        )

        _figpath = figure_path.parent
        # Matriz de confusión / importancia de features se calculan sobre el
        # pipeline SIN el wrapper de calibración: la calibración isotónica es
        # monótona, así que no cambia el orden de las predicciones ni la
        # importancia relativa de las features, solo la escala de la
        # probabilidad -- y estas dos funciones ya saben inspeccionar un
        # `Pipeline` con un paso "model" (ver plot_feature_importance).
        cm_path = plot_confusion_matrix(
            best_pipeline,
            split.X_test,
            split.y_test,
            best_name,
            _figpath / "12_matriz_confusion_sin_calibrar.png",
        )
        cm_path = plot_confusion_matrix(
            calibrated_model,
            split.X_test,
            split.y_test,
            best_name,
            _figpath / "13_matriz_confusion_calibrada.png",
        )
        logger.success(f"Wrote {cm_path}")
        fi_path = plot_feature_importance(
            best_pipeline,
            split.X_test,
            split.y_test,
            best_name,
            _figpath / "14_importancia_features.png",
        )
        logger.success(f"Wrote {fi_path}")

    app()
