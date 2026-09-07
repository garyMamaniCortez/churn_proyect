"""Train and compare churn-prediction models on churn_ciclos.csv.

Design notes (SOLID)
---------------------
- `GroupAwareSplitter` has one job: split into train/test BY `persona_id`, not
  by row. A client with 3 membership cycles has 3 correlated rows (same
  underlying person, overlapping history); if two of them landed on opposite
  sides of a random row-level split, the model could partly memorize that
  specific client instead of learning a generalizable pattern, and the test
  score would be optimistic. Splitting by persona_id keeps all of one
  client's cycles together on one side.
- `ChurnModelCandidate` (ABC): each concrete candidate only knows how to build
  its own preprocessing + estimator `Pipeline`. Adding a new candidate model
  is one small class, not a change to the training loop (Open/Closed).
- `ModelEvaluator`: single responsibility, turns predictions into a metrics
  dict. Used identically for every candidate so comparisons are apples-to-apples.
- MLflow: every candidate's params, metrics, and fitted pipeline are logged
  under one experiment, so a training run is reproducible/comparable later,
  not just "whichever model happened to win this time".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
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


class GroupAwareSplitter:
    """Train/test split grouped by persona_id, so no client's cycles straddle both sides."""

    def __init__(self, test_size: float = 0.25, random_state: int = 42) -> None:
        self._test_size = test_size
        self._random_state = random_state

    def split(self, df: pd.DataFrame, feature_columns: list[str]) -> ChurnSplit:
        labeled = df.dropna(subset=["churn_label"]).reset_index(drop=True)
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=self._test_size, random_state=self._random_state
        )
        train_idx, test_idx = next(splitter.split(labeled, groups=labeled["persona_id"]))
        train, test = labeled.iloc[train_idx], labeled.iloc[test_idx]
        return ChurnSplit(
            X_train=train[feature_columns],
            X_test=test[feature_columns],
            y_train=train["churn_label"].astype(int),
            y_test=test["churn_label"].astype(int),
        )


class ChurnModelCandidate(ABC):
    """Contract every candidate model implements: build its own pipeline."""

    name: str

    @abstractmethod
    def build_pipeline(self) -> Pipeline: ...


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
        import numpy as np

        X = np.asarray(X, dtype=float).copy()
        X[:, self._skewed_idx] = np.log1p(np.clip(X[:, self._skewed_idx], 0, None))
        return X


class KerasBinaryClassifier(BaseEstimator, ClassifierMixin):
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
        import numpy as np

        X = np.asarray(X, dtype="float32")
        y = np.asarray(y, dtype="float32")
        self.classes_ = np.array([0, 1])
        self._model = self._build_model(X.shape[1])
        self._model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self

    def predict_proba(self, X):
        import numpy as np

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

    def build_pipeline(self) -> Pipeline:
        skewed_idx = [FEATURE_COLUMNS.index(c) for c in SKEWED_FEATURES]
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("log1p_skewed", FunctionTransformer(_Log1pSkewedColumns(skewed_idx))),
                ("scaler", StandardScaler()),
                ("model", KerasBinaryClassifier()),
            ]
        )


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
                ("imputer", SimpleImputer(strategy="median")),
                ("log1p_skewed", FunctionTransformer(_Log1pSkewedColumns(skewed_idx))),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )


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


class ModelEvaluator:
    """Turns (y_true, y_pred, y_proba) into a standard classification metrics dict."""

    @staticmethod
    def evaluate(y_true: pd.Series, y_pred, y_proba) -> dict[str, float]:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "pr_auc": average_precision_score(y_true, y_proba),
        }


def plot_confusion_matrix(pipeline, X_test, y_test, model_name: str, output_path) -> object:
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


def plot_feature_importance(pipeline, X_test, y_test, model_name: str, output_path):
    """Saves a feature-importance bar chart for `pipeline`.

    Uses, in order of preference: `feature_importances_` (tree impurity-based,
    e.g. a tree-ensemble model), `coef_` (linear models, e.g. logistic
    regression), and permutation importance as a universal fallback for
    anything else (e.g. HistGradientBoosting or the neural network, which
    expose neither of the above) -- so this never silently produces nothing
    regardless of which candidate wins the comparison.
    """
    import matplotlib.pyplot as plt
    import numpy as np

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
    from pathlib import Path

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
        figure_path: Path = FIGURES_DIR / "07_curvas_roc.png",
        selection_metric: str = "roc_auc",
    ) -> None:
        import joblib
        import matplotlib

        matplotlib.use("Agg")
        from loguru import logger
        import matplotlib.pyplot as plt
        import mlflow
        import mlflow.sklearn
        from sklearn.metrics import roc_curve

        df = pd.read_csv(input_path)
        split = GroupAwareSplitter().split(df, FEATURE_COLUMNS)
        logger.info(f"Train: {len(split.X_train)} ciclos. Test: {len(split.X_test)} ciclos.")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        candidates: list[ChurnModelCandidate] = [
            LogisticRegressionCandidate(),
            HistGradientBoostingCandidate(),
            NeuralNetworkCandidate(),
        ]

        results = []
        fitted_pipelines = {}
        fig, ax = plt.subplots(figsize=(7, 6))

        for candidate in candidates:
            with mlflow.start_run(run_name=candidate.name):
                pipeline = candidate.build_pipeline()
                pipeline.fit(split.X_train, split.y_train)
                y_pred = pipeline.predict(split.X_test)
                y_proba = pipeline.predict_proba(split.X_test)[:, 1]
                metrics = ModelEvaluator.evaluate(split.y_test, y_pred, y_proba)

                mlflow.log_params(
                    {
                        "model": candidate.name,
                        "n_train": len(split.X_train),
                        "n_test": len(split.X_test),
                        "n_features": len(FEATURE_COLUMNS),
                    }
                )
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipeline, name="model", serialization_format="pickle")

                results.append({"modelo": candidate.name, **metrics})
                fitted_pipelines[candidate.name] = pipeline

                fpr, tpr, _ = roc_curve(split.y_test, y_proba)
                ax.plot(fpr, tpr, label=f"{candidate.name} (AUC={metrics['roc_auc']:.3f})")
                logger.success(f"{candidate.name}: {metrics}")

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        ax.set_title("Curvas ROC por modelo (test set)")
        ax.legend()
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"Wrote {figure_path}")

        comparison = pd.DataFrame(results).sort_values(selection_metric, ascending=False)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(comparison_path, index=False)
        logger.info(f"Comparación de modelos:\n{comparison}")

        best_name = comparison.iloc[0]["modelo"]
        best_pipeline = fitted_pipelines[best_name]
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, model_output_path)
        logger.success(
            f"Mejor modelo por {selection_metric}: {best_name}. Guardado en {model_output_path}"
        )

        _figpath = figure_path.parent
        cm_path = plot_confusion_matrix(
            best_pipeline,
            split.X_test,
            split.y_test,
            best_name,
            _figpath / "12_matriz_confusion.png",
        )
        logger.success(f"Wrote {cm_path}")
        fi_path = plot_feature_importance(
            best_pipeline,
            split.X_test,
            split.y_test,
            best_name,
            _figpath / "13_importancia_features.png",
        )
        logger.success(f"Wrote {fi_path}")

    app()
