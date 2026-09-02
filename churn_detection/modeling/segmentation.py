"""Customer segmentation of clientes_segmentacion.csv via K-Means clustering.

Feature selection (why these 11 and not all ~22 columns)
-----------------------------------------------------------
`CLUSTER_FEATURES` below is a deliberately curated subset, not every numeric
column in the table. It includes three "trend" features added after the first
clustering pass found weak structure (silhouette ~0.23) using only "level"
features: `ratio_actividad_reciente` (is this client's last 30 days busier or
quieter than their own historical pace), `cv_gap_visitas` (how regular their
attendance rhythm is, not just how often), and `monto_gastado_ultimos_90d`
(recent vs. total spend). The idea: two clients can look identical on
recency/frequency alone while one is accelerating and the other decelerating,
a distinction the original 8 features couldn't see.

- Dropped for redundancy: `n_checkins_total`, `n_checkins_ultimos_30d`,
  `n_checkins_ultimos_90d` and `n_inscripciones_membresia` are all strongly
  correlated with `recencia_dias` and `frecuencia_visitas_semanal` (see
  reports/figures/03_correlacion.png) -- keeping both would let "how often
  this client visits" dominate the distance metric twice over.
- Dropped for near-zero variance: `es_multisucursal` (under 4% True) and
  `tiene_pago_pendiente` (under 1% True) barely separate anyone and would add
  noise more than signal to K-Means. They're kept in the output table so
  clusters can still be *described* by them after the fact.
- Dropped as categorical: `dia_semana_mas_frecuente` isn't numeric; encoding
  and including it is a reasonable future iteration, not done here.
- `monto_promedio_venta` dropped: redundant with `monto_total_gastado` for
  clustering purposes (correlated, and ticket size is a finer distinction than
  this first segmentation pass needs).

Missing data
-------------
11.6% to 22.9% of clients have NaN in one or more clustering features (see
features.py -- mostly clients with no reliably-tracked check-ins, per
RELIABLE_ACCESS_TRACKING_SINCE). K-Means cannot run on NaNs. Rather than drop
those clients (losing exactly the group that may be most "at risk") or impute
silently (hiding the gap), `ClusteringDatasetPreparer` flags every client with
any missing clustering feature in `datos_incompletos`, then imputes with the
column median. The flag survives into the output table so any cluster that
turns out to concentrate incomplete-data clients is visible, not hidden.

Transforms
-----------
`recencia_dias`, `monto_total_gastado`, `frecuencia_visitas_semanal` and
`n_inscripciones_total` are right-skewed (see reports/figures/02_*.png) --
log1p-transformed before scaling so a handful of extreme values don't dominate
Euclidean distance. Everything is then standardized (zero mean, unit variance)
since K-Means is scale-sensitive and these features live on very different
scales (days vs. percentages vs. currency).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

CLUSTER_FEATURES = [
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

LOG1P_FEATURES = [
    "recencia_dias",
    "monto_total_gastado",
    "frecuencia_visitas_semanal",
    "n_inscripciones_total",
    "ratio_actividad_reciente",
    "cv_gap_visitas",
    "monto_gastado_ultimos_90d",
]


@dataclass
class ClusteringDataset:
    """Everything needed to fit/describe a K-Means model, plus a data-quality flag."""

    persona_ids: pd.Series
    X: np.ndarray
    datos_incompletos: pd.Series
    feature_names: list[str]


class ClusteringDatasetPreparer:
    """Turns the segmentation table into a clean numeric matrix ready for K-Means.

    SRP: this class only prepares X (impute, transform, scale). It does not
    choose k or fit a model -- see `KMeansSegmenter`.
    """

    def __init__(
        self,
        feature_columns: list[str] = CLUSTER_FEATURES,
        log1p_columns: list[str] = LOG1P_FEATURES,
    ) -> None:
        self._feature_columns = feature_columns
        self._log1p_columns = log1p_columns
        self._scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame) -> ClusteringDataset:
        subset = df[self._feature_columns].copy()
        datos_incompletos = subset.isna().any(axis=1)
        subset = subset.fillna(subset.median(numeric_only=True))

        for col in self._log1p_columns:
            subset[col] = np.log1p(subset[col].clip(lower=0))

        X = self._scaler.fit_transform(subset)
        return ClusteringDataset(
            persona_ids=df["persona_id"].reset_index(drop=True),
            X=X,
            datos_incompletos=datos_incompletos.reset_index(drop=True),
            feature_names=list(self._feature_columns),
        )


class KMeansSegmenter:
    """Fits K-Means and helps choose k via silhouette score / inertia (elbow)."""

    algorithm_name = "kmeans"

    def __init__(self, random_state: int = 42) -> None:
        self._random_state = random_state

    def _build_model(self, k: int):
        return KMeans(n_clusters=k, random_state=self._random_state, n_init=10)

    def score_k_range(self, X: np.ndarray, k_range: range) -> pd.DataFrame:
        rows = []
        for k in k_range:
            model = self._build_model(k)
            labels = model.fit_predict(X)
            rows.append(
                {
                    "algoritmo": self.algorithm_name,
                    "k": k,
                    "inertia": model.inertia_,
                    "silhouette": silhouette_score(X, labels),
                }
            )
        return pd.DataFrame(rows)

    def fit_predict(self, X: np.ndarray, k: int) -> tuple[np.ndarray, KMeans]:
        model = self._build_model(k)
        labels = model.fit_predict(X)
        return labels, model


class BaseSegmenter(ABC):
    """Shared scoring/fitting logic for the alternative clustering algorithms.

    K-Means keeps its own standalone class above (kept backward compatible, and
    it's the one algorithm with a natural "inertia"/elbow interpretation).
    GMM and hierarchical clustering share this base since both plug a model
    into the same score/fit workflow, only the model constructor differs.
    """

    algorithm_name: str

    @abstractmethod
    def _build_model(self, k: int): ...

    def score_k_range(self, X: np.ndarray, k_range: range) -> pd.DataFrame:
        rows = []
        for k in k_range:
            model = self._build_model(k)
            labels = model.fit_predict(X)
            row = {
                "algoritmo": self.algorithm_name,
                "k": k,
                "silhouette": silhouette_score(X, labels),
            }
            if hasattr(model, "bic"):
                row["bic"] = model.bic(X)
            rows.append(row)
        return pd.DataFrame(rows)

    def fit_predict(self, X: np.ndarray, k: int):
        model = self._build_model(k)
        labels = model.fit_predict(X)
        return labels, model


class GaussianMixtureSegmenter(BaseSegmenter):
    """Soft clustering: assumes each segment is a Gaussian blob in feature space.

    Unlike K-Means (which assumes round, equal-size clusters), GMM allows
    elongated/differently-sized clusters, which can matter here given how
    skewed some of the transformed features still are.
    """

    algorithm_name = "gmm"

    def __init__(self, random_state: int = 42) -> None:
        self._random_state = random_state

    def _build_model(self, k: int) -> GaussianMixture:
        return GaussianMixture(n_components=k, random_state=self._random_state, n_init=5)


class HierarchicalSegmenter(BaseSegmenter):
    """Agglomerative (bottom-up) clustering with Ward linkage.

    Does not assume clusters are convex/round the way K-Means does, and its
    dendrogram gives a complementary view of whether the data has natural
    nested structure at all, useful given K-Means alone found a weak split.
    """

    algorithm_name = "hierarchical"

    def __init__(self, linkage: str = "ward") -> None:
        self._linkage = linkage

    def _build_model(self, k: int) -> AgglomerativeClustering:
        return AgglomerativeClustering(n_clusters=k, linkage=self._linkage)


def compare_algorithms(
    X: np.ndarray, k_range: range, segmenters: dict[str, KMeansSegmenter | BaseSegmenter]
) -> pd.DataFrame:
    """Runs every segmenter across `k_range` and stacks their silhouette scores
    into one comparison table, so the choice of algorithm and k can be made
    together rather than picking k for K-Means alone."""
    frames = [seg.score_k_range(X, k_range) for seg in segmenters.values()]
    return pd.concat(frames, ignore_index=True)


def build_cluster_profile(
    df: pd.DataFrame, labels: np.ndarray, feature_columns: list[str] = CLUSTER_FEATURES
) -> pd.DataFrame:
    """Mean of each clustering feature per cluster, plus cluster size -- for naming/
    interpreting the segments. Uses the ORIGINAL (untransformed) feature values so
    the numbers are directly readable, not log/z-score units."""
    profile = df.assign(cluster=labels).groupby("cluster")[feature_columns].mean()
    profile["n_clientes"] = df.assign(cluster=labels).groupby("cluster").size()
    return profile.reset_index()


# --- CLI ---

if __name__ == "__main__":
    from pathlib import Path

    import typer

    from churn_detection.config import FIGURES_DIR, PROCESSED_DATA_DIR

    app = typer.Typer(
        help="Segment clients by comparing K-Means, GMM, and hierarchical clustering."
    )

    @app.command()
    def build_segmentation(
        input_path: Path = PROCESSED_DATA_DIR / "clientes_segmentacion.csv",
        output_path: Path = PROCESSED_DATA_DIR / "clientes_segmentados.csv",
        profile_path: Path = PROCESSED_DATA_DIR / "perfil_clusters.csv",
        comparison_path: Path = PROCESSED_DATA_DIR / "comparacion_algoritmos.csv",
        figure_path: Path = FIGURES_DIR / "06_seleccion_k.png",
        k_min: int = 2,
        k_max: int = 8,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from loguru import logger
        import matplotlib.pyplot as plt

        df = pd.read_csv(input_path)
        preparer = ClusteringDatasetPreparer()
        prepared = preparer.fit_transform(df)
        k_range = range(k_min, k_max + 1)

        segmenters: dict[str, object] = {
            "kmeans": KMeansSegmenter(),
            "gmm": GaussianMixtureSegmenter(),
            "hierarchical": HierarchicalSegmenter(),
        }
        comparison = compare_algorithms(prepared.X, k_range, segmenters)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(comparison_path, index=False)
        logger.info(f"Comparación de algoritmos (silhouette por k):\n{comparison}")

        best_row = comparison.loc[comparison["silhouette"].idxmax()]
        best_algo, chosen_k = best_row["algoritmo"], int(best_row["k"])
        logger.success(
            f"Mejor combinación por silhouette: {best_algo} con k={chosen_k} "
            f"(silhouette={best_row['silhouette']:.3f})"
        )

        labels, _model = segmenters[best_algo].fit_predict(prepared.X, chosen_k)

        out = df.copy()
        out["cluster"] = labels
        out["datos_incompletos"] = prepared.datos_incompletos.to_numpy()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        logger.success(f"Wrote {len(out)} client rows with cluster labels to {output_path}")

        profile = build_cluster_profile(df, labels)
        profile.to_csv(profile_path, index=False)
        logger.success(f"Wrote cluster profile to {profile_path}")
        logger.info(f"Perfil de clusters ({best_algo}, k={chosen_k}):\n{profile}")

        fig, ax = plt.subplots(figsize=(7, 5))
        for algo_name, group in comparison.groupby("algoritmo"):
            ax.plot(group["k"], group["silhouette"], marker="o", label=algo_name)
        ax.axvline(chosen_k, color="crimson", linestyle="--", linewidth=1)
        ax.set_xlabel("k")
        ax.set_ylabel("Silhouette")
        ax.set_title(f"Silhouette por algoritmo y k (elegido: {best_algo}, k={chosen_k})")
        ax.legend()
        fig.tight_layout()
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"Wrote {figure_path}")

    app()
