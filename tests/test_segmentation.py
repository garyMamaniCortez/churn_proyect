import numpy as np
import pandas as pd
import pytest

from churn_detection.modeling.segmentation import (
    CLUSTER_FEATURES,
    ClusteringDatasetPreparer,
    GaussianMixtureSegmenter,
    HierarchicalSegmenter,
    KMeansSegmenter,
    build_cluster_profile,
    compare_algorithms,
)


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = 60
    return pd.DataFrame(
        {
            "persona_id": range(1, n + 1),
            "tenure_dias": rng.integers(10, 300, n),
            "porcentaje_uso_membresia": rng.uniform(0, 1, n),
            "recencia_dias": rng.integers(0, 200, n),
            "frecuencia_visitas_semanal": rng.uniform(0, 7, n),
            "hora_promedio_checkin": rng.uniform(6, 22, n),
            "pct_visitas_fin_de_semana": rng.uniform(0, 0.3, n),
            "monto_total_gastado": rng.uniform(0, 2000, n),
            "n_inscripciones_total": rng.integers(1, 10, n),
            "ratio_actividad_reciente": rng.uniform(0, 3, n),
            "cv_gap_visitas": rng.uniform(0, 2, n),
            "monto_gastado_ultimos_90d": rng.uniform(0, 800, n),
        }
    )


def test_preparer_flags_and_imputes_missing_values(sample_df):
    df = sample_df.copy()
    df.loc[0, "recencia_dias"] = np.nan
    df.loc[1, "porcentaje_uso_membresia"] = np.nan

    preparer = ClusteringDatasetPreparer()
    prepared = preparer.fit_transform(df)

    assert prepared.datos_incompletos.iloc[0]
    assert prepared.datos_incompletos.iloc[1]
    assert not prepared.datos_incompletos.iloc[2]
    assert not np.isnan(prepared.X).any()


def test_preparer_output_shape_matches_features(sample_df):
    preparer = ClusteringDatasetPreparer()
    prepared = preparer.fit_transform(sample_df)

    assert prepared.X.shape == (len(sample_df), len(CLUSTER_FEATURES))
    assert prepared.feature_names == CLUSTER_FEATURES
    assert len(prepared.persona_ids) == len(sample_df)


def test_scaled_features_are_roughly_standardized(sample_df):
    preparer = ClusteringDatasetPreparer()
    prepared = preparer.fit_transform(sample_df)

    means = prepared.X.mean(axis=0)
    stds = prepared.X.std(axis=0)
    assert np.allclose(means, 0, atol=1e-8)
    assert np.allclose(stds, 1, atol=1e-8)


def test_score_k_range_returns_one_row_per_k(sample_df):
    preparer = ClusteringDatasetPreparer()
    prepared = preparer.fit_transform(sample_df)

    segmenter = KMeansSegmenter()
    scores = segmenter.score_k_range(prepared.X, range(2, 5))

    assert list(scores["k"]) == [2, 3, 4]
    assert (scores["silhouette"] <= 1).all()
    assert (scores["silhouette"] >= -1).all()


def test_fit_predict_returns_a_label_per_client(sample_df):
    preparer = ClusteringDatasetPreparer()
    prepared = preparer.fit_transform(sample_df)

    segmenter = KMeansSegmenter()
    labels, model = segmenter.fit_predict(prepared.X, k=3)

    assert len(labels) == len(sample_df)
    assert set(labels) <= {0, 1, 2}
    assert model.n_clusters == 3


def test_build_cluster_profile_has_one_row_per_cluster_and_sizes_sum_correctly(sample_df):
    preparer = ClusteringDatasetPreparer()
    prepared = preparer.fit_transform(sample_df)
    segmenter = KMeansSegmenter()
    labels, _ = segmenter.fit_predict(prepared.X, k=3)

    profile = build_cluster_profile(sample_df, labels)

    assert len(profile) == 3
    assert profile["n_clientes"].sum() == len(sample_df)
    for col in CLUSTER_FEATURES:
        assert col in profile.columns


@pytest.mark.parametrize(
    "segmenter_factory",
    [KMeansSegmenter, GaussianMixtureSegmenter, HierarchicalSegmenter],
)
def test_every_segmenter_implements_the_same_score_and_fit_interface(sample_df, segmenter_factory):
    preparer = ClusteringDatasetPreparer()
    prepared = preparer.fit_transform(sample_df)
    segmenter = segmenter_factory()

    scores = segmenter.score_k_range(prepared.X, range(2, 5))
    assert list(scores["k"]) == [2, 3, 4]
    assert (scores["silhouette"] <= 1).all() and (scores["silhouette"] >= -1).all()
    assert (scores["algoritmo"] == segmenter.algorithm_name).all()

    labels, _model = segmenter.fit_predict(prepared.X, k=3)
    assert len(labels) == len(sample_df)
    assert len(set(labels)) <= 3


def test_compare_algorithms_stacks_every_segmenter_into_one_table(sample_df):
    preparer = ClusteringDatasetPreparer()
    prepared = preparer.fit_transform(sample_df)
    segmenters = {
        "kmeans": KMeansSegmenter(),
        "gmm": GaussianMixtureSegmenter(),
        "hierarchical": HierarchicalSegmenter(),
    }

    comparison = compare_algorithms(prepared.X, range(2, 4), segmenters)

    assert set(comparison["algoritmo"]) == {"kmeans", "gmm", "hierarchical"}
    assert len(comparison) == 3 * 2  # 3 algorithms x 2 k values
