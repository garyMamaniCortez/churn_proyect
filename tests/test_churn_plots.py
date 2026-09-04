import numpy as np
import pandas as pd
import pytest

from churn_detection.churn_plots import ChurnEDAFigureGenerator


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(11)
    n = 40
    estado = (["renovado"] * 15) + (["churned"] * 15) + (["censurado"] * 10)
    return pd.DataFrame(
        {
            "persona_id": range(1, n + 1),
            "estado_ciclo": estado,
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


def test_generate_all_writes_four_png_files(tmp_path, sample_df):
    generator = ChurnEDAFigureGenerator(tmp_path)
    paths = generator.generate_all(sample_df)

    assert len(paths) == 4
    for path in paths:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


def test_handles_missing_optional_columns_gracefully(tmp_path, sample_df):
    minimal_df = sample_df[["persona_id", "estado_ciclo", "tenure_dias", "recencia_dias"]]
    generator = ChurnEDAFigureGenerator(tmp_path)
    paths = generator.generate_all(minimal_df)

    assert len(paths) == 4
    for path in paths:
        assert path.exists()
