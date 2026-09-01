import pandas as pd
import pytest

from churn_detection.plots import EDAFigureGenerator


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "persona_id": range(1, 21),
            "tenure_dias": [30 * i for i in range(1, 21)],
            "porcentaje_uso_membresia": [0.1 * (i % 10) for i in range(20)],
            "recencia_dias": [i for i in range(20)],
            "frecuencia_visitas_semanal": [0.5 * i for i in range(20)],
            "hora_promedio_checkin": [8 + (i % 12) for i in range(20)],
            "monto_total_gastado": [100.0 * i for i in range(20)],
            "n_inscripciones_total": [i % 5 + 1 for i in range(20)],
            "n_checkins_total": [i for i in range(20)],
            "n_checkins_ultimos_30d": [i % 8 for i in range(20)],
            "hora_checkin_std": [1.0] * 10 + [None] * 10,
            "pct_visitas_fin_de_semana": [0.1] * 20,
            "monto_promedio_venta": [50.0] * 20,
            "dia_semana_mas_frecuente": (["Monday", "Tuesday", "Saturday"] * 7)[:20],
            "es_multisucursal": [i % 2 == 0 for i in range(20)],
            "tiene_pago_pendiente": [i % 5 == 0 for i in range(20)],
        }
    )


def test_generate_all_writes_five_png_files(tmp_path, sample_df):
    generator = EDAFigureGenerator(tmp_path)
    paths = generator.generate_all(sample_df)

    assert len(paths) == 5
    for path in paths:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


def test_output_directory_is_created_if_missing(tmp_path, sample_df):
    output_dir = tmp_path / "does" / "not" / "exist" / "yet"
    generator = EDAFigureGenerator(output_dir)
    generator.generate_all(sample_df)
    assert output_dir.exists()


def test_handles_missing_optional_columns_gracefully(tmp_path, sample_df):
    minimal_df = sample_df[["persona_id", "tenure_dias", "recencia_dias"]]
    generator = EDAFigureGenerator(tmp_path)
    paths = generator.generate_all(minimal_df)

    assert len(paths) == 5
    for path in paths:
        assert path.exists()
