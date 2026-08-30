import json

import pandas as pd
import pytest

from churn_detection.dataset import EXTRACTION_TARGETS, extract_raw_tables


class FakeRepository:
    """Test double implementing the same methods as ClientDataRepository."""

    def get_personas(self):
        return pd.DataFrame({"persona_id": [1, 2, 3]})

    def get_inscripciones(self):
        return pd.DataFrame({"inscripcion_id": [1, 2]})

    def get_servicios(self):
        return pd.DataFrame({"servicio_id": [1]})

    def get_registros_acceso(self):
        return pd.DataFrame({"registro_id": []})

    def get_ventas_servicios(self):
        return pd.DataFrame({"venta_servicio_id": [1, 2, 3, 4]})

    def get_pagos_pendientes(self):
        return pd.DataFrame({"pago_pendiente_id": [1]})


def test_extract_raw_tables_writes_one_csv_per_target_and_metadata(tmp_path):
    repository = FakeRepository()

    row_counts = extract_raw_tables(repository, output_dir=tmp_path)

    for table_name in EXTRACTION_TARGETS:
        csv_path = tmp_path / f"{table_name}.csv"
        assert csv_path.exists()

    assert row_counts == {
        "personas": 3,
        "inscripciones": 2,
        "servicios": 1,
        "registros_acceso": 0,
        "ventas_servicios": 4,
        "pagos_pendientes": 1,
    }

    metadata_path = tmp_path / "extraction_metadata.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text())
    assert metadata["row_counts"] == row_counts
    assert "extracted_at_utc" in metadata


def test_extraction_targets_cover_every_repository_method():
    from churn_detection.data.repository import ClientDataRepository

    expected_methods = {
        name for name in vars(ClientDataRepository) if not name.startswith("_")
    }
    assert set(EXTRACTION_TARGETS.values()) == expected_methods
