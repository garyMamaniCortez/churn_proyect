import pandas as pd

from churn_detection.data.repository import PostgresClientDataRepository
from churn_detection.data import queries


def test_get_personas_reads_from_engine_with_expected_query(mocker):
    fake_df = pd.DataFrame({"persona_id": [1, 2]})
    read_sql = mocker.patch("churn_detection.data.repository.pd.read_sql", return_value=fake_df)
    fake_engine = mocker.Mock()

    repository = PostgresClientDataRepository(engine=fake_engine)
    result = repository.get_personas()

    read_sql.assert_called_once_with(queries.QUERY_PERSONAS_CLIENTES, fake_engine)
    pd.testing.assert_frame_equal(result, fake_df)


def test_get_inscripciones_reads_from_engine_with_expected_query(mocker):
    fake_df = pd.DataFrame({"inscripcion_id": [1]})
    read_sql = mocker.patch("churn_detection.data.repository.pd.read_sql", return_value=fake_df)
    fake_engine = mocker.Mock()

    repository = PostgresClientDataRepository(engine=fake_engine)
    result = repository.get_inscripciones()

    read_sql.assert_called_once_with(queries.QUERY_INSCRIPCIONES, fake_engine)
    pd.testing.assert_frame_equal(result, fake_df)


def test_personas_query_excludes_employees_and_pii_columns():
    query = queries.QUERY_PERSONAS_CLIENTES
    assert "empleados" in query
    for pii_column in ("p.ci", "p.telefono", "p.huella_digital"):
        assert pii_column not in query


def test_registros_acceso_query_filters_only_clients():
    assert "tipo_persona = 'cliente'" in queries.QUERY_REGISTROS_ACCESO
