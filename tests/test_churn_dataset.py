import numpy as np
import pandas as pd
import pytest

from churn_detection.churn_dataset import ChurnCycleDatasetBuilder, ChurnDatasetConfig

SNAPSHOT = pd.Timestamp("2026-08-30")


@pytest.fixture
def config():
    return ChurnDatasetConfig(grace_days=30, snapshot_date=SNAPSHOT, excluded_persona_ids=frozenset())


@pytest.fixture
def servicios():
    return pd.DataFrame(
        {
            "servicio_id": [1, 2],
            "cantidad_duracion": [1, 1],
            "tipo_duracion": ["dias", "meses"],
            "numero_ingresos": [1, 30],
        }
    )


def _personas(ids):
    return pd.DataFrame({"persona_id": ids})


def _empty_reg():
    return pd.DataFrame(columns=["persona_id", "servicio_id", "acceso_estado", "fecha"])


def _empty_ventas():
    return pd.DataFrame(columns=["persona_id", "venta_servicio_id", "total", "fecha"])


def test_renewal_within_grace_labels_the_earlier_cycle_as_renovado(config, servicios):
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1, 2],
            "persona_id": [1, 1],
            "servicio_id": [2, 2],
            "sucursal_id": [1, 1],
            "fecha_inicio": ["2026-01-01", "2026-02-05"],
            "fecha_vencimiento": ["2026-01-31", "2026-03-05"],
            "ingresos_disponibles": [5, 20],
        }
    )
    builder = ChurnCycleDatasetBuilder(config)
    out = builder.build(_personas([1]), inscripciones, servicios, _empty_reg(), _empty_ventas())

    first_cycle = out[out["inscripcion_id"] == 1].iloc[0]
    assert first_cycle["estado_ciclo"] == "renovado"
    assert first_cycle["churn_label"] == 0.0


def test_last_cycle_past_grace_with_no_renewal_is_churned(config, servicios):
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1],
            "persona_id": [1],
            "servicio_id": [2],
            "sucursal_id": [1],
            "fecha_inicio": ["2026-01-01"],
            "fecha_vencimiento": ["2026-01-31"],  # lapsed ~7 months ago, well past grace
            "ingresos_disponibles": [10],
        }
    )
    builder = ChurnCycleDatasetBuilder(config)
    out = builder.build(_personas([1]), inscripciones, servicios, _empty_reg(), _empty_ventas())

    row = out.iloc[0]
    assert row["estado_ciclo"] == "churned"
    assert row["churn_label"] == 1.0


def test_last_cycle_still_within_grace_is_censored_not_churned(config, servicios):
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1],
            "persona_id": [1],
            "servicio_id": [2],
            "sucursal_id": [1],
            "fecha_inicio": ["2026-07-20"],
            "fecha_vencimiento": ["2026-08-19"],  # lapsed 11 days ago, grace is 30
            "ingresos_disponibles": [10],
        }
    )
    builder = ChurnCycleDatasetBuilder(config)
    out = builder.build(_personas([1]), inscripciones, servicios, _empty_reg(), _empty_ventas())

    row = out.iloc[0]
    assert row["estado_ciclo"] == "censurado"
    assert pd.isna(row["churn_label"])


def test_a_day_pass_between_cycles_does_not_count_as_the_renewal(config, servicios):
    # Membership ends, client buys a day-pass a week later (not a renewal),
    # then genuinely renews their membership within the grace window.
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1, 2, 3],
            "persona_id": [1, 1, 1],
            "servicio_id": [2, 1, 2],  # Mensual, Sesion (day-pass), Mensual
            "sucursal_id": [1, 1, 1],
            "fecha_inicio": ["2026-01-01", "2026-02-05", "2026-02-20"],
            "fecha_vencimiento": ["2026-01-31", "2026-02-06", "2026-03-20"],
            "ingresos_disponibles": [5, 0, 20],
        }
    )
    builder = ChurnCycleDatasetBuilder(config)
    out = builder.build(_personas([1]), inscripciones, servicios, _empty_reg(), _empty_ventas())

    # Only 2 membership cycles should exist (the day-pass isn't one at all).
    assert len(out) == 2
    first_cycle = out[out["inscripcion_id"] == 1].iloc[0]
    assert first_cycle["estado_ciclo"] == "renovado"


def test_features_never_use_data_after_the_cycles_own_cutoff(config, servicios):
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1],
            "persona_id": [1],
            "servicio_id": [2],
            "sucursal_id": [1],
            "fecha_inicio": ["2026-06-01"],
            "fecha_vencimiento": ["2026-07-01"],
            "ingresos_disponibles": [10],
        }
    )
    registros = pd.DataFrame(
        {
            "persona_id": [1, 1],
            "servicio_id": [2, 2],
            "acceso_estado": ["exitoso", "exitoso"],
            "fecha": ["2026-06-15 08:00:00", "2026-08-01 08:00:00"],  # one before, one AFTER cutoff
        }
    )
    ventas = pd.DataFrame(
        {
            "persona_id": [1, 1],
            "venta_servicio_id": [1, 2],
            "total": [50.0, 999.0],
            "fecha": ["2026-06-15 08:00:00", "2026-08-01 08:00:00"],  # one before, one AFTER cutoff
        }
    )
    builder = ChurnCycleDatasetBuilder(config)
    out = builder.build(_personas([1]), inscripciones, servicios, registros, ventas)

    row = out.iloc[0]
    assert row["n_checkins_total"] == 1  # only the pre-cutoff check-in
    assert row["monto_total_gastado"] == pytest.approx(50.0)  # only the pre-cutoff sale


def test_porcentaje_uso_membresia_includes_the_cycle_being_evaluated(config, servicios):
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1],
            "persona_id": [1],
            "servicio_id": [2],
            "sucursal_id": [1],
            "fecha_inicio": ["2026-01-01"],
            "fecha_vencimiento": ["2026-01-31"],
            "ingresos_disponibles": [0],  # used all 30 entries by the time it expired
        }
    )
    builder = ChurnCycleDatasetBuilder(config)
    out = builder.build(_personas([1]), inscripciones, servicios, _empty_reg(), _empty_ventas())

    row = out.iloc[0]
    assert row["porcentaje_uso_membresia"] == pytest.approx(1.0)


def test_excluded_persona_id_produces_no_rows(config, servicios):
    config_excl = ChurnDatasetConfig(
        grace_days=30, snapshot_date=SNAPSHOT, excluded_persona_ids=frozenset({1})
    )
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1],
            "persona_id": [1],
            "servicio_id": [2],
            "sucursal_id": [1],
            "fecha_inicio": ["2026-01-01"],
            "fecha_vencimiento": ["2026-01-31"],
            "ingresos_disponibles": [10],
        }
    )
    builder = ChurnCycleDatasetBuilder(config_excl)
    out = builder.build(_personas([1]), inscripciones, servicios, _empty_reg(), _empty_ventas())
    assert len(out) == 0
