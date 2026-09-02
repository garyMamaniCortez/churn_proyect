import pandas as pd
import pytest

from churn_detection.features import ClientSegmentationTableBuilder, SegmentationConfig

SNAPSHOT = pd.Timestamp("2026-08-30")


@pytest.fixture
def config():
    return SegmentationConfig(snapshot_date=SNAPSHOT, excluded_persona_ids=frozenset({999}))


@pytest.fixture
def servicios():
    return pd.DataFrame(
        {
            "servicio_id": [1, 2],
            "servicio_nombre": ["Sesion", "Mensual"],
            "cantidad_duracion": [1, 1],
            "tipo_duracion": ["dias", "meses"],
            "numero_ingresos": [1, 30],
        }
    )


def _personas(ids):
    return pd.DataFrame({"persona_id": ids, "fecha_nacimiento": [None] * len(ids)})


def _empty(cols):
    return pd.DataFrame({c: [] for c in cols})


def _build(config, servicios, personas, inscripciones, registros_acceso=None, ventas=None, pagos=None):
    builder = ClientSegmentationTableBuilder(config)
    return builder.build(
        personas,
        inscripciones,
        servicios,
        registros_acceso
        if registros_acceso is not None
        else _empty(["persona_id", "servicio_id", "acceso_estado", "fecha"]),
        ventas if ventas is not None else _empty(["persona_id", "venta_servicio_id", "total", "fecha"]),
        pagos if pagos is not None else _empty(["persona_id", "pago_estado"]),
    )


def test_excluded_persona_id_is_dropped_entirely(config, servicios):
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1],
            "persona_id": [999],
            "servicio_id": [2],
            "sucursal_id": [1],
            "fecha_inicio": ["2026-08-01"],
            "fecha_vencimiento": ["2026-08-31"],
            "ingresos_disponibles": [30],
        }
    )
    out = _build(config, servicios, _personas([999]), inscripciones)
    assert 999 not in out["persona_id"].values


def test_membership_usage_percentage_is_computed_and_day_passes_excluded(config, servicios):
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1, 2],
            "persona_id": [1, 1],
            "servicio_id": [2, 1],  # Mensual (30 entries, used 10) + Sesion (day-pass)
            "sucursal_id": [1, 1],
            "fecha_inicio": ["2026-08-01", "2026-08-10"],
            "fecha_vencimiento": ["2026-08-31", "2026-08-11"],
            "ingresos_disponibles": [20, 0],  # used 10 of 30 on the membership
        }
    )
    out = _build(config, servicios, _personas([1]), inscripciones)
    row = out.set_index("persona_id").loc[1]
    assert row["porcentaje_uso_membresia"] == pytest.approx(10 / 30)
    assert row["n_inscripciones_total"] == 2
    assert row["n_inscripciones_membresia"] == 1


def test_client_with_only_day_passes_has_nan_usage_percentage(config, servicios):
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1],
            "persona_id": [1],
            "servicio_id": [1],  # Sesion only, no real membership
            "sucursal_id": [1],
            "fecha_inicio": ["2026-08-10"],
            "fecha_vencimiento": ["2026-08-11"],
            "ingresos_disponibles": [0],
        }
    )
    out = _build(config, servicios, _personas([1]), inscripciones)
    row = out.set_index("persona_id").loc[1]
    assert pd.isna(row["porcentaje_uso_membresia"])


def test_schedule_features_are_computed_from_checkins(config, servicios):
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1],
            "persona_id": [1],
            "servicio_id": [2],
            "sucursal_id": [1],
            "fecha_inicio": ["2026-08-01"],
            "fecha_vencimiento": ["2026-08-31"],
            "ingresos_disponibles": [20],
        }
    )
    registros = pd.DataFrame(
        {
            "persona_id": [1, 1, 1],
            "servicio_id": [2, 2, 2],
            "acceso_estado": ["exitoso", "exitoso", "denegado"],
            "fecha": ["2026-08-10 08:00:00", "2026-08-17 08:00:00", "2026-08-20 20:00:00"],
        }
    )
    out = _build(config, servicios, _personas([1]), inscripciones, registros_acceso=registros)
    row = out.set_index("persona_id").loc[1]
    # only the 2 'exitoso' rows count; the 'denegado' one is excluded
    assert row["n_checkins_total"] == 2
    assert row["hora_promedio_checkin"] == pytest.approx(8.0)
    assert row["recencia_dias"] == (SNAPSHOT - pd.Timestamp("2026-08-17 08:00:00")).days


def test_no_estado_or_age_columns_are_present(config, servicios):
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1],
            "persona_id": [1],
            "servicio_id": [2],
            "sucursal_id": [1],
            "fecha_inicio": ["2026-08-01"],
            "fecha_vencimiento": ["2026-08-31"],
            "ingresos_disponibles": [20],
        }
    )
    out = _build(config, servicios, _personas([1]), inscripciones)
    for forbidden in ("edad", "edad_desconocida", "persona_estado", "estado_membresia", "churn_label"):
        assert forbidden not in out.columns


def test_cv_gap_visitas_needs_at_least_three_checkins(config, servicios):
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1],
            "persona_id": [1],
            "servicio_id": [2],
            "sucursal_id": [1],
            "fecha_inicio": ["2026-08-01"],
            "fecha_vencimiento": ["2026-08-31"],
            "ingresos_disponibles": [20],
        }
    )
    # Only 2 check-ins -> 1 gap -> not enough to compute a coefficient of variation.
    registros_dos = pd.DataFrame(
        {
            "persona_id": [1, 1],
            "servicio_id": [2, 2],
            "acceso_estado": ["exitoso", "exitoso"],
            "fecha": ["2026-08-05 08:00:00", "2026-08-12 08:00:00"],
        }
    )
    out = _build(config, servicios, _personas([1]), inscripciones, registros_acceso=registros_dos)
    assert pd.isna(out.set_index("persona_id").loc[1, "cv_gap_visitas"])

    # A perfectly regular client (visits every 7 days) should have cv close to 0.
    registros_regular = pd.DataFrame(
        {
            "persona_id": [1, 1, 1, 1],
            "servicio_id": [2, 2, 2, 2],
            "acceso_estado": ["exitoso"] * 4,
            "fecha": [
                "2026-08-05 08:00:00",
                "2026-08-12 08:00:00",
                "2026-08-19 08:00:00",
                "2026-08-26 08:00:00",
            ],
        }
    )
    out_regular = _build(
        config, servicios, _personas([1]), inscripciones, registros_acceso=registros_regular
    )
    assert out_regular.set_index("persona_id").loc[1, "cv_gap_visitas"] == pytest.approx(0.0, abs=1e-9)


def test_ratio_actividad_reciente_flags_acceleration_and_deceleration(config, servicios):
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1],
            "persona_id": [1],
            "servicio_id": [2],
            "sucursal_id": [1],
            "fecha_inicio": ["2026-02-01"],
            "fecha_vencimiento": ["2026-08-31"],
            "ingresos_disponibles": [20],
        }
    )
    # Steady visitor for ~6 months (weekly), which sets a baseline weekly
    # frequency, then nothing at all in the most recent 30 days -> decelerating.
    fechas_historicas = pd.date_range("2026-02-05", "2026-07-25", freq="14D")
    registros = pd.DataFrame(
        {
            "persona_id": [1] * len(fechas_historicas),
            "servicio_id": [2] * len(fechas_historicas),
            "acceso_estado": ["exitoso"] * len(fechas_historicas),
            "fecha": [d.strftime("%Y-%m-%d 08:00:00") for d in fechas_historicas],
        }
    )
    out = _build(config, servicios, _personas([1]), inscripciones, registros_acceso=registros)
    row = out.set_index("persona_id").loc[1]
    assert row["n_checkins_ultimos_30d"] == 0
    assert row["ratio_actividad_reciente"] < 1


def test_monto_gastado_ultimos_90d_only_counts_recent_sales(config, servicios):
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1],
            "persona_id": [1],
            "servicio_id": [2],
            "sucursal_id": [1],
            "fecha_inicio": ["2026-01-01"],
            "fecha_vencimiento": ["2026-01-31"],
            "ingresos_disponibles": [20],
        }
    )
    ventas = pd.DataFrame(
        {
            "persona_id": [1, 1],
            "venta_servicio_id": [1, 2],
            "total": [100.0, 50.0],
            "fecha": ["2026-01-01 10:00:00", "2026-08-15 10:00:00"],  # one old, one recent
        }
    )
    out = _build(config, servicios, _personas([1]), inscripciones, ventas=ventas)
    row = out.set_index("persona_id").loc[1]
    assert row["monto_total_gastado"] == pytest.approx(150.0)
    assert row["monto_gastado_ultimos_90d"] == pytest.approx(50.0)


def test_membership_usage_ignores_inscripciones_before_reliable_tracking_cutoff(config, servicios):
    # Started well before the access-logging module worked reliably (2026-02-01).
    # ingresos_disponibles == numero_ingresos here looks like "never attended", but
    # that's the broken tracker, not the client -- must NOT count as 0% usage.
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1],
            "persona_id": [1],
            "servicio_id": [2],
            "sucursal_id": [1],
            "fecha_inicio": ["2025-11-01"],
            "fecha_vencimiento": ["2025-12-01"],
            "ingresos_disponibles": [30],  # looks fully unused
        }
    )
    out = _build(config, servicios, _personas([1]), inscripciones)
    row = out.set_index("persona_id").loc[1]
    assert pd.isna(row["porcentaje_uso_membresia"])


def test_checkins_before_reliable_tracking_cutoff_are_excluded(config, servicios):
    inscripciones = pd.DataFrame(
        {
            "inscripcion_id": [1],
            "persona_id": [1],
            "servicio_id": [2],
            "sucursal_id": [1],
            "fecha_inicio": ["2025-11-01"],
            "fecha_vencimiento": ["2025-12-01"],
            "ingresos_disponibles": [20],
        }
    )
    registros = pd.DataFrame(
        {
            "persona_id": [1],
            "servicio_id": [2],
            "acceso_estado": ["exitoso"],
            "fecha": ["2025-11-15 08:00:00"],  # before the reliable-tracking cutoff
        }
    )
    out = _build(config, servicios, _personas([1]), inscripciones, registros_acceso=registros)
    row = out.set_index("persona_id").loc[1]
    assert row["n_checkins_total"] == 0
    assert pd.isna(row["recencia_dias"])
