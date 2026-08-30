"""Raw SQL used by `PostgresClientDataRepository`.

Kept as named constants, separate from the repository class, so the SQL can be
reviewed, unit-tested (e.g. "does the query still reference existing columns"),
and reused without touching Python control flow (SRP / easier code review by a DBA).

Privacy / governance note
--------------------------
`personas.ci`, `personas.telefono` and `personas.huella_digital` (a fingerprint
template) are intentionally NEVER selected here. They are direct/biometric
identifiers with no predictive value for churn or segmentation, and pulling them
into flat analytical CSVs that later get versioned with DVC would create an
unnecessary PII/biometric-data exposure surface. `persona_id` is enough to join
everything downstream.
"""

# Only customers: personas with no matching row in `empleados`.
QUERY_PERSONAS_CLIENTES = """
SELECT
    p.id AS persona_id,
    p.fecha_nacimiento,
    p.estado AS persona_estado
FROM personas p
LEFT JOIN empleados e ON e.persona_id = p.id
WHERE e.id IS NULL
"""

QUERY_INSCRIPCIONES = """
SELECT
    i.id AS inscripcion_id,
    i.persona_id,
    i.servicio_id,
    i.sucursal_id,
    i.fecha_inicio,
    i.fecha_vencimiento,
    i.ingresos_disponibles,
    i.estado AS inscripcion_estado
FROM inscripciones i
"""

QUERY_SERVICIOS = """
SELECT
    s.id AS servicio_id,
    s.nombre AS servicio_nombre,
    s.precio,
    s.numero_ingresos,
    s.tipo_duracion,
    s.cantidad_duracion,
    s.multisucursal,
    s.estado AS servicio_estado
FROM servicios s
"""

# Only client check-ins (tipo_persona = 'cliente'), not staff access records.
QUERY_REGISTROS_ACCESO = """
SELECT
    r.id AS registro_id,
    r.persona_id,
    r.servicio_id,
    r.estado AS acceso_estado,
    r.sucursal_id,
    r.fecha
FROM registros_acceso r
WHERE r.tipo_persona = 'cliente'
"""

QUERY_VENTAS_SERVICIOS = """
SELECT
    v.id AS venta_servicio_id,
    v.persona_id,
    v.sucursal_id,
    v.subtotal,
    v.descuento,
    v.total,
    v.forma_pago,
    v.fecha
FROM ventas_servicios v
"""

QUERY_PAGOS_PENDIENTES = """
SELECT
    pp.id AS pago_pendiente_id,
    pp.persona_id,
    pp.venta_servicio_id,
    pp.monto_total,
    pp.monto_pagado,
    pp.monto_pendiente,
    pp.fecha_inscripcion,
    pp.fecha_ultima_actualizacion,
    pp.estado AS pago_estado
FROM pagos_pendientes pp
"""
