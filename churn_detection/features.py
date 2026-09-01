"""Build the client-level analytical table used for customer segmentation.

Scope of this module (as of this iteration)
---------------------------------------------
This builds ONLY the segmentation dataset: one row per client, all features
computed as of a single global `snapshot_date` (defaults to "today"). That's
fine for segmentation — it's a descriptive snapshot of "what does the customer
base look like right now" — but it is NOT what the churn model should train on.

The churn training dataset needs a *different* snapshot semantics: features
computed as of a cutoff *per client*, before their outcome is known, otherwise
the model leaks the future into the predictors (e.g. a client who lapsed in
February would trivially show `recencia_dias` in the hundreds if measured
"today", which doesn't help predict who's ABOUT to churn). That per-client-cutoff
builder is a deliberately separate piece of work, not implemented here yet.

Design notes (SOLID)
---------------------
`ClientSegmentationTableBuilder.build()` takes plain, already-loaded DataFrames
— it never touches disk or a database itself (Dependency Inversion / SRP:
reading raw CSVs is dataset.py's job). That also makes it unit-testable with
small hand-built fixtures, no DB or real files required.

Columns intentionally NOT included, and why
----------------------------------------------
- No `persona_estado`, `inscripcion_estado`, `servicio_estado`, `pago_estado`,
  or any other raw DB "estado"/status column: per instruction, these carry
  little independent signal here (e.g. `persona_estado` is 1 for only 14 of
  4,435 clients) and mixing operational status flags into descriptive
  segmentation features tends to create false structure.
- No age / `fecha_nacimiento`-derived feature: 66% of clients have no
  `fecha_nacimiento` on file, which is above the 40% missingness threshold
  this project uses to decide "drop the column entirely" rather than impute
  and risk silently manufacturing signal for two-thirds of the base.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from churn_detection.config import RELIABLE_ACCESS_TRACKING_SINCE


@dataclass(frozen=True)
class SegmentationConfig:
    """Parameters controlling how the segmentation snapshot is computed."""

    snapshot_date: pd.Timestamp
    excluded_persona_ids: frozenset[int] = field(default_factory=frozenset)


class ClientSegmentationTableBuilder:
    """Builds one row per client with descriptive features for clustering.

    All features are computed as of `config.snapshot_date` for every client
    alike — appropriate for a "how does today's customer base look" view, not
    for supervised training (see module docstring).
    """

    def __init__(self, config: SegmentationConfig) -> None:
        self._config = config

    def build(
        self,
        personas: pd.DataFrame,
        inscripciones: pd.DataFrame,
        servicios: pd.DataFrame,
        registros_acceso: pd.DataFrame,
        ventas_servicios: pd.DataFrame,
        pagos_pendientes: pd.DataFrame,
    ) -> pd.DataFrame:
        personas = self._exclude_bad_personas(personas)
        inscripciones = inscripciones.merge(
            self._flag_membership_services(servicios), on="servicio_id", how="left"
        )

        lifecycle = self._build_lifecycle_features(inscripciones)
        usage = self._build_membership_usage_features(inscripciones)
        engagement = self._build_engagement_features(registros_acceso)
        monetary = self._build_monetary_features(ventas_servicios)
        payments = self._build_payment_features(pagos_pendientes)

        master = (
            personas[["persona_id"]]
            .merge(lifecycle, on="persona_id", how="left")
            .merge(usage, on="persona_id", how="left")
            .merge(engagement, on="persona_id", how="left")
            .merge(monetary, on="persona_id", how="left")
            .merge(payments, on="persona_id", how="left")
        )

        count_cols = [
            "n_inscripciones_total",
            "n_inscripciones_membresia",
            "n_servicios_distintos",
            "n_sucursales_distintas",
            "n_checkins_total",
            "n_checkins_ultimos_30d",
            "n_checkins_ultimos_90d",
        ]
        master[count_cols] = master[count_cols].fillna(0).astype(int)
        master["es_multisucursal"] = master["n_sucursales_distintas"] > 1

        money_cols = ["monto_total_gastado", "monto_promedio_venta"]
        master[money_cols] = master[money_cols].fillna(0.0)

        master["tiene_pago_pendiente"] = master["tiene_pago_pendiente"].fillna(False)

        # Left as NaN (NOT imputed here) when undefined for a client, e.g.
        # `hora_promedio_checkin` for someone with no check-ins in the reliable
        # window, or `porcentaje_uso_membresia` for someone with no membership-type
        # inscripcion that started on/after RELIABLE_ACCESS_TRACKING_SINCE. Both are
        # under the 40% missingness threshold (17.5% and 10.9% respectively as of
        # the 2026-08-30 extraction) so the columns stay; how to handle the
        # remaining NaNs (impute vs. flag vs. drop those rows) is a modeling-stage
        # decision, not a feature-engineering one.
        return master

    # -- private helpers -------------------------------------------------

    def _exclude_bad_personas(self, personas: pd.DataFrame) -> pd.DataFrame:
        return personas[~personas["persona_id"].isin(self._config.excluded_persona_ids)].copy()

    @staticmethod
    def _flag_membership_services(servicios: pd.DataFrame) -> pd.DataFrame:
        is_day_pass = (servicios["cantidad_duracion"] == 1) & (
            servicios["tipo_duracion"] == "dias"
        )
        return servicios.assign(es_membresia=~is_day_pass)[
            ["servicio_id", "es_membresia", "numero_ingresos"]
        ]

    def _build_lifecycle_features(self, inscripciones: pd.DataFrame) -> pd.DataFrame:
        insc = inscripciones.copy()
        insc["fecha_inicio"] = pd.to_datetime(insc["fecha_inicio"])

        out = insc.groupby("persona_id").agg(
            n_inscripciones_total=("inscripcion_id", "count"),
            n_inscripciones_membresia=("es_membresia", "sum"),
            n_servicios_distintos=("servicio_id", "nunique"),
            n_sucursales_distintas=("sucursal_id", "nunique"),
            fecha_primera_inscripcion=("fecha_inicio", "min"),
        )
        out["tenure_dias"] = (
            self._config.snapshot_date - out["fecha_primera_inscripcion"]
        ).dt.days
        return out.reset_index()[
            [
                "persona_id",
                "n_inscripciones_total",
                "n_inscripciones_membresia",
                "n_servicios_distintos",
                "n_sucursales_distintas",
                "tenure_dias",
            ]
        ]

    @staticmethod
    def _build_membership_usage_features(inscripciones: pd.DataFrame) -> pd.DataFrame:
        """% of paid-for entries actually used, averaged across membership inscripciones.

        Per inscripcion: `1 - ingresos_disponibles / numero_ingresos`. Clients whose
        only inscripciones are day-passes (excluded via `es_membresia`) get NaN here,
        not 0 -- "never had a membership" is a different thing from "had one and
        never showed up", and collapsing them to the same value would blur that.

        Inscripciones that started before `RELIABLE_ACCESS_TRACKING_SINCE` are
        EXCLUDED from this calculation, not treated as "0% used": the access-logging
        module that decrements `ingresos_disponibles` was unreliable before that
        date (confirmed by the client + by the data -- see config.py), so a low
        value there reflects a broken tracker, not a disengaged customer. A client
        whose only membership inscripciones are all that old gets NaN.
        """

        membership = inscripciones[inscripciones["es_membresia"].fillna(False)].copy()
        membership["fecha_inicio"] = pd.to_datetime(membership["fecha_inicio"])
        membership = membership[membership["fecha_inicio"] >= RELIABLE_ACCESS_TRACKING_SINCE]

        # 2 rows out of 10,442 in the 2026-08-30 extract have ingresos_disponibles
        # slightly above numero_ingresos (data-entry inconsistency upstream) -- clip
        # so usage stays in a sane [0, 1] range instead of going slightly negative.
        capped_disponibles = membership["ingresos_disponibles"].clip(
            upper=membership["numero_ingresos"]
        )
        membership["pct_uso"] = 1 - (capped_disponibles / membership["numero_ingresos"])
        out = membership.groupby("persona_id")["pct_uso"].mean()
        return out.reset_index().rename(columns={"pct_uso": "porcentaje_uso_membresia"})

    def _build_engagement_features(self, registros_acceso: pd.DataFrame) -> pd.DataFrame:

        cfg = self._config
        reg = registros_acceso.copy()
        reg["fecha"] = pd.to_datetime(reg["fecha"])
        # Access records before this date measure a broken/absent logging module,
        # not real attendance (see config.RELIABLE_ACCESS_TRACKING_SINCE) -- drop
        # them from every schedule/engagement feature rather than let them read as
        # "this client never visits".
        reg = reg[reg["fecha"] >= RELIABLE_ACCESS_TRACKING_SINCE]
        reg = reg[reg["acceso_estado"] == "exitoso"].copy()

        expected_cols = [
            "persona_id",
            "n_checkins_total",
            "n_checkins_ultimos_30d",
            "n_checkins_ultimos_90d",
            "recencia_dias",
            "frecuencia_visitas_semanal",
            "hora_promedio_checkin",
            "hora_checkin_std",
            "pct_visitas_fin_de_semana",
            "dia_semana_mas_frecuente",
        ]
        if reg.empty:
            # groupby/agg on an empty frame can't infer real dtypes (e.g. a
            # datetime column collapses to an ambiguous empty array), which
            # breaks later arithmetic/comparisons -- short-circuit instead.
            # persona_id must stay int64 (not the default object dtype for an
            # empty column) so the later merge on that key doesn't blow up.
            empty = pd.DataFrame(columns=expected_cols)
            return empty.astype({"persona_id": "int64"})

        total = reg.groupby("persona_id").size().rename("n_checkins_total")

        last_30 = reg[reg["fecha"] >= cfg.snapshot_date - pd.Timedelta(days=30)]
        c30 = last_30.groupby("persona_id").size().rename("n_checkins_ultimos_30d")

        last_90 = reg[reg["fecha"] >= cfg.snapshot_date - pd.Timedelta(days=90)]
        c90 = last_90.groupby("persona_id").size().rename("n_checkins_ultimos_90d")

        last_visit = reg.groupby("persona_id")["fecha"].max().rename("ultimo_checkin")

        reg["hora"] = reg["fecha"].dt.hour + reg["fecha"].dt.minute / 60
        hora_media = reg.groupby("persona_id")["hora"].mean().rename("hora_promedio_checkin")
        hora_std = reg.groupby("persona_id")["hora"].std().rename("hora_checkin_std")

        reg["es_fin_de_semana"] = reg["fecha"].dt.dayofweek >= 5
        pct_finde = (
            reg.groupby("persona_id")["es_fin_de_semana"]
            .mean()
            .rename("pct_visitas_fin_de_semana")
        )

        dia_frecuente = (
            reg.groupby("persona_id")["fecha"]
            .agg(lambda s: s.dt.day_name().mode().iloc[0])
            .rename("dia_semana_mas_frecuente")
        )

        # Span (days) between a client's first and last successful check-in,
        # floored at 7 so a single busy week doesn't blow up the weekly-frequency
        # ratio below. Deliberately based on check-in activity, not on
        # `tenure_dias` (which comes from inscripciones) -- this measures how
        # spread out their actual VISITS are, not how long ago they first signed up.
        visit_span_days = (
            reg.groupby("persona_id")["fecha"]
            .agg(lambda s: (s.max() - s.min()).days)
            .clip(lower=7)
            .rename("_visit_span_dias")
        )

        out = pd.concat(
            [
                total,
                c30,
                c90,
                last_visit,
                hora_media,
                hora_std,
                pct_finde,
                dia_frecuente,
                visit_span_days,
            ],
            axis=1,
        ).reset_index()
        out = out.rename(columns={"index": "persona_id"})

        out["recencia_dias"] = (cfg.snapshot_date - out["ultimo_checkin"]).dt.days
        out["frecuencia_visitas_semanal"] = out["n_checkins_total"] / (out["_visit_span_dias"] / 7)

        return out[expected_cols]

    @staticmethod
    def _build_monetary_features(ventas_servicios: pd.DataFrame) -> pd.DataFrame:
        # n_ventas deliberately NOT included: EDA showed it correlates 0.99 with
        # n_inscripciones_total (each service sale maps almost 1:1 to an
        # inscripcion), so keeping both would double-weight the same signal in
        # clustering. n_inscripciones_total is already in the lifecycle features.
        out = ventas_servicios.groupby("persona_id").agg(
            monto_total_gastado=("total", "sum"),
            monto_promedio_venta=("total", "mean"),
        )
        return out.reset_index()

    @staticmethod
    def _build_payment_features(pagos_pendientes: pd.DataFrame) -> pd.DataFrame:
        pendientes = pagos_pendientes[pagos_pendientes["pago_estado"] == "pendiente"]
        ids_con_pago_pendiente = set(pendientes["persona_id"].unique())
        out = pd.DataFrame({"persona_id": list(ids_con_pago_pendiente)})
        out["tiene_pago_pendiente"] = True
        return out


# --- CLI: orchestrates reading data/raw/*.csv and writing data/processed/clientes_segmentacion.csv ---

if __name__ == "__main__":
    import typer

    from churn_detection.config import EXCLUDED_PERSONA_IDS, PROCESSED_DATA_DIR, RAW_DATA_DIR

    app = typer.Typer(help="Build the client segmentation table from extracted raw CSVs.")

    @app.command()
    def build_segmentation_table(
        raw_dir: str = str(RAW_DATA_DIR),
        output_path: str = str(PROCESSED_DATA_DIR / "clientes_segmentacion.csv"),
    ) -> None:
        from pathlib import Path

        from loguru import logger

        raw_dir_path = Path(raw_dir)
        personas = pd.read_csv(raw_dir_path / "personas.csv")
        inscripciones = pd.read_csv(raw_dir_path / "inscripciones.csv")
        servicios = pd.read_csv(raw_dir_path / "servicios.csv")
        registros_acceso = pd.read_csv(raw_dir_path / "registros_acceso.csv")
        ventas_servicios = pd.read_csv(raw_dir_path / "ventas_servicios.csv")
        pagos_pendientes = pd.read_csv(raw_dir_path / "pagos_pendientes.csv")

        config = SegmentationConfig(
            snapshot_date=pd.Timestamp.now().normalize(),
            excluded_persona_ids=EXCLUDED_PERSONA_IDS,
        )
        builder = ClientSegmentationTableBuilder(config)
        table = builder.build(
            personas,
            inscripciones,
            servicios,
            registros_acceso,
            ventas_servicios,
            pagos_pendientes,
        )

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)
        logger.success(f"Wrote {len(table)} client rows to {out_path}")
        null_pct = (table.isna().mean() * 100).round(1)
        logger.info(f"% nulos por columna:\n{null_pct[null_pct > 0]}")

    app()
