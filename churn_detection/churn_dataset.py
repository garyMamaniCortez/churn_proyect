"""Churn TRAINING dataset: one row per membership cycle, not per client.

Why per-cycle, not per-client
-------------------------------
Using each client's single most-recent membership as the observation would be
circular: if they renewed, the renewal itself becomes their "most recent"
membership, so the one we're checking can never resolve to "renewed" under
that definition -- only "churned" or "still open". A model trained that way
would never see a single positive (retained) example.

Instead, each MEMBERSHIP-TYPE inscripcion a client ever had is its own
observation: "given that this cycle ended on `fecha_vencimiento`, did the
client start a new membership cycle within `grace_days` afterward?" A client
with 3 membership inscripciones over time contributes 3 rows (each can
resolve independently to renewed or churned), except their very last cycle,
which is only labeled once enough time has passed to know the answer -- see
"Censoring" below.

No leakage: every feature for a given row is computed using only
registros_acceso / ventas_servicios / inscripciones data with a date at or
before that row's OWN `fecha_vencimiento` (its cutoff). Data from during or
after the next cycle is never used to predict this cycle's outcome.

Censoring
----------
A client's chronologically LAST membership cycle has no "next cycle" to check
yet. Three outcomes are possible for it:
  - `churn_label = 0` ("renovado"): impossible for the last cycle by
    definition (see above) -- this is exactly why per-cycle rows are needed.
  - `churn_label = 1` ("churned"): the grace window has already fully
    elapsed (`snapshot_date > fecha_vencimiento + grace_days`) with no
    renewal -- a resolved outcome.
  - `churn_label = NaN` ("censurado"): still within the grace window as of
    `snapshot_date` -- outcome not yet known, excluded from training. These
    rows are exactly the "clients to score" set for live inference later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from churn_detection.config import RELIABLE_ACCESS_TRACKING_SINCE
from churn_detection.membership_utils import flag_membership_services


@dataclass(frozen=True)
class ChurnDatasetConfig:
    grace_days: int
    snapshot_date: pd.Timestamp
    excluded_persona_ids: frozenset[int] = field(default_factory=frozenset)


class ChurnCycleDatasetBuilder:
    """Builds the per-membership-cycle churn training dataset."""

    def __init__(self, config: ChurnDatasetConfig) -> None:
        self._config = config

    def build(
        self,
        personas: pd.DataFrame,
        inscripciones: pd.DataFrame,
        servicios: pd.DataFrame,
        registros_acceso: pd.DataFrame,
        ventas_servicios: pd.DataFrame,
    ) -> pd.DataFrame:
        cycles = self._build_cycles_with_labels(personas, inscripciones, servicios)

        reg_by_client = self._group_reliable_checkins(registros_acceso)
        ventas_by_client = self._group_ventas(ventas_servicios)
        cycles_by_client = {pid: g for pid, g in cycles.groupby("persona_id")}

        feature_rows = [
            self._compute_point_in_time_features(
                row,
                reg_by_client.get(row.persona_id),
                ventas_by_client.get(row.persona_id),
                cycles_by_client.get(row.persona_id),
            )
            for row in cycles.itertuples()
        ]
        features_df = pd.DataFrame(feature_rows)

        out = pd.concat(
            [
                cycles[
                    [
                        "persona_id",
                        "inscripcion_id",
                        "fecha_inicio",
                        "fecha_vencimiento",
                        "estado_ciclo",
                        "churn_label",
                    ]
                ].reset_index(drop=True),
                features_df,
            ],
            axis=1,
        )
        return out

    # -- label construction -------------------------------------------------

    def _build_cycles_with_labels(
        self, personas: pd.DataFrame, inscripciones: pd.DataFrame, servicios: pd.DataFrame
    ) -> pd.DataFrame:
        cfg = self._config
        valid_ids = set(
            personas[~personas["persona_id"].isin(cfg.excluded_persona_ids)]["persona_id"]
        )

        insc = inscripciones.merge(
            flag_membership_services(servicios), on="servicio_id", how="left"
        )
        insc = insc[insc["persona_id"].isin(valid_ids) & insc["es_membresia"].fillna(False)].copy()
        insc["fecha_inicio"] = pd.to_datetime(insc["fecha_inicio"])
        insc["fecha_vencimiento"] = pd.to_datetime(insc["fecha_vencimiento"])
        insc = insc.sort_values(["persona_id", "fecha_inicio"]).reset_index(drop=True)

        insc["siguiente_fecha_inicio"] = insc.groupby("persona_id")["fecha_inicio"].shift(-1)
        insc["fecha_limite_renovacion"] = insc["fecha_vencimiento"] + pd.Timedelta(
            days=cfg.grace_days
        )

        if insc.empty:
            insc["estado_ciclo"] = pd.Series(dtype="object")
            insc["churn_label"] = pd.Series(dtype="float64")
            return insc

        def _label(row: pd.Series) -> pd.Series:
            es_ultimo_ciclo = pd.isna(row["siguiente_fecha_inicio"])
            if es_ultimo_ciclo:
                if cfg.snapshot_date <= row["fecha_limite_renovacion"]:
                    return pd.Series({"estado_ciclo": "censurado", "churn_label": np.nan})
                return pd.Series({"estado_ciclo": "churned", "churn_label": 1.0})
            if row["siguiente_fecha_inicio"] <= row["fecha_limite_renovacion"]:
                return pd.Series({"estado_ciclo": "renovado", "churn_label": 0.0})
            return pd.Series({"estado_ciclo": "churned", "churn_label": 1.0})

        insc[["estado_ciclo", "churn_label"]] = insc.apply(_label, axis=1)
        return insc

    # -- point-in-time feature computation -----------------------------------

    @staticmethod
    def _group_reliable_checkins(registros_acceso: pd.DataFrame) -> dict[int, pd.DataFrame]:
        reg = registros_acceso.copy()
        reg["fecha"] = pd.to_datetime(reg["fecha"])
        reg = reg[
            (reg["acceso_estado"] == "exitoso") & (reg["fecha"] >= RELIABLE_ACCESS_TRACKING_SINCE)
        ]
        return {pid: g.sort_values("fecha") for pid, g in reg.groupby("persona_id")}

    @staticmethod
    def _group_ventas(ventas_servicios: pd.DataFrame) -> dict[int, pd.DataFrame]:
        ventas = ventas_servicios.copy()
        ventas["fecha"] = pd.to_datetime(ventas["fecha"])
        return {pid: g.sort_values("fecha") for pid, g in ventas.groupby("persona_id")}

    def _compute_point_in_time_features(
        self,
        row,
        reg_hist: pd.DataFrame | None,
        ventas_hist: pd.DataFrame | None,
        cycles_hist: pd.DataFrame | None,
    ) -> dict:
        cutoff = row.fecha_vencimiento
        feats: dict = {}

        reg_upto = (
            reg_hist[reg_hist["fecha"] <= cutoff]
            if reg_hist is not None
            else pd.DataFrame(columns=["fecha"])
        )
        n_checkins = len(reg_upto)
        feats["n_checkins_total"] = n_checkins
        if n_checkins == 0:
            feats["recencia_dias"] = np.nan
            feats["frecuencia_visitas_semanal"] = np.nan
            feats["hora_promedio_checkin"] = np.nan
            feats["pct_visitas_fin_de_semana"] = np.nan
            feats["cv_gap_visitas"] = np.nan
            feats["ratio_actividad_reciente"] = np.nan
        else:
            ultimo_checkin = reg_upto["fecha"].max()
            feats["recencia_dias"] = (cutoff - ultimo_checkin).days
            span_dias = max((ultimo_checkin - reg_upto["fecha"].min()).days, 7)
            frecuencia = n_checkins / (span_dias / 7)
            feats["frecuencia_visitas_semanal"] = frecuencia
            horas = reg_upto["fecha"].dt.hour + reg_upto["fecha"].dt.minute / 60
            feats["hora_promedio_checkin"] = horas.mean()
            feats["pct_visitas_fin_de_semana"] = (reg_upto["fecha"].dt.dayofweek >= 5).mean()

            if n_checkins >= 3:
                gaps = reg_upto["fecha"].sort_values().diff().dt.days.dropna()
                mean_gap = gaps.mean()
                feats["cv_gap_visitas"] = (gaps.std() / mean_gap) if mean_gap > 0 else np.nan
            else:
                feats["cv_gap_visitas"] = np.nan

            ultimos_30d = reg_upto[reg_upto["fecha"] > cutoff - pd.Timedelta(days=30)]
            expected_30d = frecuencia * (30 / 7)
            feats["ratio_actividad_reciente"] = (
                len(ultimos_30d) / expected_30d if expected_30d > 0 else np.nan
            )

        ventas_upto = (
            ventas_hist[ventas_hist["fecha"] <= cutoff]
            if ventas_hist is not None
            else pd.DataFrame(columns=["total", "fecha"])
        )
        feats["monto_total_gastado"] = ventas_upto["total"].sum() if len(ventas_upto) else 0.0
        feats["monto_promedio_venta"] = ventas_upto["total"].mean() if len(ventas_upto) else 0.0
        ultimos_90d_ventas = ventas_upto[ventas_upto["fecha"] > cutoff - pd.Timedelta(days=90)]
        feats["monto_gastado_ultimos_90d"] = (
            ultimos_90d_ventas["total"].sum() if len(ultimos_90d_ventas) else 0.0
        )

        ciclos_upto = (
            cycles_hist[cycles_hist["fecha_inicio"] <= cutoff]
            if cycles_hist is not None
            else pd.DataFrame()
        )
        feats["n_inscripciones_total"] = len(ciclos_upto)
        feats["tenure_dias"] = (
            (cutoff - ciclos_upto["fecha_inicio"].min()).days if len(ciclos_upto) else 0
        )
        capped = ciclos_upto["ingresos_disponibles"].clip(upper=ciclos_upto["numero_ingresos"])
        pct_uso = 1 - (capped / ciclos_upto["numero_ingresos"])
        feats["porcentaje_uso_membresia"] = pct_uso.mean() if len(ciclos_upto) else np.nan

        return feats


# --- CLI ---

if __name__ == "__main__":
    from pathlib import Path

    import typer

    from churn_detection.config import (
        EXCLUDED_PERSONA_IDS,
        PROCESSED_DATA_DIR,
        RAW_DATA_DIR,
        ChurnSettings,
    )

    app = typer.Typer(help="Build the per-membership-cycle churn training dataset.")

    @app.command()
    def build_churn_dataset(
        raw_dir: Path = RAW_DATA_DIR,
        output_path: Path = PROCESSED_DATA_DIR / "churn_ciclos.csv",
    ) -> None:
        from loguru import logger

        personas = pd.read_csv(raw_dir / "personas.csv")
        inscripciones = pd.read_csv(raw_dir / "inscripciones.csv")
        servicios = pd.read_csv(raw_dir / "servicios.csv")
        registros_acceso = pd.read_csv(raw_dir / "registros_acceso.csv")
        ventas_servicios = pd.read_csv(raw_dir / "ventas_servicios.csv")

        churn_settings = ChurnSettings()
        config = ChurnDatasetConfig(
            grace_days=churn_settings.churn_grace_days,
            snapshot_date=pd.Timestamp.now().normalize(),
            excluded_persona_ids=EXCLUDED_PERSONA_IDS,
        )
        builder = ChurnCycleDatasetBuilder(config)
        out = builder.build(personas, inscripciones, servicios, registros_acceso, ventas_servicios)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        logger.success(f"Wrote {len(out)} membership-cycle rows to {output_path}")
        logger.info(f"estado_ciclo distribution:\n{out['estado_ciclo'].value_counts()}")
        labeled = out.dropna(subset=["churn_label"])
        logger.info(
            f"Labeled rows: {len(labeled)} ({len(labeled) / len(out):.1%} of total). "
            f"Churn rate among labeled: {labeled['churn_label'].mean():.1%}"
        )

    app()
