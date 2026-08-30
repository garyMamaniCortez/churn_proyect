"""CLI entrypoint to extract raw client data from the gym's Postgres database.

Usage
-----
    python -m churn_detection.dataset extract-all

Requires a `.env` file (see `.env.example`) with DB_HOST, DB_PORT, DB_NAME,
DB_USER, DB_PASSWORD pointing at a real, reachable Postgres instance — this
script must be run from an environment that has network access to that
database (this is NOT something the assistant/sandbox that wrote this code can
execute itself).

Each table is written as its own CSV under `data/raw/`, plus an
`extraction_metadata.json` capturing when the pull happened and how many rows
came back — useful context once the outputs are versioned with DVC.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from churn_detection.config import RAW_DATA_DIR, DatabaseSettings
from churn_detection.data.connection import DatabaseConnectionFactory, PostgresConnectionSettings
from churn_detection.data.repository import ClientDataRepository, PostgresClientDataRepository

app = typer.Typer(help="Extract raw client data from the gym's operational database.")

# Table name -> repository method name. Single source of truth for "extract-all"
# and for tests that want to assert every table is covered.
EXTRACTION_TARGETS: dict[str, str] = {
    "personas": "get_personas",
    "inscripciones": "get_inscripciones",
    "servicios": "get_servicios",
    "registros_acceso": "get_registros_acceso",
    "ventas_servicios": "get_ventas_servicios",
    "pagos_pendientes": "get_pagos_pendientes",
}


def _build_repository() -> ClientDataRepository:
    db_settings = DatabaseSettings()
    connection_settings = PostgresConnectionSettings(
        host=db_settings.host,
        port=db_settings.port,
        database=db_settings.name,
        user=db_settings.user,
        password=db_settings.password,
    )
    engine = DatabaseConnectionFactory.create_engine(connection_settings)
    return PostgresClientDataRepository(engine)


def extract_raw_tables(repository: ClientDataRepository, output_dir: Path) -> dict[str, int]:
    """Pull every table in `EXTRACTION_TARGETS` and write it to `output_dir` as CSV.

    Returns a dict of table_name -> row_count, so callers (CLI or tests) can
    report/assert on what was written without re-reading files from disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}

    for table_name, method_name in EXTRACTION_TARGETS.items():
        logger.info(f"Extracting '{table_name}'...")
        extractor = getattr(repository, method_name)
        df: pd.DataFrame = extractor()

        output_path = output_dir / f"{table_name}.csv"
        df.to_csv(output_path, index=False)
        row_counts[table_name] = len(df)
        logger.success(f"Wrote {len(df)} rows to {output_path}")

    metadata = {
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_counts": row_counts,
    }
    metadata_path = output_dir / "extraction_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    logger.info(f"Wrote extraction metadata to {metadata_path}")

    return row_counts


@app.command()
def extract_all(output_dir: Path = RAW_DATA_DIR) -> None:
    """Extract every client-related table into `output_dir` as CSV files."""
    repository = _build_repository()
    row_counts = extract_raw_tables(repository, output_dir)
    logger.success(f"Extraction complete: {row_counts}")


if __name__ == "__main__":
    app()
