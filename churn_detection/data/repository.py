"""Repositories for reading gym client data out of the operational database.

Design notes (SOLID)
---------------------
- `ClientDataRepository` is the abstraction the rest of the codebase (dataset.py,
  future feature engineering, tests) depends on — never on Postgres directly
  (Dependency Inversion). A CSV-backed or mocked repository can implement the same
  interface for tests or for an offline/air-gapped environment.
- Each method has exactly one reason to change: the shape of one source table
  (Single Responsibility / Interface Segregation — callers only need the subset of
  methods relevant to them).
- `PostgresClientDataRepository` takes its `Engine` via constructor injection; it
  never builds its own connection, which keeps it trivially testable with a mock.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from sqlalchemy.engine import Engine

from churn_detection.data import queries


class ClientDataRepository(ABC):
    """Contract for retrieving gym client-related data, decoupled from the storage engine."""

    @abstractmethod
    def get_personas(self) -> pd.DataFrame:
        """Return one row per customer (staff excluded)."""

    @abstractmethod
    def get_inscripciones(self) -> pd.DataFrame:
        """Return the full history of service enrollments."""

    @abstractmethod
    def get_servicios(self) -> pd.DataFrame:
        """Return the service catalog (price, duration, multi-branch flag, etc.)."""

    @abstractmethod
    def get_registros_acceso(self) -> pd.DataFrame:
        """Return client check-in/access events."""

    @abstractmethod
    def get_ventas_servicios(self) -> pd.DataFrame:
        """Return service-sale transactions (monetary signal)."""

    @abstractmethod
    def get_pagos_pendientes(self) -> pd.DataFrame:
        """Return outstanding-payment records."""


class PostgresClientDataRepository(ClientDataRepository):
    """`ClientDataRepository` backed by the gym's Postgres database."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get_personas(self) -> pd.DataFrame:
        return pd.read_sql(queries.QUERY_PERSONAS_CLIENTES, self._engine)

    def get_inscripciones(self) -> pd.DataFrame:
        return pd.read_sql(queries.QUERY_INSCRIPCIONES, self._engine)

    def get_servicios(self) -> pd.DataFrame:
        return pd.read_sql(queries.QUERY_SERVICIOS, self._engine)

    def get_registros_acceso(self) -> pd.DataFrame:
        return pd.read_sql(queries.QUERY_REGISTROS_ACCESO, self._engine)

    def get_ventas_servicios(self) -> pd.DataFrame:
        return pd.read_sql(queries.QUERY_VENTAS_SERVICIOS, self._engine)

    def get_pagos_pendientes(self) -> pd.DataFrame:
        return pd.read_sql(queries.QUERY_PAGOS_PENDIENTES, self._engine)
