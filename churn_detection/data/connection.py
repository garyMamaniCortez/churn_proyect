"""Postgres connection management.

Kept deliberately small and single-purpose (SRP): this module's only job is to turn
validated connection settings into a usable SQLAlchemy `Engine`. It knows nothing
about gym-domain tables or queries — that lives in `repository.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


@dataclass(frozen=True)
class PostgresConnectionSettings:
    """Immutable value object describing how to reach the gym's Postgres instance."""

    host: str
    port: int
    database: str
    user: str
    password: str

    @property
    def sqlalchemy_url(self) -> str:
        """Build a psycopg2-flavored SQLAlchemy URL from the settings."""
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class DatabaseConnectionFactory:
    """Builds SQLAlchemy engines from `PostgresConnectionSettings`.

    Isolated behind a factory (Dependency Inversion) so repositories and tests depend
    on an `Engine` abstraction, never on how the connection string is assembled or on
    a hardcoded set of credentials.
    """

    @staticmethod
    def create_engine(settings: PostgresConnectionSettings) -> Engine:
        """Create a pooled SQLAlchemy engine. Does not open a connection eagerly."""
        return create_engine(settings.sqlalchemy_url, pool_pre_ping=True, future=True)
