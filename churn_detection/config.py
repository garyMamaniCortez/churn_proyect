from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


class DatabaseSettings(BaseSettings):
    """Postgres connection settings, read from `.env` / real environment variables.

    Expected variables: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD.
    No defaults are given for credentials/db name on purpose: fail loudly at
    startup instead of silently pointing at the wrong database.
    """

    model_config = SettingsConfigDict(
        env_prefix="DB_", env_file=str(PROJ_ROOT / ".env"), extra="ignore"
    )

    host: str = "localhost"
    port: int = 5432
    name: str
    user: str
    password: str


class ChurnSettings(BaseSettings):
    """Business parameters for churn labeling.

    churn_grace_days: how many days after `fecha_vencimiento` a client is still
    allowed to renew before being labeled as churned. Default of 30 is a
    PROVISIONAL assumption (roughly one billing cycle) — confirm with the gym's
    actual renewal patterns during EDA before training on it.
    """

    model_config = SettingsConfigDict(env_file=str(PROJ_ROOT / ".env"), extra="ignore")

    churn_grace_days: int = Field(default=30, ge=0)


# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
