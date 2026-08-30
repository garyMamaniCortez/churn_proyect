# Churn Detection Model

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto de deteccion de churn para clientes de un gimnasio

## Estado del proyecto

**Fase actual: extracción de datos + preparación para EDA/segmentación.**
Aún no hay modelo de churn ni de segmentación — primero necesitamos datos reales
extraídos de Postgres para hacer EDA con criterio.

## Cómo extraer los datos (ejecutar en un entorno CON acceso a la base de datos)

1. `cp .env.example .env` y completar `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`,
   `DB_PASSWORD` con las credenciales reales. `CHURN_GRACE_DAYS` es un parámetro
   de negocio (ver nota más abajo) — 30 es un valor provisional.
2. `pip install -r requirements.txt`
3. `make extract-data` (o `python -m churn_detection.dataset extract-all`).
   Esto escribe `personas.csv`, `inscripciones.csv`, `servicios.csv`,
   `registros_acceso.csv`, `ventas_servicios.csv`, `pagos_pendientes.csv` y
   `extraction_metadata.json` en `data/raw/`.
4. Versionar lo extraído con DVC: `make version-data`, luego
   `git add data/raw/*.dvc .gitignore && git commit -m "data: extracción inicial"`.
   **Nota:** el repo ya tiene `dvc init` corrido, pero todavía no hay un remote
   de DVC configurado (no asumí dónde quieres guardar los datos — S3, GCS, un
   disco compartido, etc.). Configúralo con:
   `dvc remote add -d storage <url-o-path>` antes de hacer `dvc push`.

### Nota sobre `personas.csv`
Por diseño, la extracción de `personas` **excluye** `ci`, `telefono` y
`huella_digital` (huella biométrica) — no aportan valor predictivo para churn/
segmentación y no tiene sentido versionarlos en CSVs planos. Solo se usa
`persona_id` como llave.

### Nota sobre la definición de churn
Se implementó (a nivel de configuración, `ChurnSettings.churn_grace_days`) el
criterio que elegiste: un cliente se considera **churned** si, tras el
vencimiento de su última inscripción, no genera una inscripción nueva dentro de
`CHURN_GRACE_DAYS` días. El cálculo del label en sí (usando `inscripciones.csv`)
se hará en la siguiente fase, una vez tengamos datos reales para validar que el
criterio produce una distribución de clases razonable.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         churn_detection and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── churn_detection   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes churn_detection a Python module
    │
    ├── config.py               <- Paths + DatabaseSettings/ChurnSettings (pydantic-settings)
    │
    ├── data                    <- DB access layer (SOLID: interface + Postgres impl)
    │   ├── __init__.py
    │   ├── connection.py       <- PostgresConnectionSettings + DatabaseConnectionFactory
    │   ├── queries.py          <- Raw SQL, isolated from Python control flow
    │   └── repository.py       <- ClientDataRepository (ABC) + PostgresClientDataRepository
    │
    ├── dataset.py              <- CLI (typer) to extract raw tables into data/raw/*.csv
    │
    ├── features.py             <- Code to create features for modeling (next phase)
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Stack (según lo definido)

POO + SOLID + PEP8, `pytest` para tests, `dvc` para versionado de datos, `MLflow`
para tracking de experimentos, `FastAPI` para servir el modelo y `Docker` para
contenerizar. Por ahora solo está implementada la capa de extracción de datos
(esta fase); MLflow, FastAPI y Docker se agregan cuando lleguemos a modelado y
despliegue, para no acumular dependencias/código sin uso.

--------

