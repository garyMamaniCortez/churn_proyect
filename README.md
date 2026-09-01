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

### Dataset de segmentación (esta fase)

`churn_detection/features.py` (`ClientSegmentationTableBuilder`) arma **solo**
el dataset de segmentación: una fila por cliente, todas las features medidas
en un único `snapshot_date` global (hoy). Es la vista correcta para "cómo se
ve la base de clientes ahora mismo", pero **no** es lo que usaremos para
entrenar el modelo de churn — para eso, cada cliente necesita su propia fecha
de corte (justo antes de que se sepa su desenlace), o el modelo aprende a
reconocer clientes que ya se fueron en vez de anticipar quién está por irse.
Ese dataset de churn con corte por cliente es la siguiente fase, todavía no
implementada.

**Columnas explícitamente excluidas y por qué:**
- Ningún campo de `estado` (ni `persona_estado`, ni ningún otro) — por
  instrucción explícita, no se agregan campos de "estado" del negocio.
- `edad` / `fecha_nacimiento` — 66% de los clientes no tienen fecha de
  nacimiento registrada. Regla del proyecto: si una columna supera 40% de
  nulos, se elimina en vez de imputar.

**Features nuevas de esta fase:**
- `hora_promedio_checkin`, `hora_checkin_std` — horario habitual de asistencia
  y qué tan variable es (rutinario vs. esporádico).
- `pct_visitas_fin_de_semana`, `dia_semana_mas_frecuente` — patrón semanal.
- `porcentaje_uso_membresia` — de los ingresos que pagó, qué porcentaje usó
  realmente (`1 - ingresos_disponibles / numero_ingresos`, promediado sobre
  sus inscripciones de tipo membresía; los pases de un solo día no cuentan).
- `frecuencia_visitas_semanal` — check-ins por semana, normalizado por el
  período real en que estuvo activo (no por la antigüedad total como cliente).

Todas las columnas quedan bajo 40% de nulos en el corte de 2026-08-30 (la peor
es `hora_checkin_std` con 22.9%, porque pedir desviación estándar requiere al
menos 2 check-ins confiables). Los nulos restantes se dejan tal cual — no se
imputan acá a propósito — porque decidir cómo tratarlos (¿0? ¿mediana? ¿una
categoría aparte?) es una decisión de la etapa de modelado, no de esta.

**Corrección importante aplicada en esta fase:** el módulo de registro de
accesos no funcionó de forma confiable antes de febrero-2026 (lo confirmó el
cliente, y los datos lo respaldan: enero-2026 tiene solo 5 registros de acceso
en total, y las inscripciones que empezaron antes de esa fecha muestran 61%
"nunca usadas" vs. 10% desde febrero en adelante). Sin esta corrección, ese
hueco de datos se leía como "el cliente nunca vino", cuando en realidad es "el
sistema no lo registró". Se agregó `config.RELIABLE_ACCESS_TRACKING_SINCE =
2026-02-01`; tanto `registros_acceso` como el cálculo de
`porcentaje_uso_membresia` excluyen (no imputan a 0) todo lo anterior a esa
fecha. El % de nulos en esas columnas subió como consecuencia (ver arriba) —
es más honestidad sobre lo que no sabemos, no menos información real.

## Cómo construir el dataset de segmentación (después de extraer los datos)

```powershell
python -m churn_detection.features
```

Lee todo `data/raw/*.csv` y escribe `data/processed/clientes_segmentacion.csv`.
Luego versiónalo:

```powershell
dvc add data/processed/clientes_segmentacion.csv
git add data/processed/clientes_segmentacion.csv.dvc .gitignore
git commit -m "data: dataset de segmentacion de clientes"
```

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
    ├── features.py             <- ClientSegmentationTableBuilder: builds
    │                              data/processed/clientes_segmentacion.csv
    │                              (tenure + engagement/horario + uso + RFM) from data/raw/*.csv
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

