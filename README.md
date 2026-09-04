# Churn Detection Model

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto de deteccion de churn para clientes de un gimnasio.

## Estado del proyecto

Pipeline de extracción + dataset de churn + EDA + modelo supervisado
entrenado y puntuando clientes en riesgo, todo reproducible con DVC. La
segmentación de clientes (clustering no supervisado) se evaluó en una fase
anterior del proyecto pero se descartó por no aportar valor al problema de
churn; no queda código de esa vía en el repositorio.

## Cómo extraer los datos (ejecutar en un entorno CON acceso a la base de datos)

1. `cp .env.example .env` y completar `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`,
   `DB_PASSWORD` con las credenciales reales. `CHURN_GRACE_DAYS` es un parámetro
   de negocio (días de gracia tras el vencimiento antes de considerar que un
   ciclo de membresía no se renovó) — 30 es el valor usado en todo el proyecto.
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
`huella_digital` (huella biométrica) — no aportan valor predictivo para churn
y no tiene sentido versionarlos en CSVs planos. Solo se usa `persona_id` como
llave.

### Corrección de calidad de datos: ventana de tracking confiable

El módulo de registro de accesos del gimnasio no funcionó de forma confiable
antes de febrero-2026 (confirmado por el cliente, y respaldado por los datos:
enero-2026 tiene solo 5 registros de acceso en total, y las inscripciones que
empezaron antes de esa fecha muestran 61% "nunca usadas" vs. 10% desde
febrero en adelante). Sin esta corrección, ese hueco de datos se leería como
"el cliente nunca vino", cuando en realidad es "el sistema no lo registró".
Se agregó `config.RELIABLE_ACCESS_TRACKING_SINCE = 2026-02-01`; toda feature
derivada de `registros_acceso` (y el cálculo de `porcentaje_uso_membresia`,
que depende de `ingresos_disponibles`) excluye —no imputa a 0— todo lo
anterior a esa fecha.

## Dataset de churn: una fila por ciclo de membresía, no por cliente

`churn_detection/churn_dataset.py` (`ChurnCycleDatasetBuilder`) construye el
dataset de entrenamiento.

**Por qué una fila por ciclo y no por cliente:** usar la última membresía de
cada cliente como única observación es circular. Si el cliente renovó, esa
renovación pasa a ser su membresía "más reciente", así que la que se estaba
evaluando nunca puede resolver en "renovó" bajo esa definición, solo en
"abandonó" o "todavía sin resolver". Un modelo entrenado así jamás vería un
ejemplo positivo de retención. La solución: cada inscripción de tipo
membresía que un cliente tuvo es su propia observación. Un cliente con 3
membresías a lo largo del tiempo aporta hasta 3 filas.

**Sin fuga de datos:** cada fila usa `fecha_vencimiento` de ESA membresía
como corte. Todas sus features (check-ins, gasto, uso, antigüedad) se calculan
únicamente con información con fecha anterior o igual a ese corte. Nada de lo
que pasó durante o después del ciclo siguiente se usa para predecir el
resultado de este.

**Label y censura**, por ciclo:
- `renovado` (`churn_label=0`): hubo una inscripción de membresía siguiente
  que empezó dentro de `CHURN_GRACE_DAYS` días tras el vencimiento de esta.
  Los pases de un solo día en el medio no cuentan como renovación.
- `churned` (`churn_label=1`): no hubo renovación a tiempo, y ya pasó
  suficiente tiempo como para saberlo con certeza.
- `censurado` (`churn_label=NaN`): es el último ciclo conocido del cliente y
  todavía está dentro de la ventana de gracia, no se sabe el resultado
  todavía. Se excluye del entrenamiento; es exactamente el conjunto de
  clientes a los que se les aplicaría el modelo en producción.

**Resultado real (`CHURN_GRACE_DAYS=30`):** 8,573 ciclos de membresía, de los
cuales 6,728 (78.5%) tienen resultado conocido: 3,548 renovaron, 3,180 no.
**Tasa de churn entre los ciclos con resultado: ~47.3%.**

Como control de calidad: comparando el promedio de las features entre ciclos
`renovado` y `churned`, todas las diferencias van en la dirección esperada
(los que abandonan tienen menos antigüedad, menos uso de membresía, menos
ritmo reciente y menos gasto), señal de que el dataset es coherente antes de
modelar.

Salida: `data/processed/churn_ciclos.csv`, con columnas de identidad
(`persona_id`, `inscripcion_id`, fechas), `estado_ciclo`, `churn_label`, y 11
features de comportamiento (antigüedad, uso de membresía, recencia,
frecuencia, horario, patrón semanal, gasto y su tendencia reciente, número de
inscripciones, ritmo reciente vs. histórico, y regularidad de visitas).

Ninguna columna supera el 40% de nulos; el peor caso ronda el 27%
(`cv_gap_visitas`, porque requiere al menos 3 check-ins para calcularse). Los
nulos restantes se dejan tal cual a propósito — decidir cómo tratarlos es una
decisión de modelado, no de esta etapa.

## EDA del dataset de churn

`churn_detection/churn_plots.py` (`ChurnEDAFigureGenerator`) genera 4 figuras
propias (`08` a `11` en `reports/figures/`):
- Distribución de `estado_ciclo` (41.4% renovado, 37.0% churned, 21.6%
  censurado).
- Nulos por columna.
- **Distribución de cada feature separada por `renovado` vs. `churned`**: la
  más útil de las cuatro, muestra directamente qué variables separan a quien
  se queda de quien se va. `porcentaje_uso_membresia`, `n_inscripciones_total`
  y `ratio_actividad_reciente` muestran una separación visual clara;
  `hora_promedio_checkin`, `cv_gap_visitas` y `pct_visitas_fin_de_semana` se
  superponen casi por completo, señal de que aportan poco al modelo.
- Correlación entre features.

## Entrenamiento del modelo

`churn_detection/modeling/train.py` compara 3 modelos candidatos (patrón de
clase-por-candidato: `ChurnModelCandidate` ABC + una subclase por modelo), con
split train/test **agrupado por `persona_id`** (`GroupAwareSplitter`) para que
ningún cliente tenga ciclos repartidos entre train y test, y con tracking en
**MLflow** (experimento `churn_gimnasio`, backend SQLite local en
`mlflow.db`).

**Preprocesamiento aplicado, por modelo** (no es el mismo para los tres, y es
así a propósito):

| | Imputación de nulos | Escalado | Log1p en features asimétricas |
|---|---|---|---|
| Red neuronal (TensorFlow) | mediana | `StandardScaler` | sí (`SKEWED_FEATURES`) |
| Random Forest | mediana | no (invariante a escala) | no (invariante a transformaciones monótonas) |
| Hist Gradient Boosting | no, usa `NaN` nativo | no | no |

No se hizo eliminación de outliers como paso de modelado: los valores
extremos que aparecen en `churn_ciclos.csv` (por ejemplo, `monto_total_gastado`
hasta $1,919, o `tenure_dias` hasta 696 días por planes anuales largos) se
verificaron manualmente y son clientes reales, no errores de carga. El único
outlier real de todo el proyecto (una cuenta genérica de recepción usada para
pases de día, id=32) ya se excluye en `config.EXCLUDED_PERSONA_IDS`, antes de
llegar a este dataset.

### Red neuronal (reemplazó a la regresión logística)

`KerasBinaryClassifier` es un wrapper propio (no `scikeras`, para no sumar una
dependencia extra) que hace que un modelo de Keras se comporte como cualquier
estimador de scikit-learn (`.fit`, `.predict`, `.predict_proba`), para poder
meterlo en el mismo `Pipeline` que los demás candidatos. Arquitectura: red
densa feed-forward de 2 capas ocultas (32 y 16 neuronas, activación ReLU),
salida sigmoide, optimizador Adam, 40 épocas.

**Bug encontrado y corregido al implementarla:** un modelo de Keras crudo
**no es serializable con joblib/pickle** por defecto (tiene estado interno de
TensorFlow que no se puede picklear tal cual). Es el mismo tipo de bug que ya
nos había pasado con la regresión logística (ahí era una función closure
local). Se corrigió implementando `__getstate__`/`__setstate__` en
`KerasBinaryClassifier`: al picklear, el modelo se guarda con el formato
nativo de Keras a un archivo temporal y se convierte a bytes; al despicklear,
se reconstruye desde esos bytes. Cubierto por
`test_neural_network_pipeline_is_picklable`, siguiendo el mismo patrón de
"agregar el test que hubiera atrapado esto" que ya usamos antes.

**Dos bugs adicionales corregidos en fases anteriores de esta etapa:**
1. La regresión logística (ahora eliminada) no tenía la transformación
   `log1p` en las variables asimétricas. Al corregirlo en su momento, su
   ROC-AUC había subido de 0.777 a 0.790.
2. Ese mismo `log1p` estaba originalmente definido como closure local, lo
   cual tampoco es picklable — se corrigió moviéndolo a una clase a nivel de
   módulo (`_Log1pSkewedColumns`), reutilizada ahora también por la red
   neuronal.

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Hist Gradient Boosting** (ganador) | 0.745 | 0.713 | 0.763 | 0.737 | **0.823** | 0.797 |
| Random Forest | 0.736 | 0.709 | 0.740 | 0.724 | 0.818 | 0.793 |
| Red neuronal (TensorFlow) | 0.720 | 0.692 | 0.724 | 0.708 | 0.810 | 0.774 |

Los tres por encima de 0.81 de ROC-AUC con un split honesto (agrupado por
cliente). La red neuronal queda última, no muy lejos de los otros dos, pero
sin superarlos, algo esperable con un dataset tabular relativamente chico (11
features, ~5,000 filas de entrenamiento): los modelos basados en árboles
suelen ganarle a una red densa en este régimen de datos, donde no hay
suficiente volumen para que la red aproveche su capacidad extra. Se guardó
Hist Gradient Boosting en `models/churn_model.joblib`.

Como ese modelo no expone `feature_importances_` ni `coef_`, la importancia
de variables (`reports/figures/13_*.png`) se calculó con **permutation
importance** (`sklearn.inspection.permutation_importance`, caída de ROC-AUC al
mezclar cada columna), un método universal que también funciona para
cualquiera de los otros dos modelos. Top variables: `n_inscripciones_total`,
`recencia_dias`, `tenure_dias`, `porcentaje_uso_membresia`, coherente con el
EDA por separación visual.

`Hist Gradient Boosting` se incluyó a propósito porque maneja `NaN`
nativamente, sin imputar, algo relevante dado que buena parte de la
"faltante" en estos datos es información real (falta de tracking confiable),
no ruido aleatorio.

## Scoring de clientes en riesgo

`churn_detection/modeling/predict.py` puntúa únicamente los ciclos
`censurado` (los ~1,845 clientes sin resultado resuelto todavía) y los
clasifica en riesgo bajo/medio/alto. `data/processed/predicciones_churn.csv`
queda ordenado de mayor a menor probabilidad de abandono, listo para que el
equipo de retención lo use como lista de priorización.

## Pipeline de DVC

`dvc.yaml` declara 4 etapas, cada una atada por hash a su código y a sus
dependencias de datos:

- **`churn_dataset`**: `churn_dataset.py` + los 6 CSV crudos → `churn_ciclos.csv`.
- **`churn_eda`**: `churn_plots.py` + `churn_ciclos.csv` → figuras `08`-`11`.
- **`train_churn_model`**: `train.py` + `churn_ciclos.csv` → `churn_model.joblib`,
  comparación de modelos, curva ROC, matriz de confusión, importancia de
  features.
- **`score_churn`**: `predict.py` + `churn_ciclos.csv` + el modelo entrenado →
  `predicciones_churn.csv`.

```powershell
dvc repro
```

Regenera solo lo que cambió (o todo, la primera vez) y actualiza `dvc.lock`.
Después, para versionar:

```powershell
git add dvc.yaml dvc.lock data/processed/.gitignore reports/figures/.gitignore models/.gitignore
git commit -m "churn: pipeline completo"
dvc push   # requiere tener un remote configurado (ver mas arriba)
```

Los datos crudos (`data/raw/*.csv`) siguen versionados aparte con `dvc add`,
y el pipeline los referencia como dependencias de solo lectura.

### Por qué MLflow va a `.gitignore` y no a DVC

`mlflow.db` (metadata de runs), `mlruns/` (artefactos de cada modelo
logueado, ~43 MB y creciendo) y `mlartifacts/` quedan en `.gitignore`, no
versionados con DVC. Es una decisión deliberada, no un descuido:

- **No son una función pura de los inputs.** Cada corrida de
  `train_churn_model` crea un `run_id` nuevo con timestamp propio, incluso
  si los datos y el código no cambiaron. El modelo de reproducibilidad de
  DVC asume que las mismas dependencias producen las mismas salidas
  (por eso puede cachear); la carpeta de MLflow viola eso por diseño, crece
  con cada corrida en vez de estabilizarse.
- **Ya están versionados, solo que por otra herramienta.** Para eso existe
  MLflow: es su propio sistema de tracking de experimentos. Pedirle a DVC
  que además versione la base de datos de tracking de MLflow es duplicar
  responsabilidades sin ganar nada.
- **El artefacto que de verdad importa ya está en DVC.** El modelo ganador
  se guarda aparte, de forma determinística, en
  `models/churn_model.joblib` (salida de la etapa `train_churn_model` en
  `dvc.yaml`). Eso sí es reproducible y sí vale la pena versionar.

Si en algún momento se quiere compartir el historial de experimentos entre
el equipo, la solución correcta no es forzarlo dentro de DVC sino apuntar
`MLFLOW_TRACKING_URI` a un tracking server remoto (o a un backend
compartido), que es exactamente para lo que existe esa opción de
configuración.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models (churn_model.joblib)
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
    ├── config.py               <- Paths + DatabaseSettings/ChurnSettings (pydantic-settings) +
    │                              EXCLUDED_PERSONA_IDS + RELIABLE_ACCESS_TRACKING_SINCE + MLflow config
    │
    ├── data                    <- DB access layer (SOLID: interface + Postgres impl)
    │   ├── __init__.py
    │   ├── connection.py       <- PostgresConnectionSettings + DatabaseConnectionFactory
    │   ├── queries.py          <- Raw SQL, isolated from Python control flow
    │   └── repository.py       <- ClientDataRepository (ABC) + PostgresClientDataRepository
    │
    ├── dataset.py              <- CLI (typer) to extract raw tables into data/raw/*.csv
    │
    ├── membership_utils.py     <- Shared: flag_membership_services (día-pass vs. membresía)
    │
    ├── churn_dataset.py        <- ChurnCycleDatasetBuilder: builds data/processed/churn_ciclos.csv
    │                              (una fila por ciclo de membresía, sin fuga de datos)
    │
    ├── churn_plots.py          <- ChurnEDAFigureGenerator: builds reports/figures/08-11_churn_*.png
    │
    └── modeling                
        ├── __init__.py 
        ├── predict.py          <- Scores open ("censurado") cycles -> predicciones_churn.csv
        └── train.py            <- Trains + compares churn models (MLflow-tracked) -> churn_model.joblib
```

## Stack

POO + SOLID + PEP8, `pytest` para tests, `dvc` para versionado de datos y
pipeline, `MLflow` para tracking de experimentos, `scikit-learn` y
`TensorFlow`/`Keras` para los modelos. `FastAPI` para servir el modelo y
`Docker` para contenerizar todavía no están implementados — quedan como
siguiente fase.

--------
