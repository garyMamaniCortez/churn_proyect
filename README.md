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
cuales 6,750 (78.7%) tienen resultado conocido: 3,548 renovaron, 3,202 no.
**Tasa de churn entre los ciclos con resultado: ~47.4%.** (Estos números
avanzan levemente cada vez que se corre el pipeline en una fecha distinta,
ver nota de reproducibilidad temporal más abajo.)

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
| Regresión logística | mediana | `StandardScaler` | sí (`SKEWED_FEATURES`) |
| Red neuronal (TensorFlow) | mediana | `StandardScaler` | sí (`SKEWED_FEATURES`) |
| Hist Gradient Boosting | no, usa `NaN` nativo | no | no |

No se hizo eliminación de outliers como paso de modelado: los valores
extremos que aparecen en `churn_ciclos.csv` (por ejemplo, `monto_total_gastado`
hasta $1,919, o `tenure_dias` hasta 696 días por planes anuales largos) se
verificaron manualmente y son clientes reales, no errores de carga. El único
outlier real de todo el proyecto (una cuenta genérica de recepción usada para
pases de día, id=32) ya se excluye en `config.EXCLUDED_PERSONA_IDS`, antes de
llegar a este dataset.

### Por qué volvió la regresión logística (como baseline, no como reemplazo)

En una iteración anterior se había sacado la regresión logística del proyecto
para meter la red neuronal, dejando la comparación entre Random Forest, Hist
Gradient Boosting y la red neuronal: tres modelos "complejos" entre sí, sin
ningún modelo de referencia simple. Al revisar la monografía contra una lista
de errores frecuentes en proyectos de Ciencia de Datos, esto encajaba
directamente en uno de ellos: *no definir un modelo baseline*, es decir, no
tener cómo demostrar que la complejidad extra de los otros modelos realmente
se traduce en mejor desempeño. Se sacó **Random Forest** (quedaba redundante
con Hist Gradient Boosting, ambos basados en árboles) y volvió la **regresión
logística**, esta vez explícitamente como el modelo de referencia contra el
que se comparan los otros dos.

### Red neuronal

`KerasBinaryClassifier` es un wrapper propio (no `scikeras`, para no sumar una
dependencia extra) que hace que un modelo de Keras se comporte como cualquier
estimador de scikit-learn (`.fit`, `.predict`, `.predict_proba`), para poder
meterlo en el mismo `Pipeline` que los demás candidatos. Arquitectura: red
densa feed-forward de 2 capas ocultas (32 y 16 neuronas, activación ReLU),
salida sigmoide, optimizador Adam, 40 épocas.

**Bug encontrado y corregido al implementarla:** un modelo de Keras crudo
**no es serializable con joblib/pickle** por defecto (tiene estado interno de
TensorFlow que no se puede picklear tal cual). Es el mismo tipo de bug que ya
nos había pasado con la regresión logística la primera vez (ahí era una
función closure local). Se corrigió implementando `__getstate__`/`__setstate__`
en `KerasBinaryClassifier`: al picklear, el modelo se guarda con el formato
nativo de Keras a un archivo temporal y se convierte a bytes; al despicklear,
se reconstruye desde esos bytes. Cubierto por
`test_neural_network_pipeline_is_picklable`, siguiendo el mismo patrón de
"agregar el test que hubiera atrapado esto" que ya usamos antes.

**Otro bug corregido en el camino:** la transformación `log1p` de las
variables asimétricas estaba originalmente definida como closure local, lo
cual tampoco es picklable. Se corrigió moviéndola a una clase a nivel de
módulo (`_Log1pSkewedColumns`), compartida hoy entre la regresión logística y
la red neuronal.

### Tuning de hiperparámetros con validación cruzada agrupada

Hasta esta fase, los tres modelos entrenaban con hiperparámetros fijos a
mano, nunca se había corrido una búsqueda real. Se agregó
`HyperparameterTuner`, que envuelve `GridSearchCV` pero usando `GroupKFold`
en vez de un k-fold común, agrupando por `persona_id` con el mismo criterio
que `GroupAwareSplitter` ya usaba para el split train/test: si los folds de
validación cruzada pudieran mezclar ciclos del mismo cliente, una
combinación de hiperparámetros podría verse mejor solo porque el modelo
memorizó parcialmente a ese cliente, no porque generalice mejor.

Cada candidato declara su propia grilla vía `param_grid()` (patrón
Open/Closed, igual que `build_pipeline()`): regresión logística busca sobre
`C` (4 valores), Hist Gradient Boosting sobre `learning_rate`, `max_depth` y
`max_iter` (27 combinaciones), y la red neuronal sobre `hidden_units` y
`learning_rate` (4 combinaciones, deliberadamente chico porque cada
combinación reentrena la red desde cero en cada fold).

**Bug real encontrado al correr esto por primera vez:** `GridSearchCV` con
`scoring="roc_auc"` fallaba para la red neuronal con el error *"Got a
regressor with response_method=predict_proba"*. La causa: `KerasBinaryClassifier`
estaba declarada como `class KerasBinaryClassifier(BaseEstimator,
ClassifierMixin)`, y en scikit-learn 1.8 el orden de las clases base importa
para que el sistema de tags (`__sklearn_tags__`) resuelva correctamente vía
MRO. Con `BaseEstimator` primero, su propia implementación de
`__sklearn_tags__` se ejecuta antes que la de `ClassifierMixin` y nunca
incorpora `estimator_type="classifier"`, así que `is_classifier(...)` daba
`False` para un modelo que evidentemente es un clasificador. Se corrigió
invirtiendo el orden a `class KerasBinaryClassifier(ClassifierMixin,
BaseEstimator)` (el orden que la propia documentación de scikit-learn
recomienda y que fácilmente se pasa por alto), con dos tests dedicados que lo
cubren: uno directo sobre `is_classifier()` y otro de integración corriendo
`HyperparameterTuner` de punta a punta sobre la red neuronal.

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Mejores hiperparámetros |
|---|---|---|---|---|---|---|---|
| **Hist Gradient Boosting** (ganador) | 0.747 | 0.715 | 0.765 | 0.739 | **0.823** | 0.795 | `learning_rate=0.05, max_depth=5, max_iter=100` |
| Red neuronal (TensorFlow) | 0.720 | 0.692 | 0.728 | 0.710 | 0.809 | 0.787 | `hidden_units=(32,16), learning_rate=0.001` |
| Regresión logística (baseline) | 0.718 | 0.693 | 0.719 | 0.706 | 0.791 | 0.744 | `C=10.0` |

El orden entre los tres modelos no cambió respecto a la corrida sin tuning,
y el ganador sigue superando claramente al baseline (0.823 vs. 0.791). El
tuning aportó una mejora modesta pero real en las tres métricas de ROC-AUC
frente a los hiperparámetros fijos anteriores, y sobre todo reemplazó
parámetros elegidos a mano por parámetros elegidos con evidencia. Se guardó
Hist Gradient Boosting en `models/churn_model.joblib`.

**Nota sobre reproducibilidad temporal:** `churn_dataset.py` usa la fecha
actual como corte para decidir qué ciclos están `censurado` vs. ya resueltos,
así que volver a correr `dvc repro` en una fecha distinta puede mover
ligeramente algunos ciclos de `censurado` a `churned` (los que ya agotaron su
ventana de gracia desde la última corrida) y cambiar las métricas en un
margen pequeño. No es no determinismo del modelo, es que el "hoy" del
snapshot efectivamente avanza.

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
`censurado` (1,823 clientes sin resultado resuelto todavía, al último corte)
y los clasifica en riesgo bajo/medio/alto (733 alto, 364 medio, 726 bajo).
`data/processed/predicciones_churn.csv` queda ordenado de mayor a menor
probabilidad de abandono, listo para que el equipo de retención lo use como
lista de priorización.

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
