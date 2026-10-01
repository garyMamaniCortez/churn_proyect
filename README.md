# Churn Detection Model

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto de detección de abandono (churn) de clientes de una cadena de
gimnasios: a partir del historial de membresías, visitas y pagos, estima la
probabilidad de que un ciclo de membresía abierto termine en abandono, y
genera una lista de clientes priorizada por riesgo para el equipo de
retención.

Este documento explica cómo correr el proyecto de punta a punta y cómo está
armado cada paso. Para el detalle metodológico completo (justificación de
cada decisión, resultados y discusión) ver `MONOGRAFIA.md`.

## Índice

- [Requisitos](#requisitos)
- [Puesta en marcha](#puesta-en-marcha)
- [Cómo correr el proyecto](#cómo-correr-el-proyecto-paso-a-paso)
- [Cómo funciona cada etapa](#cómo-funciona-cada-etapa)
- [Archivos que produce cada etapa](#archivos-que-produce-cada-etapa)
- [Tests](#tests)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Stack](#stack)

## Requisitos

- Python ≥ 3.10
- Acceso a la base de datos Postgres del gimnasio (solo para el paso de
  extracción; si ya tenés `data/raw/*.csv`, no hace falta)
- `pip`, y opcionalmente `dvc` si querés correr el pipeline completo con un
  solo comando

## Puesta en marcha

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Completar `.env` con las credenciales reales de la base de datos
(`DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`). `CHURN_GRACE_DAYS`
es un parámetro de negocio (días de gracia tras el vencimiento de una
membresía antes de considerarla no renovada) — el proyecto usa `30`.

Si no tenés acceso a la base de datos pero ya existen los CSV en
`data/raw/`, podés saltar directamente al paso 2 de la sección siguiente.

## Cómo correr el proyecto, paso a paso

Cada paso es un módulo ejecutable con `python -m`. El orden importa: cada
uno consume la salida del anterior.

### 1. Extraer los datos crudos (requiere `.env` con acceso a la base de datos)

```bash
python -m churn_detection.dataset extract-all
```

Escribe `personas.csv`, `inscripciones.csv`, `servicios.csv`,
`registros_acceso.csv`, `ventas_servicios.csv`, `pagos_pendientes.csv` y
`extraction_metadata.json` en `data/raw/`.

### 2. Construir el dataset de churn

```bash
python -m churn_detection.churn_dataset
```

Lee `data/raw/*.csv` y escribe `data/processed/churn_ciclos.csv`: una fila
por ciclo de membresía, con sus 11 variables de comportamiento y su
resultado (`renovado` / `churned` / `censurado`). Ver
[Cómo funciona cada etapa](#2-dataset-de-churn-una-fila-por-ciclo-no-por-cliente)
para el porqué de este diseño.

### 3. Generar las figuras del análisis exploratorio (opcional)

```bash
python -m churn_detection.churn_plots
```

Escribe 4 figuras (`08` a `11`) en `reports/figures/`.

### 4. Entrenar y comparar los modelos

```bash
python -m churn_detection.modeling.train
```

Compara 3 algoritmos, optimiza hiperparámetros, calibra probabilidades y
guarda el modelo final en `models/churn_model.joblib`. Tarda varios minutos
(la red neuronal es el paso más lento). Parámetros configurables — ver
`python -m churn_detection.modeling.train --help`:

| Parámetro | Default | Qué controla |
|---|---|---|
| `--test-size` | `0.20` | Proporción de ciclos reservada para el conjunto de prueba (split cronológico) |
| `--selection-metric` | `log_loss` | Métrica usada para elegir modelo, hiperparámetros y estado de calibración |
| `--min-val-size` | `100` | Tamaño mínimo de una ventana de validación en la validación cruzada walk-forward |
| `--min-train-size` | `700` | Tamaño mínimo de entrenamiento para que una partición walk-forward cuente |

### 5. Puntuar los ciclos abiertos

```bash
python -m churn_detection.modeling.predict --window-start 2026-08-30
```

Carga `models/churn_model.joblib` y puntúa los ciclos `censurado` (sin
resultado todavía) cuya `fecha_vencimiento` cae entre `--window-start` y esa
fecha más 7 días — la ventana operativa que el equipo de retención trabaja
en esa semana. Escribe `data/processed/predicciones_churn.csv`, ordenado de
mayor a menor probabilidad de abandono, con el riesgo clasificado en
bajo/medio/alto.

### Alternativa: correr todo con DVC

```bash
dvc repro
```

`dvc.yaml` encadena los pasos 2 a 5 (no el 1, que requiere credenciales y no
es determinístico) y solo recalcula lo que cambió. Para versionar:

```bash
git add dvc.yaml dvc.lock data/processed/.gitignore reports/figures/.gitignore models/.gitignore
git commit -m "churn: pipeline completo"
dvc push   # requiere un remote configurado: dvc remote add -d storage <url-o-path>
```

Los datos crudos (`data/raw/*.csv`) se versionan aparte con `dvc add` y el
pipeline los referencia como dependencias de solo lectura. **Nota:** algunos
de los artefactos nuevos descritos en la tabla de la sección siguiente
(análisis de sensibilidad, calibración y matrices por candidato) todavía no
están declarados como `outs` en `dvc.yaml`; se generan igual al correr
`train.py`, pero `dvc repro` no los trackea uno por uno todavía.

## Cómo funciona cada etapa

### 1. Extracción de datos

`churn_detection/dataset.py` lee la base operativa del gimnasio (capa de
acceso en `churn_detection/data/`: interfaz `ClientDataRepository` +
implementación Postgres, para no acoplar el resto del proyecto a un motor de
base de datos concreto) y vuelca las tablas relevantes a CSV en `data/raw/`.

Por diseño, la extracción de `personas` **excluye** `ci`, `telefono` y
`huella_digital` — no aportan valor predictivo y no tiene sentido
versionarlos en CSVs planos. Solo se usa `persona_id` como llave.

### 2. Dataset de churn: una fila por ciclo, no por cliente

`churn_detection/churn_dataset.py` (`ChurnCycleDatasetBuilder`) arma el
dataset de entrenamiento.

**Por qué una fila por ciclo y no por cliente:** usar la última membresía de
cada cliente como única observación es circular — si el cliente renovó, esa
renovación pasa a ser su membresía "más reciente", así que la que se estaba
evaluando nunca puede resolver en "renovó". Cada inscripción de tipo
membresía que un cliente tuvo es su propia observación; un cliente con 3
membresías aporta hasta 3 filas.

**Sin fuga de datos:** cada fila usa `fecha_vencimiento` de ESA membresía
como corte. Todas sus features (check-ins, gasto, uso, antigüedad) se
calculan únicamente con información de fecha anterior o igual a ese corte.

**Resultado de cada ciclo:**
- `renovado` (`churn_label=0`): hubo una inscripción de membresía siguiente
  dentro de `CHURN_GRACE_DAYS` días tras el vencimiento.
- `churned` (`churn_label=1`): no hubo renovación a tiempo, y ya pasó
  suficiente tiempo como para saberlo con certeza.
- `censurado` (`churn_label=NaN`): es el último ciclo conocido del cliente y
  todavía está dentro de la ventana de gracia. Se excluye del entrenamiento
  — es exactamente el conjunto de clientes que `predict.py` puntúa.

**Corrección de calidad de datos:** el registro de accesos no funcionó de
forma confiable antes de febrero de 2026 (confirmado por el cliente y por
los datos: enero-2026 tiene solo 5 registros de acceso en total).
`config.RELIABLE_ACCESS_TRACKING_SINCE = 2026-02-01` hace que toda feature
derivada de check-ins **excluya** —no impute a 0— lo anterior a esa fecha,
para no confundir "no hay registro" con "el cliente no vino".

Salida: `data/processed/churn_ciclos.csv`, con columnas de identidad
(`persona_id`, `inscripcion_id`, fechas), `estado_ciclo`, `churn_label`, y 11
features de comportamiento (antigüedad, uso de membresía, recencia,
frecuencia, horario, patrón semanal, gasto y su tendencia reciente, número
de inscripciones, ritmo reciente vs. histórico, y regularidad de visitas).

### 3. Análisis exploratorio

`churn_detection/churn_plots.py` genera 4 figuras (`08`-`11` en
`reports/figures/`): distribución de `estado_ciclo`, nulos por columna,
distribución de cada feature separada por `renovado` vs. `churned`, y
correlación entre features.

### 4. Entrenamiento (`churn_detection/modeling/train.py`)

El split train/test es **cronológico** (`ChronologicalSplitter`): train solo
contiene ciclos resueltos antes de una fecha de corte, test solo ciclos
resueltos después, para medir desempeño sobre datos genuinamente futuros. Un
mismo cliente puede aparecer a ambos lados (ciclo temprano en train, uno
posterior en test) porque `persona_id` nunca es una feature.

**Preprocesamiento, por modelo** (no es el mismo para los tres, a propósito):

| | Imputación de nulos | Escalado | Log1p en features asimétricas |
|---|---|---|---|
| Regresión logística | mediana | `StandardScaler` | sí |
| Red neuronal (TensorFlow/Keras) | mediana | `StandardScaler` | sí |
| Hist Gradient Boosting | no, usa `NaN` nativo | no | no |

**Validación cruzada: ventana expansiva por mes, no K-Fold aleatorio.**
Dentro del train, la selección de modelo, el tuning de hiperparámetros y la
calibración usan `WalkForwardGroupSplitter`: cada partición entrena solo con
los ciclos resueltos hasta un mes calendario y valida con el mes siguiente,
purgando del entrenamiento a cualquier cliente presente en el mes de
validación. Un K-Fold aleatorio (incluso agrupado por cliente) no evita que
una partición de validación de un mes temprano se evalúe con un modelo
entrenado también con meses posteriores — exactamente lo que el split
cronológico externo ya evita, pero sin protección un nivel más abajo. En una
comparación con el esquema anterior (K-Fold aleatorio), Hist Gradient
Boosting ganaba; bajo ventana expansiva pasa a ser el peor de los tres, lo
que indica que su ventaja aparente dependía de mezclar información de meses
futuros en el entrenamiento — el motivo concreto por el que se cambió de
esquema.

**Fase 1 — selección de candidatos:** los 3 algoritmos (Regresión Logística,
Hist Gradient Boosting, Red Neuronal) se validan con 5 particiones walk-forward
sobre el train set. Se retienen los **2 mejores** por `log_loss` medio de
validación cruzada para la siguiente fase (actualmente: Regresión Logística y
Red Neuronal; Hist Gradient Boosting queda descartado).

**Fase 2 — tuning + análisis de sensibilidad:** cada uno de los 2 candidatos
retenidos se afina por separado (`HyperparameterTuner` sobre `GridSearchCV`,
mismo `WalkForwardGroupSplitter`). Además de la combinación ganadora, se
guarda el `log_loss` de **todas** las combinaciones probadas
(`sensibilidad_hiperparametros_<modelo>.csv`), para saber qué tan sensible es
cada modelo a la elección de sus hiperparámetros — un rango angosto indica un
modelo robusto, uno amplio indica que la combinación ganadora pudo depender
de la partición de datos.

**Fase 3 — calibración, para los 2 candidatos tuneados:** seleccionar por
`log_loss` favorece que las probabilidades salgan razonablemente calibradas,
pero no lo garantiza, así que se verifica y corrige explícitamente:

1. Se generan probabilidades **out-of-fold** de cada modelo ya afinado sobre
   el train set (nunca las predicciones in-sample: un modelo es
   sistemáticamente más confiado sobre datos que ya vio).
2. Se elige entre regresión isotónica y escalado sigmoide por validación
   cruzada (`log_loss`, no `brier_score` — este último no fue lo bastante
   sensible al sobreajuste de isotónica en los deciles de mayor riesgo,
   donde hay pocas observaciones).
3. El umbral de decisión se elige por validación cruzada maximizando **F1**
   (no `log_loss`: el `log_loss` no depende del umbral, porque evalúa la
   probabilidad cruda, no una etiqueta dura — maximizar F1 es el criterio
   correcto justamente porque F1 sí depende de dónde se corta).

Los diagramas y tablas de confiabilidad de esta fase se construyen sobre el
**conjunto de evaluación** (las probabilidades out-of-fold), no sobre el
conjunto de prueba: es la misma evidencia que ya se usó para elegir el
método de calibración y el umbral.

**Selección final: 4 combinaciones, no 2.** Calibrar mejora el `log_loss`
en ambos modelos, pero no garantiza que el que ganaba sin calibrar siga
ganando calibrado, así que la decisión final compara las 4 combinaciones de
modelo × estado de calibración por `log_loss` de validación cruzada:

| Modelo | Calibrado | Log Loss (validación cruzada) |
|---|---|---|
| Regresión Logística | No | 0,6045 |
| **Regresión Logística** | **Sí** | **0,5611** |
| Red Neuronal | No | 0,6940 |
| Red Neuronal | Sí | 0,6232 |

La Regresión Logística Calibrada gana, y es el modelo guardado en
`models/churn_model.joblib`. Sobre el conjunto de prueba (tocado una sola
vez, solo para reportar, nunca para decidir): `log_loss=0,5895`,
`brier_score=0,1965`, `roc_auc=0,7985`, `precision=0,6179`, `recall=0,6696`
al umbral `0,2136`.

El modelo final guardado es siempre el ganador real de esta comparación de 4
combinaciones, sin importar cuál sea: los archivos de calibración y matriz
de confusión del ganador se copian a los nombres "canónicos"
(`calibracion_modelo_ganador.csv`, `14_calibracion.png`,
`12_matriz_confusion_sin_calibrar.png`, `13_matriz_confusion_calibrada.png`)
al final de la corrida; el candidato no elegido queda igual en sus propios
archivos sufijados con su nombre, solo para comparación.

Por no exponer `coef_` ni `feature_importances_` de forma consistente entre
los 3 candidatos durante la Fase 1, y porque el modelo final sí es lineal,
la importancia de variables (`14_importancia_features.png`) usa los
coeficientes de la Regresión Logística ganadora. Variables con mayor peso:
`n_inscripciones_total` (negativo — más inscripciones acumuladas, menor
riesgo), `tenure_dias` y `recencia_dias` (positivos — más antigüedad y más
días sin visitar, mayor riesgo), `porcentaje_uso_membresia` (negativo).

Todo el entrenamiento queda registrado en **MLflow** (experimento
`churn_gimnasio`, backend SQLite local en `mlflow.db`).

**Nota sobre reproducibilidad temporal:** `churn_dataset.py` usa la fecha
actual como corte para decidir qué ciclos están `censurado` vs. ya resueltos,
así que volver a correr el pipeline en una fecha distinta puede mover
ligeramente algunos ciclos de `censurado` a `churned` y cambiar las métricas
en un margen pequeño. No es falta de determinismo del modelo: es que el
"hoy" del snapshot avanza.

### 5. Scoring de clientes en riesgo (`churn_detection/modeling/predict.py`)

Puntúa únicamente los ciclos `censurado` cuya `fecha_vencimiento` cae dentro
de una ventana operativa de 7 días (`--window-start`, por defecto
`2026-08-30`) — el equipo de retención trabaja la lista semana a semana, no
el total acumulado de ciclos abiertos.

Los niveles de riesgo (bajo/medio/alto) se calculan por **terciles de la
propia distribución de probabilidades del lote puntuado en esa corrida**, no
con puntos de corte fijos como 0,33/0,66: "alto" es el tercio más riesgoso
de los clientes de ESA corrida, no un valor absoluto de probabilidad. Si el
lote es demasiado chico o uniforme para tres terciles distintos, el proceso
cae a dos niveles (bajo/alto) alrededor de la mediana, con una advertencia
en el log.

`data/processed/predicciones_churn.csv` queda ordenado de mayor a menor
probabilidad de abandono, listo para que el equipo de retención lo use como
lista de priorización: trabajar primero "alto", después "medio", y "bajo"
según la capacidad de contacto disponible.

## Archivos que produce cada etapa

| Comando | Archivos principales |
|---|---|
| `dataset extract-all` | `data/raw/{personas,inscripciones,servicios,registros_acceso,ventas_servicios,pagos_pendientes}.csv`, `extraction_metadata.json` |
| `churn_dataset` | `data/processed/churn_ciclos.csv` |
| `churn_plots` | `reports/figures/08_churn_estado_ciclo.png` … `11_churn_correlacion.png` |
| `modeling.train` | `models/churn_model.joblib`; `data/processed/comparacion_modelos_churn.csv` (Fase 1), `metricas_tuning_ganador.csv` (Fase 2), `sensibilidad_hiperparametros_<modelo>.csv` (Fase 2), `calibracion_<modelo>.csv` + `calibracion_modelo_ganador.csv` (Fase 3); `reports/figures/07_curvas_roc.png`, `14_importancia_features.png`, `12_matriz_confusion_sin_calibrar.png`, `13_matriz_confusion_calibrada.png`, `14_calibracion.png`, y sus equivalentes sufijados por modelo (`15_calibracion_<modelo>.png`, `16_17_matriz_confusion_*_<modelo>.png`) |
| `modeling.predict` | `data/processed/predicciones_churn.csv` |

## Tests

```bash
python -m pytest tests
# o: make test
```

Cubren la capa de acceso a datos, la construcción del dataset de churn, las
figuras de EDA, y el pipeline de entrenamiento/predicción (incluyendo
`WalkForwardGroupSplitter`, el tuning, la calibración y el filtro de ventana
de `predict.py`). El entrenamiento real de la red neuronal hace que la
suite tarde más de un minuto.

## Estructura del proyecto

```
├── Makefile                 <- Atajos: make requirements / test / lint / format / churn-pipeline
├── README.md                <- Este archivo
├── MONOGRAFIA.md             <- Documento metodológico completo del proyecto
├── dvc.yaml / dvc.lock       <- Pipeline versionado (dataset -> EDA -> train -> score)
├── mlflow.db, mlruns/         <- Tracking de experimentos de MLflow (no versionado con DVC)
├── data
│   ├── raw                  <- CSVs extraídos de la base de datos (versionados con `dvc add`)
│   └── processed            <- churn_ciclos.csv, comparaciones, calibraciones, predicciones
├── models                   <- churn_model.joblib (modelo final, calibrado)
├── reports/figures          <- Figuras de EDA, ROC, calibración, matrices de confusión
├── tests                    <- pytest
└── churn_detection           <- Código fuente
    ├── config.py             <- Settings (pydantic-settings), EXCLUDED_PERSONA_IDS,
    │                            RELIABLE_ACCESS_TRACKING_SINCE, config de MLflow
    ├── data/                 <- Acceso a la base de datos (interfaz + implementación Postgres)
    ├── dataset.py            <- CLI: extrae las tablas crudas a data/raw/*.csv
    ├── membership_utils.py   <- Utilidad compartida (día-pass vs. membresía)
    ├── churn_dataset.py      <- Construye data/processed/churn_ciclos.csv
    ├── churn_plots.py        <- Genera las figuras de EDA
    └── modeling
        ├── train.py          <- Entrena, tunea, calibra y selecciona el modelo final
        └── predict.py        <- Puntúa ciclos abiertos dentro de una ventana de fechas
```

## Stack

POO + SOLID + PEP8, `pytest` para tests, `dvc` para versionado de datos y
pipeline, `MLflow` para tracking de experimentos, `scikit-learn` y
`TensorFlow`/`Keras` para los modelos. Servir el modelo por API y
contenerizarlo todavía no están implementados — quedan como siguiente fase.

--------
