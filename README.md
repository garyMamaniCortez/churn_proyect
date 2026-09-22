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
split train/test **cronológico** (`ChronologicalSplitter`): train solo
contiene ciclos cuyo resultado se resolvió (`fecha_vencimiento`) antes de una
fecha de corte, test solo ciclos resueltos después. Esto demuestra desempeño
sobre datos genuinamente futuros, algo que un split agrupado por
`persona_id` pero sin orden temporal no garantiza (podía dejar ciclos más
recientes en train que en test). Un mismo cliente puede seguir apareciendo a
ambos lados (su ciclo temprano en train, uno posterior en test) porque
`persona_id` nunca es una feature; ese `persona_id` sí se conserva para que la
validación cruzada interna (selección de modelo y tuning, ver más abajo) no
reparta los ciclos de un mismo cliente entre folds. Todo el entrenamiento
queda con tracking en **MLflow** (experimento `churn_gimnasio`, backend
SQLite local en `mlflow.db`).

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

### Selección de modelo y tuning, ambos por validación cruzada

Los 3 candidatos ya no se comparan ajustándolos una vez y mirando su
desempeño en el test set (eso usaría el test set para elegir un ganador, y
después otra vez para "reportar" ese mismo ganador -- el mismo dato
respaldando dos afirmaciones distintas). En cambio, cada candidato se valida
con `StratifiedGroupKFold` (`cross_validate_candidate`) **solo sobre el
train set**, y el ganador se elige por la métrica media de validación
(`cv_mean_<selection_metric>`). El ajuste sobre todo el train + evaluación en
test que también se ve en los logs es puramente descriptivo (la curva ROC
comparativa, la tabla `comparacion_modelos_churn.csv`), nunca decide nada.

El tuning de hiperparámetros del ganador (`HyperparameterTuner`, que envuelve
`GridSearchCV`) sigue el mismo principio un nivel más abajo: usa el mismo
`StratifiedGroupKFold` agrupado por `persona_id`, y el `refit` también se
decide con la métrica de validación, nunca con el test set. Agrupar por
`persona_id` en ambos casos evita que los folds de validación cruzada
mezclen ciclos del mismo cliente: si eso pasara, una combinación de
hiperparámetros (o un candidato) podría verse mejor solo porque el modelo
memorizó parcialmente a ese cliente, no porque generalice mejor.

**Métrica de selección: `log_loss`, no `roc_auc`.** El objetivo del proyecto
es estimar una *probabilidad* de abandono, no solo ordenar clientes de más a
menos riesgosos. ROC-AUC, PR-AUC y recall miden exclusivamente
discriminación: son invariantes a cualquier recalibración monótona de la
probabilidad predicha, así que un modelo puede tener ROC-AUC alto y seguir
prediciendo "0.95" para clientes que en realidad abandonan el 60% de las
veces. `log_loss` (y `brier_score`, que también se reporta) son *proper
scoring rules*: solo mejoran cuando la probabilidad predicha se acerca a la
tasa real observada, así que elegir el modelo/hiperparámetros que minimizan
`log_loss` en validación favorece directamente el objetivo del proyecto. Ver
"Calibración de probabilidades" más abajo para cómo se verifica y corrige
esto de forma explícita, no solo se selecciona por ello.

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

**Resultados de la corrida con la metodología corregida** (split cronológico,
selección por validación cruzada): Hist Gradient Boosting ganó la comparación
de los 3 candidatos por `log_loss` en CV, aunque el desempeño de discriminación
(Accuracy) fue prácticamente idéntico entre los tres (~0.714-0.715). Tras el
tuning (`StratifiedGroupKFold(n_splits=5)`, 27 combinaciones,
`learning_rate=0.05, max_depth=5, max_iter=100`):

| Métrica | CV media | CV desv. estándar | Test (sin calibrar) |
|---|---|---|---|
| Accuracy | 0.7249 | 0.0148 | 0.7784 |
| Precision | 0.7126 | 0.0097 | 0.7158 |
| Recall | 0.7856 | 0.0389 | 0.6422 |
| F1 | 0.7469 | 0.0191 | 0.6770 |
| ROC-AUC | 0.8040 | 0.0162 | 0.8255 |
| PR-AUC | 0.8038 | 0.0155 | 0.7150 |
| Brier Score | 0.1800 | 0.0074 | 0.1623 |
| Log Loss | 0.5350 | 0.0175 | 0.4996 |

Sobre ese mismo test set, la Red Neuronal (no elegida, por perder en `log_loss`
de CV) obtuvo ROC-AUC 0.8415, PR-AUC 0.6879 y Log Loss 0.4965 -- ligeramente
mejor que Hist Gradient Boosting en esa partición puntual. Esto no es un error
de selección: es la varianza esperable de un único test set frente a un
promedio de validación cruzada, que es justamente la razón de elegir por CV
y no por el resultado de una sola partición (ver `cv_final_detalle_por_fold.csv`
para el detalle por fold).

### Calibración de probabilidades (método elegido por validación cruzada)

Seleccionar por `log_loss`/`brier_score` ayuda a que el ganador tienda a
salir bien calibrado, pero no lo garantiza -- así que se verifica y se
corrige explícitamente, en un paso aparte, antes de guardar el modelo final:

1. Se generan probabilidades **out-of-fold** del modelo ganador ya afinado
   sobre el train set (`cross_val_predict` con el mismo `StratifiedGroupKFold`
   agrupado por `persona_id`). Deben ser out-of-fold y no las predicciones
   in-sample del modelo: un modelo es sistemáticamente más confiado sobre
   datos que ya vio, así que calibrar contra sus propias predicciones
   in-sample solo le enseñaría a un calibrador a reproducir ese exceso de
   confianza, no a corregirlo.
2. **`select_calibrator` elige entre isotónica y sigmoid (Platt scaling) por
   validación cruzada**, no aplica isotónica sin más. La primera versión de
   este paso sí lo hacía, y el resultado real de una corrida fue que la
   calibración isotónica *empeoró* el Brier Score (0.1623 → 0.1635) y el Log
   Loss (0.4996 → 0.5061) en el test set: la tabla de confiabilidad mostró
   que, en el decil de mayor riesgo, isotónica predijo 0.968 cuando la tasa
   observada era 0.797 -- muy pocas observaciones out-of-fold caen en ese
   extremo, y la flexibilidad de isotónica (puede ajustar un escalón
   arbitrario) sobreajustó ese puñado de puntos en vez de generalizar.
   Una segunda versión comparó isotónica vs. sigmoid por **Brier Score**
   cross-validado (agrupado por cliente, repetido 5 veces) y **isotónica
   siguió ganando** (0.1803 vs. 0.1809): el decil problemático es una
   fracción tan pequeña del train set que un promedio de error cuadrático
   casi no lo nota. Solo al cambiar el criterio de comparación a **Log
   Loss** -- que penaliza de forma logarítmica una probabilidad confiada y
   equivocada, exactamente lo que pasa en ese decil -- la comparación se dio
   vuelta (isotónica 0.5399 vs. sigmoid 0.5385) y `select_calibrator` eligió
   sigmoid. En la corrida más reciente, con sigmoid, el Brier Score del test
   set bajó a 0.1607 y el Log Loss a 0.4955 (mejoras reales, no un
   empeoramiento como con isotónica sin más). Es la misma razón por la que
   el proyecto ya usaba Log Loss, y no ROC-AUC, para elegir modelo e
   hiperparámetros (sección de arriba): resultó ser también el criterio
   correcto para elegir el propio método de calibración.
3. El modelo final guardado en `models/churn_model.joblib`
   (`CalibratedChurnModel`) envuelve el pipeline afinado + el calibrador
   elegido: `predict_proba` ya devuelve la probabilidad calibrada, no la cruda.

La evidencia queda en dos artefactos, ambos sobre el test set held-out y
comparando antes/después de calibrar: `reports/figures/14_calibracion.png`
(diagrama de confiabilidad: probabilidad media predicha vs. tasa de churn
observada, por decil) y `data/processed/calibracion_modelo_ganador.csv` (la
misma comparación en tabla). `roc_auc`/`pr_auc` no cambian entre la versión
cruda y la calibrada, como se espera de una transformación monótona (0.8255 /
0.7150 en ambas); `brier_score` y `log_loss` sí mejoraron con sigmoid.

**Bug real encontrado al conectar esto con `predict.py`:** `models/
churn_model.joblib` se genera corriendo `train.py` como script (`python -m
churn_detection.modeling.train`), lo que hace que Python trate esa ejecución
de `train.py` como el módulo `"__main__"` -- por lo que `CalibratedChurnModel`
(y las demás clases propias del archivo) quedan *pickleadas* como si vivieran
en `"__main__"`, no en su ruta real. `predict.py`, corrido después como su
propio proceso (`python -m churn_detection.modeling.predict`), tiene su
*propio* `"__main__"` -- que nunca definió esas clases -- así que un
`joblib.load` directo fallaba con `AttributeError: Can't get attribute
'CalibratedChurnModel' on <module '__main__' ...>`. Se corrigió con
`_register_train_classes_under_main()` en `predict.py`: antes de cargar el
artefacto, registra las clases reales (importadas normalmente) bajo
`sys.modules["__main__"]` de ese proceso, para que la búsqueda de `pickle`
las encuentre. Cubierto por
`test_score_open_cycles_loads_a_calibrated_model_pickled_under_main`, que
reproduce el bug pickleando bajo `"__main__"` a propósito antes de cargar.

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
