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

**Corrección adicional:** el EDA mostró que `n_ventas` correlaciona 0.99 con
`n_inscripciones_total` (cada venta de servicio corresponde casi 1 a 1 con una
inscripción), así que se sacó de la tabla para no duplicar el peso de esa señal
en el clustering. `n_inscripciones_total` ya la representa.

## Pipeline de DVC (datos, features y EDA)

En vez de versionar archivos sueltos con `dvc add`, el proyecto usa un
pipeline declarado en `dvc.yaml` con dos etapas:

- **`build_features`**: corre `python -m churn_detection.features`, depende
  del código (`features.py`, `config.py`) y de los 6 CSV crudos, y produce
  `data/processed/clientes_segmentacion.csv`.
- **`eda`**: corre `python -m churn_detection.plots`, depende de `plots.py` y
  del CSV de segmentación, y produce las 5 figuras en `reports/figures/`.

Esto significa que las figuras quedan atadas por hash tanto al dataset como al
código que las generó: si cambia cualquiera de los dos, `dvc status` lo va a
marcar como desactualizado.

```powershell
dvc repro
```

Regenera solo lo que cambió (o todo, la primera vez) y actualiza `dvc.lock`.
Después, para versionar:

```powershell
git add dvc.yaml dvc.lock data/processed/.gitignore reports/figures/.gitignore
git commit -m "data: pipeline de segmentacion + eda"
dvc push   # requiere tener un remote configurado (ver mas abajo)
```

Los datos crudos (`data/raw/*.csv`) siguen versionados aparte con `dvc add`
como en la fase anterior, y el pipeline los referencia como dependencias de
solo lectura.

## Hallazgos del EDA (2026-09-01, 4,434 clientes)

- **`n_inscripciones_total` y `n_ventas` correlacionaban 0.99** entre sí, señal
  de que eran casi la misma variable; se sacó `n_ventas` de la tabla (ver más
  arriba).
- **`recencia_dias` y `monto_total_gastado` están fuertemente sesgados a la
  derecha**: la mayoría de los clientes están concentrados en valores bajos
  (visitaron hace poco, gastan poco) con una cola larga de pocos clientes con
  valores altos. Esto es relevante para la etapa de clustering: probablemente
  convenga escalar o transformar (por ejemplo log) estas variables antes de
  correr K-Means, que es sensible a la escala y a los outliers.
- **`es_multisucursal` y `tiene_pago_pendiente` tienen varianza casi nula**
  (menos del 4% y menos del 1% de los clientes en `True`, respectivamente).
  Aportan poco como variable de clustering tal cual están; podrían mantenerse
  como atributo descriptivo de cada segmento en vez de como input del modelo.
- **La asistencia está fuertemente concentrada de lunes a viernes**, con muy
  poca actividad los sábados y prácticamente nula los domingos, consistente
  con el horario de atención del gimnasio.
- El dispersograma de `porcentaje_uso_membresia` vs. `recencia_dias` muestra
  una relación negativa razonable, más días sin venir tienden a asociarse con
  menor uso de la membresía, pero con dispersión suficiente como para esperar
  más de un segmento dentro de cada nivel de recencia.

## Intento adicional: check-ins relativos al propio abandono (2026-09-02)

El usuario preguntó si convenía medir `n_checkins_ultimos_30d` en relación al
comportamiento propio del cliente antes de irse, en vez de siempre en relación
a "hoy" para todos por igual. Se verificó empíricamente: dentro del cluster
"frenado", `n_checkins_ultimos_30d` (desde hoy) tiene mediana 0, prácticamente
sin varianza, mientras que la misma ventana de 30 días medida desde el
**último check-in propio de cada cliente** tiene mediana 5 y llega hasta 25.
Para los clientes activos, ambas versiones correlacionan 0.84 entre sí (casi
lo mismo, tiene sentido: su último check-in es reciente).

Se agregó `checkins_ultimo_mes_activo` a `features.py` con esa lógica. Al
meterla en el clustering, sin embargo, **bajó el silhouette de 0.273 a 0.257**:
correlaciona 0.87 con `porcentaje_uso_membresia` (ambas miden, en el fondo,
"qué tan intenso era este cliente mientras estaba activo"), así que sumarla
solo duplicaba peso en una dimensión que ya estaba representada, en vez de
aportar una dimensión nueva. Se sacó de `CLUSTER_FEATURES` pero se dejó en la
tabla de salida como columna descriptiva: no ayuda a *definir* los clusters,
pero sí sirve para *explicarlos* después (por ejemplo, comparar qué tan
intensos eran los clientes de cada cluster en su último mes activo).

## Enriquecimiento de features (2026-09-02): variables de tendencia y regularidad

Las 8 features originales solo describían "nivel" (cuánto, qué tan seguido),
no "dirección". Se agregaron 3 features nuevas a `features.py` para capturar
eso:

- **`ratio_actividad_reciente`**: check-ins de los últimos 30 días contra el
  ritmo semanal histórico del propio cliente. `<1` significa que está
  frenando, `>1` que está acelerando. Dos clientes con la misma frecuencia
  promedio pueden estar yendo en direcciones opuestas, y eso era invisible
  antes.
- **`cv_gap_visitas`**: coeficiente de variación de los días entre visitas
  consecutivas. Mide qué tan regular es el ritmo de asistencia, no cuán
  seguido viene. Requiere al menos 3 check-ins; si no, `NaN`.
- **`monto_gastado_ultimos_90d`**: gasto en los últimos 90 días, análogo a los
  check-ins recientes pero para dinero. No depende de
  `RELIABLE_ACCESS_TRACKING_SINCE` porque las ventas no pasan por el módulo de
  accesos que falló.

**Bug encontrado y corregido en el camino:** al construir `ratio_actividad_reciente`
se detectó que `n_checkins_ultimos_30d`/`90d` quedaban en `NaN` en vez de `0`
para cualquier cliente sin check-ins en esa ventana, porque el `groupby` sobre
esa ventana simplemente omite a esos clientes en lugar de darles una fila en
cero. El `fillna(0)` que ya existía corría demasiado tarde para evitar que ese
`NaN` se propagara. Esto afectaba silenciosamente al dataset de la fase
anterior; ya está corregido y cubierto con test.

Con las 3 features nuevas, el clustering se re-corrió comparando K-Means, GMM
y jerárquico (11 features en total, ver `CLUSTER_FEATURES` en
`segmentation.py`): el mejor silhouette subió de **0.228 a 0.273** (jerárquico,
k=2), y los tres algoritmos ahora coinciden mucho más claramente en que k=2 es
la mejor opción, con una caída más nítida después (ver
`reports/figures/06_seleccion_k.png`). Sigue siendo una separación débil, no
fuerte, pero es una mejora real y no un empate como antes.

Perfil actualizado de los 2 clusters (jerárquico, ganador):

| | Cluster 0 (80%, 3,540) | Cluster 1 (20%, 894) |
|---|---|---|
| Recencia (días desde último check-in) | 70 | 8 |
| Ritmo reciente vs. histórico | 0.11 (muy frenado) | 0.91 (estable) |
| % uso membresía | 22% | 35% |
| Gasto últimos 90 días | $58 | $269 |
| Inscripciones totales | 1.43 | 3.93 |

**Nota de diseño:** se evaluó agregar el `cluster` de la segmentación como
feature del dataset de churn, pero se decidió NO hacerlo por ahora. Pegar el
cluster de `clientes_segmentados.csv` (calculado con el comportamiento
completo hasta hoy) a un ciclo de membresía vencido hace meses filtraría
información futura relativa al corte de ese ciclo. Hacerlo bien requeriría
reentrenar con K-Means (el único de los tres algoritmos que puede asignar
cluster a un punto nuevo con solo sus features, ya que el ganador por
silhouette fue jerárquico, que no generaliza a datos nuevos) y recalcular un
cluster por cada fila usando solo sus propias features. Se dejan segmentación
y churn como dos análisis complementarios pero independientes; revisar esta
decisión si más adelante se justifica el trabajo extra.

## Dataset de churn: una fila por ciclo de membresía, no por cliente (2026-09-02)

`churn_detection/churn_dataset.py` (`ChurnCycleDatasetBuilder`) construye el
dataset de entrenamiento para el modelo de churn, con una arquitectura
distinta a la de segmentación:

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

**Resultado real (2026-09-02, `CHURN_GRACE_DAYS=30`):** 8,573 ciclos de
membresía, de los cuales 6,718 (78.4%) tienen resultado conocido: 3,548
renovaron, 3,170 no. **Tasa de churn entre los ciclos con resultado: 47.2%**,
un número mucho más razonable que el 76.6% que había dado el enfoque anterior
(snapshot a "hoy" con sesgo de cohorte), justamente porque ahora cada ciclo se
evalúa en su propio momento, no todos contra la misma fecha actual.

Como control de calidad: comparando el promedio de las features entre ciclos
`renovado` y `churned`, todas las diferencias van en la dirección esperada
(los que abandonan tienen menos antigüedad, menos uso de membresía, menos
ritmo reciente y menos gasto), lo cual es una señal de que el dataset es
coherente antes de empezar a modelar.

Salida: `data/processed/churn_ciclos.csv`, con columnas de identidad
(`persona_id`, `inscripcion_id`, fechas), `estado_ciclo`, `churn_label`, y las
mismas 11 features de comportamiento que en segmentación (calculadas de forma
independiente, respetando el corte de cada fila).

## Segmentación: comparación de algoritmos (2026-09-01)

Con K-Means solo, el mejor silhouette fue 0.228 en k=2, un valor débil (por
debajo de 0.25 se considera que no hay evidencia sólida de clusters bien
separados). Para descartar que fuera una limitación propia de K-Means, se
agregó `churn_detection/modeling/segmentation.py` con **Gaussian Mixture** y
**clustering jerárquico (Ward)**, comparables en la misma tabla vía
`compare_algorithms()`.

Resultado con las 8 features originales: los tres coincidían en que k=2 era la
mejor opción disponible, sin separación fuerte. Ver la sección de arriba para
el resultado actualizado tras agregar las features de tendencia.

El split de k=2 queda guardado en `data/processed/clientes_segmentados.csv`
(persona + `cluster` + `datos_incompletos`) y perfilado en
`data/processed/perfil_clusters.csv`. La columna `comparacion_algoritmos.csv`
deja el detalle completo (silhouette y BIC por algoritmo y k).

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
    ├── membership_utils.py     <- Shared: flag_membership_services (día-pass vs. membresía)
    │
    ├── churn_dataset.py        <- ChurnCycleDatasetBuilder: builds data/processed/churn_ciclos.csv
    │                              (una fila por ciclo de membresía, sin fuga de datos)
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
    └── plots.py                <- EDAFigureGenerator: builds reports/figures/*.png
                                    (nulos, distribuciones, correlación, categóricas,
                                    uso vs. recencia) from data/processed/clientes_segmentacion.csv
```

## Stack (según lo definido)

POO + SOLID + PEP8, `pytest` para tests, `dvc` para versionado de datos, `MLflow`
para tracking de experimentos, `FastAPI` para servir el modelo y `Docker` para
contenerizar. Por ahora solo está implementada la capa de extracción de datos
(esta fase); MLflow, FastAPI y Docker se agregan cuando lleguemos a modelado y
despliegue, para no acumular dependencias/código sin uso.

--------

