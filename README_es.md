*Leer este documento en otros idiomas: [English](README.md)*

# Customer Sales - Pipeline Avanzado de Clustering y Segmentación

Este proyecto implementa un pipeline de Machine Learning integral y reproducible que segmenta a los clientes de un retailer online real a partir de su historial de transacciones. Las transacciones se convierten en variables de comportamiento a nivel cliente (RFM y más), se procesan con transformadores personalizados de Scikit-Learn, se agrupan con un algoritmo seleccionado y ajustado automáticamente, y se validan **contra el futuro**: los segmentos se construyen con datos hasta una fecha de corte y luego se contrastan con lo que los clientes hicieron realmente en los meses siguientes.

---

# Acerca del Conjunto de Datos

**Nombre:** [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail) — UCI Machine Learning Repository  
**Cita:** Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33  
**Licencia:** CC BY 4.0  
**Tipo de Problema:** Aprendizaje No Supervisado (Clustering / Segmentación de Clientes)

## Características del Dataset

- **Tamaño:** 541.909 líneas de factura entre el 01/12/2010 y el 09/12/2011.
- **Negocio:** retailer online del Reino Unido que vende artículos de regalo; muchos de sus clientes son mayoristas.
- **Columnas:** `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`.
- **Formato:** el archivo Excel original se guarda como CSV comprimido (`data/online_retail.csv.gz`) con las mismas columnas.

---

# Objetivo del Proyecto

Darle a marketing una respuesta accionable a *qué grupos de clientes existen, cuánto vale cada uno y qué hacer con cada uno*, con una segmentación reproducible, libre de fuga de información y validada con datos que el modelo nunca vio.

## Capacidades Principales

- Limpieza de transacciones con un registro explícito de cada regla, y cancelaciones descontadas del gasto (revenue neto).
- Diseño temporal: variables construidas en una ventana de observación y validación en una ventana de resultado posterior.
- Construcción de variables a nivel cliente (RFM + antigüedad + variedad de productos) y transformadores personalizados de Scikit-Learn.
- Preprocesamiento consciente de la asimetría: `log1p`, recorte IQR y estandarización dentro de un único pipeline.
- Selección del número de clusters basada en datos (método del codo + consenso entre Silhouette, Calinski-Harabasz y Davies-Bouldin).
- Benchmarking automatizado (K-Means, Gaussian Mixture, Agglomerative) y ajuste con Optuna adoptado solo si supera a la línea base.
- Análisis de estabilidad, validación temporal y perfiles de negocio.

---

# Preparación de Datos

## Limpieza de Transacciones

| Paso | Filas | Eliminadas |
|---|---|---|
| Transacciones originales | 541.909 | - |
| Sin `customer_id` faltante | 406.829 | 135.080 |
| Sin duplicados exactos | 401.604 | 5.225 |
| Sin precios no positivos / códigos que no son productos (envío, comisiones, ajustes) | 399.656 | 1.948 |

Las 8.506 líneas de cancelación **se conservan como revenue negativo** en lugar de eliminarse: algunos pedidos grandes se cancelaron justo después de realizarse, y eliminar solo la cancelación dejaría una compra que nunca ocurrió. Las cancelaciones equivalen al 5,4% del revenue bruto.

## Diseño Temporal

- **Ventana de observación (01/12/2010 - 31/08/2011):** los únicos datos usados para construir variables y entrenar la segmentación. En ella compraron 3.314 clientes; 9 tuvieron un revenue neto no positivo (cancelaron todo) y se excluyen, quedando **3.305 clientes**.
- **Ventana de resultado (01/09/2011 - 09/12/2011):** el modelo nunca la ve; se usa para comprobar si los segmentos anticipan la recompra y el revenue.

## Variables de Cliente

| Variable | Definición |
|---|---|
| `recency` | Días desde la última compra (a la fecha de corte) |
| `frequency` | Número de facturas distintas |
| `monetary` | Revenue neto en GBP (compras menos cancelaciones) |
| `tenure` | Días desde la primera compra |
| `n_products` | Número de productos distintos comprados |
| `avg_order_value` | `monetary / frequency` (creada por `FeatureEngineer`) |
| `orders_per_month` | `frequency / max(tenure / 30, 1)` (creada por `FeatureEngineer`) |

`country` queda fuera de la distancia (más del 90% de los clientes está en el Reino Unido y el objetivo es una segmentación de comportamiento) y se usa para el perfilado.

---

# Transformadores Personalizados y Preprocesamiento

Para garantizar un flujo de trabajo reproducible y libre de fuga de información, la lógica de preprocesamiento se encapsula en clases personalizadas que heredan de `BaseEstimator` y `TransformerMixin`.

- **FeatureEngineer:** genera `avg_order_value` y `orders_per_month` a partir de la tabla de clientes.
- **IQRTransformer:** aprende límites robustos (método IQR) en `fit` y recorta los valores en `transform`.
- **RareCategoryEncoder:** agrupa categorías poco frecuentes en `Other`; se usa para perfilar el mix de mercados de cada segmento.

El pipeline numérico aplica imputación por mediana, `log1p` (las distancias deben reflejar diferencias relativas de gasto, no absolutas), recorte IQR y estandarización.

---

# Modelado

## Número Óptimo de Clusters

Se entrena K-Means para `k = 2..8`. Cada índice interno tiene su propio sesgo, por lo que se elige el `k` con el menor ranking promedio entre Silhouette, Calinski-Harabasz y Davies-Bouldin. El consenso es **k = 3**.

## Benchmark

| Modelo | Silhouette | Calinski-Harabasz | Davies-Bouldin |
|---|---|---|---|
| **K-Means** | **0,321** | **1561,8** | **1,150** |
| Agglomerative | 0,273 | 1200,3 | 1,261 |
| Gaussian Mixture | 0,152 | 798,7 | 2,225 |

## Optimización de Hiperparámetros (Optuna)

Optuna (sampler TPE con semilla fija) ajusta los hiperparámetros del modelo ganador con `k` fijo, maximizando una Silhouette penalizada ante soluciones degeneradas o muy desbalanceadas. La configuración ajustada se adopta **solo si supera a la línea base por al menos 0,005**. En este caso el mejor trial (`n_init=25`, `init='k-means++'`, 0,3212) no mejoró de forma relevante a la línea base (0,3211), así que se mantuvo la configuración base, más simple.

---

# Resultados

| Control | Resultado |
|---|---|
| Silhouette / Calinski-Harabasz / Davies-Bouldin | 0,321 / 1561,8 / 1,150 |
| Estabilidad (ARI medio en 10 submuestras del 80%) | 0,9824 (mínimo 0,9375) |
| Tasa de recompra global en la ventana de resultado | 58,8% |

| Cluster | Perfil | Clientes | Perfil mediano | Volvió a comprar el trimestre siguiente | Participación en el revenue neto del trimestre siguiente |
|---|---|---|---|---|---|
| 0 | Núcleo leal | 1.117 (33,8%) | 26 días desde la última compra, 5 pedidos, GBP 1.749 | 84,2% | 77,7% |
| 1 | Clientes nuevos | 549 (16,6%) | Activos hace 48 días, 1 pedido, GBP 330 | 51,2% | 8,2% |
| 2 | Compradores únicos inactivos | 1.639 (49,6%) | 143 días desde la última compra, 1 pedido, GBP 312 | 44,1% | 14,1% |

Un tercio de los clientes genera más de tres cuartas partes del revenue del trimestre siguiente. La Silhouette es moderada (algo típico en datos de comportamiento continuos), pero los segmentos separan con claridad el comportamiento futuro y son estables entre submuestras.

---

# Estructura del Proyecto

```
├── data/
│   └── online_retail.csv.gz
├── notebooks/
│   └── Customer Sales - Advanced Clustering.ipynb
├── README.md
├── README_es.md
└── requirements.txt
```

# Cómo Ejecutarlo

```bash
git clone https://github.com/arguar13/Customer-Sales-Advanced-Clustering.git
cd Customer-Sales-Advanced-Clustering
pip install -r requirements.txt
jupyter notebook "notebooks/Customer Sales - Advanced Clustering.ipynb"
```

La ruta de los datos se resuelve de forma relativa al repositorio, por lo que el notebook funciona tanto desde la raíz del proyecto como desde la carpeta `notebooks/`.

---

# Licencia

El código se ofrece con fines educativos y de portafolio. El dataset pertenece a sus autores y se distribuye a través del UCI Machine Learning Repository bajo la licencia CC BY 4.0.

---

# Autor

**Armando Guarnera**  
Data Scientist — Argentina
