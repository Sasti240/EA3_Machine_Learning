# Aprendizaje supervisado y no supervisado

## Integrantes

- Santiago Jimenez Otalvaro
- Santiago Ramírez Vásquez

## Descripción del proyecto

Este proyecto utiliza técnicas de Machine Learning para identificar características que permiten clasificar viviendas según su precio, tomando como referencia la mediana del conjunto de datos.

Se desarrolló un modelo de clasificación mediante un árbol de decisión, utilizando Python y la biblioteca Scikit-learn. También se implementó una ejecución automatizada mediante GitHub Actions.

## Pregunta de negocio

¿Qué características de las viviendas permiten identificar aquellas cuyo precio está por encima del precio mediano del mercado, para ayudar a compradores e inversionistas inmobiliarios a identificar posibles oportunidades de inversión?

**Beneficiarios:** compradores e inversionistas inmobiliarios.

**Decisión que apoya:** identificar viviendas que podrían requerir un análisis más detallado antes de tomar una decisión de compra o inversión.

## Dataset

- **Nombre:** House Sales (King County).
- **Fuente:** Hugging Face Datasets.
- **Repositorio:** [inria-soda/tabular-benchmark](https://huggingface.co/datasets/inria-soda/tabular-benchmark).
- **Archivo:** `reg_cat/house_sales.csv`.
- **Registros:** 21.613.
- **Columnas originales:** 18.

El conjunto de datos contiene información de viviendas de King County, Washington, Estados Unidos. Incluye características como habitaciones, baños, superficie, ubicación, año de construcción y precio.

**Justificación de la elección del dataset**

Este conjunto de datos es adecuado para la pregunta de negocio porque contiene información sobre las características físicas y geográficas de las viviendas, además de su precio. Esto permite analizar qué variables tienen mayor relevancia para distinguir las viviendas cuyo precio se encuentra por encima de la mediana.

## Exploración y preparación de los datos

Durante la exploración inicial se obtuvo la siguiente información:

- Dimensiones del DataFrame: 21.613 filas y 18 columnas.
- Tipos de datos: enteros (`int64`) y decimales (`float64`).
- Se visualizaron los nombres de las columnas y los primeros cinco registros.
- Valores faltantes: 0.
- Registros duplicados: 0.
- Características predictoras: 17.

Como no se encontraron valores faltantes ni registros duplicados, no fue necesario eliminar información. Las dimensiones del DataFrame se conservaron después de la revisión.

Las 17 características predictoras utilizadas fueron:

`bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `waterfront`, `grade`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `lat`, `long`, `sqft_living15`, `sqft_lot15`, `date_year`, `date_month` y `date_day`.

### División de los datos

Los datos se dividieron en:

- **80 % para entrenamiento:** 17.290 registros.
- **20 % para prueba:** 4.323 registros.

La mediana se calculó únicamente con los precios del conjunto de entrenamiento, obteniendo un valor de `13.017005`. Esto evita utilizar información del conjunto de prueba para definir el umbral de clasificación.

Se construyó una variable objetivo con dos categorías:

- **Clase 0:** viviendas con precio por debajo de la mediana.
- **Clase 1:** viviendas con precio igual o superior a la mediana.

La columna `price` se utilizó para construir la variable objetivo y fue excluida de las características predictoras, evitando que el modelo utilice directamente el precio para realizar sus predicciones.

### Escalado de variables

No se aplicó normalización ni estandarización a las características. Esto se debe a que los árboles de decisión realizan divisiones mediante condiciones sobre los valores de las variables y no dependen de que todas tengan una escala similar.

Por esta razón, se conservaron los valores originales del conjunto de datos.

## Modelo de Machine Learning

Se implementó un modelo de aprendizaje supervisado mediante un árbol de decisión de clasificación (`DecisionTreeClassifier`).

### Funcionamiento del árbol de decisión

El algoritmo clasifica las viviendas mediante una serie de condiciones que se organizan en forma de árbol. Su funcionamiento se puede resumir en los siguientes pasos:

1. Analiza las características disponibles para encontrar divisiones que permitan separar las categorías de precio.
2. Selecciona las condiciones que mejor separan los registros en cada nodo.
3. Continúa realizando divisiones hasta alcanzar la profundidad establecida o cumplir otros criterios de parada.
4. En los nodos finales, llamados hojas, asigna una categoría según los registros que llegaron a cada uno.
5. Utiliza las condiciones aprendidas durante el entrenamiento para clasificar nuevas viviendas.

### Parámetros y configuración

Se utilizaron los siguientes parámetros principales:

| Parámetro | Valor | Descripción |
|---|---|---|
| `max_depth` | 10 | Limita la profundidad máxima del árbol. |
| `random_state` | 42 | Permite mantener la reproducibilidad de los resultados. |
| `test_size` | 0.20 | Reserva el 20 % de los datos para prueba. |
| `validation_size` | 0.20 | Reserva el 20 % del entrenamiento para validación. |

Se compararon las siguientes profundidades máximas:

- 3
- 5
- 7
- 10
- Sin límite (`None`)

La comparación se realizó utilizando el F1-score del conjunto de validación.

| Profundidad | F1-score de validación |
|---|---:|
| 3 | 0.8306 |
| 5 | 0.8758 |
| 7 | 0.8851 |
| 10 | 0.8879 |
| Sin límite | 0.8626 |

La configuración seleccionada fue una profundidad máxima de 10, con un F1-score de validación de `0.8879`.

El modelo definitivo se entrenó utilizando todos los datos de entrenamiento y posteriormente se evaluó con el conjunto de prueba. El árbol obtenido alcanzó una profundidad real de 10 y 490 hojas.

### Ventajas y limitaciones del algoritmo

**Ventajas:**

- Permite trabajar con variables numéricas sin necesidad de escalarlas.
- Es relativamente sencillo de interpretar.
- Permite identificar qué características tienen mayor importancia para las decisiones del modelo.
- Puede representar relaciones no lineales entre las variables.

**Limitaciones:**

- Puede presentar sobreajuste si alcanza una profundidad excesiva.
- Pequeños cambios en los datos pueden modificar la estructura del árbol.
- La importancia de las características no demuestra relaciones de causa y efecto.
- Su capacidad de generalización depende de la calidad y representatividad de los datos utilizados.

## Resultados

El modelo obtuvo los siguientes resultados sobre los datos de prueba:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0.8846 |
| Precision | 0.8923 |
| Recall | 0.8815 |
| F1-score | 0.8869 |

La exactitud alcanzada fue del **88.46 %**.

### Interpretación de las métricas

- **Accuracy:** indica que el modelo clasificó correctamente el 88.46 % de las viviendas del conjunto de prueba.
- **Precision:** de las viviendas clasificadas como iguales o superiores a la mediana, el 89.23 % realmente pertenecían a esa categoría.
- **Recall:** el modelo identificó correctamente el 88.15 % de las viviendas que realmente estaban en la categoría igual o superior a la mediana.
- **F1-score:** combina precisión y recall, obteniendo un resultado de 0.8869.

Las dos categorías tienen una cantidad de registros relativamente equilibrada en el conjunto de prueba: 2.104 viviendas por debajo de la mediana y 2.219 iguales o superiores. Por eso, además de la exactitud, se utiliza el F1-score para evaluar el equilibrio entre precisión y capacidad de identificación.

En este problema, los falsos positivos representan viviendas que el modelo clasifica como iguales o superiores a la mediana cuando realmente están por debajo. Los falsos negativos representan viviendas de precio igual o superior a la mediana que el modelo no logra identificar.

Ambos errores son importantes para la decisión de compra o inversión, por lo que resulta conveniente analizar conjuntamente precisión y recall.

### Características más importantes

Las cinco características con mayor importancia para el modelo fueron:

| Característica | Importancia |
|---|---:|
| lat | 47.50 % |
| sqft_living | 28.79 % |
| grade | 5.27 % |
| long | 4.85 % |
| sqft_living15 | 2.89 % |

La latitud y el área habitable fueron las características con mayor importancia en las decisiones del árbol.

### Visualizaciones

**Matriz de confusión**

![Matriz de confusión](matriz_confusion.png)

**Importancia de las características**

![Importancia de las características](importancia_caracteristicas.png)

## Hallazgos y conclusiones

Los resultados muestran que la ubicación geográfica y el tamaño del área habitable son características especialmente relevantes para identificar viviendas cuyo precio se encuentra igual o por encima de la mediana del conjunto analizado.

El modelo alcanzó un F1-score de `0.8869` en los datos de prueba, mostrando un buen equilibrio entre precisión y capacidad de identificación de la categoría positiva.

Estos resultados pueden servir como apoyo inicial para orientar la búsqueda y evaluación de posibles oportunidades inmobiliarias. Por ejemplo, se pueden revisar con mayor atención la ubicación y el área habitable de las viviendas identificadas por el modelo, y posteriormente contrastar sus características con las condiciones reales del inmueble y del mercado.

Sin embargo, la importancia de las características no demuestra una relación de causa y efecto, y el modelo no calcula precios de venta ni sustituye una valoración profesional.

Para consultar el análisis completo, se puede revisar el archivo [hallazgos.md](hallazgos.md).

## Estructura del proyecto

```text
EA3_Machine_Learning/
│
├── .github/
│   └── workflows/
│       └── ejecutar_proyecto.yml
│
├── analisis_datos.py
├── requirements.txt
├── .gitignore
├── README.md
├── hallazgos.md
├── resumen.txt
├── matriz_confusion.png
└── importancia_caracteristicas.png
```

## Tecnologías utilizadas

- Python 3.11
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Hugging Face Datasets
- Git y GitHub
- GitHub Actions

## Instalación y ejecución

Para ejecutar el proyecto localmente, se deben seguir estos pasos:

**1. Clonar el repositorio**

```bash
git clone https://github.com/Sasti240/EA3_Machine_Learning.git
```

**2. Ingresar a la carpeta del proyecto**

```bash
cd EA3_Machine_Learning
```

**3. Crear un entorno virtual**

```bash
python -m venv .venv
```

**4. Activar el entorno virtual**

En Windows:

```powershell
.venv\Scripts\Activate.ps1
```

**5. Instalar las dependencias**

```bash
pip install -r requirements.txt
```

**6. Ejecutar el proyecto**

```bash
python analisis_datos.py
```

Al finalizar, se generarán los siguientes archivos:

- `matriz_confusion.png`
- `importancia_caracteristicas.png`
- `resumen.txt`
- `hallazgos.md`

## Automatización con GitHub Actions

El proyecto cuenta con un flujo de trabajo automatizado mediante GitHub Actions, ubicado en `.github/workflows/ejecutar_proyecto.yml`.

La ejecución se activa cuando:

- Se realizan cambios mediante `push`.
- Se crea o actualiza un `pull_request`.
- Se solicita una ejecución manual desde GitHub Actions.

El flujo realiza automáticamente las siguientes tareas:

1. Descarga el repositorio.
2. Configura Python 3.11.
3. Instala las dependencias.
4. Descarga el dataset desde Hugging Face y ejecuta el análisis.
5. Realiza la exploración y preparación de los datos.
6. Entrena y evalúa el modelo.
7. Guarda los archivos generados como artefactos descargables.

Los resultados pueden consultarse desde la pestaña **Actions** del repositorio, seleccionando una ejecución y descargando el artefacto `resultados-machine-learning`.

**Repositorio:** https://github.com/Sasti240/EA3_Machine_Learning

## Estado del proyecto

**Finalizado.** El modelo fue entrenado y evaluado, los documentos y visualizaciones fueron generados y la ejecución automatizada mediante GitHub Actions fue comprobada correctamente.