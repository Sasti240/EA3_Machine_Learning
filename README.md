# Clasificación de viviendas mediante Machine Learning

## Integrantes
- Santiago Jimenez Otalvaro
- Santiago Ramírez Vásquez

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
- **Columnas:** 18.

El conjunto de datos contiene información de viviendas de King County, Washington, Estados Unidos, incluyendo características como habitaciones, baños, superficie, ubicación, año de construcción y precio.

## Modelo de Machine Learning

Se utilizará un árbol de decisión de clasificación, perteneciente al aprendizaje supervisado.

El modelo buscará clasificar las viviendas en dos categorías según su precio:

- Clase 0: precio inferior a la mediana.
- Clase 1: precio igual o superior a la mediana.

La variable `price` se utilizará para construir la variable objetivo y no formará parte de las variables predictoras.

## Estado del proyecto

En desarrollo.