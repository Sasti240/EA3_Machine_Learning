import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from datetime import date
from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Identificación del proyecto
pregunta_negocio = (
    "¿Qué características permiten identificar viviendas cuyo "
    "precio está por encima de la mediana del mercado?"
)

print("=" * 60)
print("PROYECTO DE MACHINE LEARNING - EA3")
print("=" * 60)

print("\nPregunta de negocio:")
print(pregunta_negocio)

# 2. Cargar automáticamente el dataset desde Hugging Face
print("\n--- CARGA DEL DATASET ---")

dataset = load_dataset(
    "inria-soda/tabular-benchmark",
    data_files="reg_cat/house_sales.csv",
    split="train"
)

# 3. Convertir los datos a un DataFrame de Pandas
df = dataset.to_pandas()

print(f"Registros: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")

# 4. Revisar la calidad de los datos
print("\n--- REVISION DE LOS DATOS ---")

valores_faltantes = int(df.isnull().sum().sum())
duplicados = int(df.duplicated().sum())

print(f"Valores faltantes: {valores_faltantes}")
print(f"Registros duplicados: {duplicados}")

# 5. Separar las características y la variable objetivo
X = df.drop(columns=["price"])
precio = df["price"]

print("\n--- CARACTERISTICAS DEL MODELO ---")
print(f"Número de características: {X.shape[1]}")
print(f"Columnas utilizadas: {list(X.columns)}")

# 6. Dividir los datos en entrenamiento y prueba
X_train, X_test, precio_train, precio_test = train_test_split(
    X,
    precio,
    test_size=0.20,
    random_state=42
)

print("\n--- DIVISION DE LOS DATOS ---")
print(f"Registros de entrenamiento: {len(X_train)}")
print(f"Registros de prueba: {len(X_test)}")

# 7. Calcular la mediana usando únicamente los datos de entrenamiento
mediana_precio = precio_train.median()

print(f"\nMediana del precio en entrenamiento: {mediana_precio:.6f}")

# 8. Crear las categorías de clasificación
y_train = (precio_train >= mediana_precio).astype(int)
y_test = (precio_test >= mediana_precio).astype(int)

print("\n--- DISTRIBUCION DE LAS CATEGORIAS ---")

print("Entrenamiento:")
print(
    y_train.value_counts().sort_index().rename(
        index={
            0: "Por debajo de la mediana",
            1: "Igual o por encima de la mediana"
        }
    )
)

print("\nPrueba:")
print(
    y_test.value_counts().sort_index().rename(
        index={
            0: "Por debajo de la mediana",
            1: "Igual o por encima de la mediana"
        }
    )
)

# 9. Separar una parte del entrenamiento para validar el modelo
X_entrenamiento, X_validacion, y_entrenamiento, y_validacion = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

# 10. Comparar diferentes profundidades del árbol
print("\n--- SELECCION DE LA PROFUNDIDAD DEL ARBOL ---")

profundidades = [3, 5, 7, 10, None]
resultados_validacion = []

mejor_f1 = -1
mejor_profundidad = None

for profundidad in profundidades:

    modelo_prueba = DecisionTreeClassifier(
        max_depth=profundidad,
        random_state=42
    )

    modelo_prueba.fit(
        X_entrenamiento,
        y_entrenamiento
    )

    predicciones_validacion = modelo_prueba.predict(
        X_validacion
    )

    resultado_f1 = f1_score(
        y_validacion,
        predicciones_validacion
    )

    resultados_validacion.append(
        (profundidad, resultado_f1)
    )

    print(
        f"Profundidad: {profundidad} | "
        f"F1: {resultado_f1:.4f}"
    )

    if resultado_f1 > mejor_f1:
        mejor_f1 = resultado_f1
        mejor_profundidad = profundidad

print(f"\nMejor profundidad seleccionada: {mejor_profundidad}")
print(f"Mejor F1 en validación: {mejor_f1:.4f}")

# 11. Entrenar el modelo definitivo
print("\n--- ENTRENAMIENTO DEL MODELO DEFINITIVO ---")

modelo = DecisionTreeClassifier(
    max_depth=mejor_profundidad,
    random_state=42
)

modelo.fit(X_train, y_train)

print("Modelo entrenado correctamente.")
print(f"Profundidad real del árbol: {modelo.get_depth()}")
print(f"Número de hojas: {modelo.get_n_leaves()}")

# 12. Mostrar la importancia de las características
print("\n--- IMPORTANCIA DE LAS CARACTERISTICAS ---")

importancias = pd.DataFrame({
    "Caracteristica": X.columns,
    "Importancia": modelo.feature_importances_
})

importancias = importancias.sort_values(
    by="Importancia",
    ascending=False
)

print(importancias.to_string(index=False))

# 13. Realizar predicciones con los datos de prueba
print("\n--- EVALUACION DEL MODELO ---")

predicciones = modelo.predict(X_test)

# 14. Calcular las métricas de clasificación
accuracy = accuracy_score(y_test, predicciones)

precision = precision_score(
    y_test,
    predicciones,
    zero_division=0
)

recall = recall_score(
    y_test,
    predicciones,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predicciones,
    zero_division=0
)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# 15. Reporte de clasificación
print("\n--- REPORTE DE CLASIFICACION ---")

reporte = classification_report(
    y_test,
    predicciones,
    target_names=[
        "Por debajo de la mediana",
        "Igual o por encima de la mediana"
    ],
    zero_division=0
)

print(reporte)

# 16. Matriz de confusión
matriz = confusion_matrix(
    y_test,
    predicciones
)

print("\n--- MATRIZ DE CONFUSION ---")
print(matriz)

# 17. Generar gráfica de la matriz de confusión
disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=[
        "Por debajo",
        "Igual o superior"
    ]
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Matriz de confusion - Arbol de decision")
plt.tight_layout()

plt.savefig(
    "matriz_confusion.png",
    dpi=300
)

plt.close()

# 18. Generar gráfica de importancia de características
plt.figure(figsize=(10, 7))

importancias_ordenadas = importancias.sort_values(
    by="Importancia",
    ascending=True
)

plt.barh(
    importancias_ordenadas["Caracteristica"],
    importancias_ordenadas["Importancia"]
)

plt.xlabel("Importancia")
plt.ylabel("Caracteristicas")
plt.title("Importancia de las caracteristicas del modelo")
plt.tight_layout()

plt.savefig(
    "importancia_caracteristicas.png",
    dpi=300
)

plt.close()

# 19. Preparar los resultados para los documentos
fecha_ejecucion = date.today().strftime("%d/%m/%Y")

mejores_caracteristicas = importancias.head(5)

# 20. Generar resumen técnico automáticamente
print("\n--- GENERACION DEL RESUMEN ---")

resultados_profundidades = "\n".join(
    [
        f"Profundidad: {profundidad} | F1: {resultado:.4f}"
        for profundidad, resultado in resultados_validacion
    ]
)

resumen = f"""
============================================================
RESUMEN DEL PROYECTO DE MACHINE LEARNING - EA3
============================================================

Fecha de ejecucion: {fecha_ejecucion}

PREGUNTA DE NEGOCIO
{pregunta_negocio}

DATASET
Repositorio: inria-soda/tabular-benchmark
Archivo: reg_cat/house_sales.csv
Registros: {df.shape[0]}
Columnas originales: {df.shape[1]}

REVISION Y PREPARACION
Valores faltantes: {valores_faltantes}
Registros duplicados: {duplicados}
Caracteristicas predictoras: {X.shape[1]}
Registros de entrenamiento: {len(X_train)}
Registros de prueba: {len(X_test)}
Mediana calculada en entrenamiento: {mediana_precio:.6f}

VARIABLE OBJETIVO
Clase 0: Precio por debajo de la mediana.
Clase 1: Precio igual o superior a la mediana.

MODELO
Tipo de aprendizaje: Supervisado.
Problema: Clasificacion binaria.
Algoritmo: Arbol de decision (DecisionTreeClassifier).
Semilla aleatoria: 42

COMPARACION DE PROFUNDIDADES
{resultados_profundidades}

Profundidad seleccionada: {mejor_profundidad}
F1 de validacion: {mejor_f1:.4f}
Profundidad real: {modelo.get_depth()}
Numero de hojas: {modelo.get_n_leaves()}

METRICAS DE EVALUACION
Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}

MATRIZ DE CONFUSION
{matriz}

CARACTERISTICAS MAS IMPORTANTES
{mejores_caracteristicas.to_string(index=False)}

ARCHIVOS GENERADOS
- matriz_confusion.png
- importancia_caracteristicas.png
- resumen.txt
- hallazgos.md
============================================================
"""

with open("resumen.txt", "w", encoding="utf-8") as archivo:
    archivo.write(resumen.strip())

print("Archivo resumen.txt generado correctamente.")

# 21. Generar el documento de hallazgos automáticamente
print("\n--- GENERACION DEL DOCUMENTO DE HALLAZGOS ---")

filas_caracteristicas = ""

for _, fila in mejores_caracteristicas.iterrows():
    filas_caracteristicas += (
        f"| {fila['Caracteristica']} | "
        f"{fila['Importancia']:.4f} "
        f"({fila['Importancia'] * 100:.2f} %) |\n"
    )

hallazgos = f"""# Hallazgos del proyecto de Machine Learning

**Integrantes:** Santiago Jimenez Otalvaro y Santiago Ramírez Vásquez.

**Fecha de ejecución:** {fecha_ejecucion}

## 1. Pregunta de negocio

{pregunta_negocio}

El propósito es identificar características que ayuden a distinguir viviendas cuyo precio se encuentra igual o por encima de la mediana del conjunto de datos.

## 2. Dataset utilizado

Se utilizó el archivo `house_sales.csv`, disponible en el repositorio de Hugging Face `inria-soda/tabular-benchmark`.

El conjunto contiene {df.shape[0]} registros y {df.shape[1]} columnas originales. Durante la revisión se encontraron {valores_faltantes} valores faltantes y {duplicados} registros duplicados.

Se utilizaron {X.shape[1]} características predictoras. La columna `price` se reservó para construir la variable objetivo y no se incluyó como característica del modelo.

## 3. Preparación de los datos

Los registros se dividieron en un 80 % para entrenamiento y un 20 % para prueba.

La mediana se calculó únicamente con los precios del conjunto de entrenamiento, cuyo valor fue **{mediana_precio:.6f}**.

Se definieron dos categorías:

- **Clase 0:** viviendas con precio por debajo de la mediana.
- **Clase 1:** viviendas con precio igual o superior a la mediana.

## 4. Modelo utilizado

Se implementó un modelo de aprendizaje supervisado mediante un árbol de decisión (`DecisionTreeClassifier`).

Se compararon profundidades de 3, 5, 7, 10 y sin límite. La configuración seleccionada fue una profundidad máxima de **{mejor_profundidad}**, con un F1 de validación de **{mejor_f1:.4f}**.

La selección se realizó utilizando una partición de los datos de entrenamiento. Posteriormente, el modelo definitivo se entrenó con todos los datos de entrenamiento.

## 5. Resultados de la evaluación

El modelo obtuvo los siguientes resultados sobre los datos de prueba:

| Métrica | Resultado |
|---|---:|
| Accuracy | {accuracy:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1-score | {f1:.4f} |

La exactitud fue de **{accuracy * 100:.2f} %**, lo que indica la proporción de clasificaciones correctas sobre las {len(X_test)} viviendas del conjunto de prueba.

### Matriz de confusión

![Matriz de confusión](matriz_confusion.png)

La matriz permite observar los aciertos y errores del modelo en ambas categorías.

### Importancia de las características

Las cinco características con mayor importancia fueron:

| Característica | Importancia |
|---|---:|
{filas_caracteristicas.rstrip()}

![Importancia de las características](importancia_caracteristicas.png)

La latitud (`lat`) y el área habitable (`sqft_living`) fueron las variables con mayor importancia en las decisiones del árbol.

## 6. Respuesta a la pregunta de negocio

Los resultados muestran que la ubicación geográfica y el tamaño del área habitable son características especialmente relevantes para identificar viviendas cuyo precio se encuentra igual o por encima de la mediana del conjunto analizado.

El modelo alcanzó un F1-score de **{f1:.4f}** en los datos de prueba, lo que refleja un equilibrio entre la precisión y la capacidad de identificar viviendas de la categoría positiva.

Esta información puede servir como apoyo inicial para compradores e inversionistas que quieran identificar viviendas que merezcan una evaluación más detallada.

## 7. Limitaciones

- El modelo se basa en los datos disponibles y en las viviendas del conjunto analizado.
- La mediana utilizada corresponde al conjunto de entrenamiento y no representa necesariamente la mediana actual del mercado inmobiliario.
- La importancia de una característica no demuestra una relación de causa y efecto.
- El modelo clasifica viviendas en dos categorías, pero no calcula un precio de venta.
- Sus resultados no sustituyen una valoración profesional de una propiedad.

## 8. Conclusión

El árbol de decisión permitió identificar patrones relacionados con el precio de las viviendas y obtener una exactitud de {accuracy * 100:.2f} % en los datos de prueba.

La ubicación y el área habitable fueron las características más relevantes. Por tanto, el modelo ofrece una herramienta de análisis inicial que puede ayudar a orientar la búsqueda y evaluación de posibles oportunidades inmobiliarias.
"""

with open("hallazgos.md", "w", encoding="utf-8") as archivo:
    archivo.write(hallazgos)

print("Archivo hallazgos.md generado correctamente.")

# 22. Confirmar los archivos generados
print("\n--- ARCHIVOS GENERADOS ---")
print("1. matriz_confusion.png")
print("2. importancia_caracteristicas.png")
print("3. resumen.txt")
print("4. hallazgos.md")

print("\nProyecto ejecutado y documentos generados correctamente.")
