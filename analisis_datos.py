
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# 1. Identificación del proyecto
print("=" * 60)
print("PROYECTO DE MACHINE LEARNING - EA3")
print("=" * 60)

print("\nPregunta de negocio:")
print(
    "¿Qué características permiten identificar viviendas cuyo "
    "precio está por encima de la mediana del mercado?"
)

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

print(f"Valores faltantes: {df.isnull().sum().sum()}")
print(f"Registros duplicados: {df.duplicated().sum()}")

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
print(y_train.value_counts().sort_index().rename(
    index={
        0: "Por debajo de la mediana",
        1: "Igual o por encima de la mediana"
    }
))

print("\nPrueba:")
print(y_test.value_counts().sort_index().rename(
    index={
        0: "Por debajo de la mediana",
        1: "Igual o por encima de la mediana"
    }
))

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
mejor_f1 = -1
mejor_profundidad = None

for profundidad in profundidades:
    modelo_prueba = DecisionTreeClassifier(
        max_depth=profundidad,
        random_state=42
    )

    modelo_prueba.fit(X_entrenamiento, y_entrenamiento)

    predicciones_validacion = modelo_prueba.predict(X_validacion)

    resultado_f1 = f1_score(
        y_validacion,
        predicciones_validacion
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

# 11. Entrenar el modelo definitivo con todos los datos de entrenamiento
print("\n--- ENTRENAMIENTO DEL MODELO DEFINITIVO ---")

modelo = DecisionTreeClassifier(
    max_depth=mejor_profundidad,
    random_state=42
)

modelo.fit(X_train, y_train)

print("Modelo entrenado correctamente.")
print(f"Profundidad real del árbol: {modelo.get_depth()}")
print(f"Número de hojas: {modelo.get_n_leaves()}")

# 12. Mostrar las características más importantes
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

print("\nImplementación y entrenamiento del modelo completados.")
