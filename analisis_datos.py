import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

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
# La columna price no se utilizará como característica predictora.
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
    index={0: "Por debajo de la mediana", 1: "Igual o por encima de la mediana"}
))

print("\nPrueba:")
print(y_test.value_counts().sort_index().rename(
    index={0: "Por debajo de la mediana", 1: "Igual o por encima de la mediana"}
))

print("\nPreparación y división de los datos completadas correctamente.")