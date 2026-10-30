"""Plantilla: predicción del precio de casas en California.

Completa las partes marcadas con TODO. No cambies la estructura general.
Autores: <Daniel Gracia>, <David Lopez>  (Práctica 1, reutilizado en la Práctica 2)
"""
import time

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

TARGET = "MedHouseVal"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_DEPTH = 5
MODEL_PATH = "modelo_california.pkl"
TREE_IMAGE_PATH = "arbol_decision.png"


def check_nulls(df: pd.DataFrame):
    """Muestra por pantalla el número de valores nulos por columna."""
    print("Valores nulos por columna:")
    print(df.isnull().sum())
    print("Total de nulos:", df.isnull().sum().sum())


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Quita las filas que tengan algún valor nulo con dropna()."""
    print("Filas antes de quitar nulos:", len(df))
    df = df.dropna()
    print("Filas después de quitar nulos:", len(df))
    return df


def plot_decision_tree(X, y, max_depth=None):
    """Entrena un árbol de decisión y lo dibuja con plot_tree()."""
    # Comprobamos max_depth: si no viene o es muy grande, usamos 3
    if max_depth is None:
        print("No se ha indicado max_depth, se usa 3")
        max_depth = 3
    elif max_depth > 6:
        print("max_depth es demasiado grande, se usa 3")
        max_depth = 3

    # Entrenamos el árbol
    arbol = DecisionTreeRegressor(max_depth=max_depth, random_state=RANDOM_STATE)
    arbol.fit(X, y)

    # Lo dibujamos y guardamos la imagen
    plt.figure(figsize=(24, 10))
    plot_tree(arbol, feature_names=list(X.columns), filled=True, rounded=True, fontsize=9)
    plt.title("Árbol de decisión - California Housing")
    plt.savefig(TREE_IMAGE_PATH)
    plt.close()
    print("Imagen del árbol guardada en", TREE_IMAGE_PATH)


def compute_errors(y_true, y_pred, print_errors: bool = True) -> tuple[float, float]:
    """Calcula el MAE y el MSE entre los valores reales y las predicciones.

    Si `print_errors` es True (por defecto), los muestra por pantalla.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    if print_errors:
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
    return mae, mse


def save_model(model, path: str = MODEL_PATH):
    """Guarda el modelo entrenado en `path` usando joblib.dump()."""
    joblib.dump(model, path)
    print("Modelo guardado en", path)


def validate_data():
    """Exploración y validación de los datos."""
    df = fetch_california_housing(as_frame=True).frame
    check_nulls(df)
    df = handle_nulls(df)


def build_model():
    """Carga los datos, entrena el modelo, muestra sus errores y lo devuelve."""
    df = fetch_california_housing(as_frame=True).frame

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    compute_errors(y_test, y_pred)

    return model


def compare_models():
    """Ejercicio 6: compara una regresión lineal con un árbol de decisión."""
    df = fetch_california_housing(as_frame=True).frame
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Modelo 1: regresión lineal
    inicio = time.time()
    lineal = LinearRegression()
    lineal.fit(X_train, y_train)
    print("Regresión lineal - tiempo de entrenamiento:", round(time.time() - inicio, 4), "s")
    compute_errors(y_test, lineal.predict(X_test))

    # Modelo 2: árbol de decisión
    inicio = time.time()
    arbol = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    arbol.fit(X_train, y_train)
    print("Árbol de decisión - tiempo de entrenamiento:", round(time.time() - inicio, 4), "s")
    compute_errors(y_test, arbol.predict(X_test))


def plot_data():
    """Visualización: dibuja un árbol de decisión pequeño."""
    df = fetch_california_housing(as_frame=True).frame
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    plot_decision_tree(X, y, max_depth=3)


def main():
    model = build_model()
    save_model(model)


if __name__ == "__main__":
    print("--- Ejercicio 3 ---")
    validate_data()
    print("\n--- Ejercicios 4 y 5 ---")
    plot_data()
    print("\n--- Ejercicio 6 ---")
    compare_models()
    print("\n--- Ejercicio 7 ---")
    main()
