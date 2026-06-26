# -*- coding: utf-8 -*-
"""
Comparación de la red neuronal propuesta frente a métodos clásicos de
aprendizaje automático para la clasificación de condiciones climáticas
(escala COCO / cobertura nubosa).

Atiende las observaciones de las personas revisoras:
  - Comparación con árbol de decisión, k-NN, regresión logística y SVM.
  - Métricas por clase (precision, recall, F1) y matriz de confusión.
  - Reporte explícito y separado de entrenamiento y prueba.
  - Reproducibilidad: TODAS las semillas fijadas para que las cifras del
    texto y las matrices de confusión provengan de UNA SOLA ejecución.

Uso:
    python comparacion_modelos.py
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

# ----------------------------------------------------------------------
# 1. Reproducibilidad: una sola fuente de aleatoriedad para todo el script
# ----------------------------------------------------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"   # evita diferencias por orden de operaciones
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# 2. Carga y preparación de los datos
# ----------------------------------------------------------------------
DATOS = "basededatos.xlsx"
df = pd.read_excel(DATOS, engine="openpyxl")

X = df.iloc[:, :6].values        # 6 variables meteorológicas
y = df.iloc[:, 6].values         # categoría (cobertura nubosa / COCO)

scaler = StandardScaler()
X = scaler.fit_transform(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
clases = list(encoder.classes_)
n_clases = len(clases)

print("=" * 70)
print("RESUMEN DEL CONJUNTO DE DATOS")
print("=" * 70)
print(f"Registros totales        : {len(df)}")
print(f"Variables de entrada     : {df.columns[:6].tolist()}")
print(f"Variable objetivo        : {df.columns[6]!r}")
print(f"Clases presentes (COCO)  : {clases}")
print(f"Número de clases         : {n_clases}")
print("Distribución por clase   :")
print(df.iloc[:, 6].value_counts().sort_index().to_string())

# División estratificada (misma partición para TODOS los modelos)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
)
print(f"\nTamaño de entrenamiento  : {len(X_train)}")
print(f"Tamaño de prueba         : {len(X_test)}")


def reporta(nombre, y_true_tr, y_pred_tr, y_true_te, y_pred_te):
    """Imprime métricas de entrenamiento y prueba de forma homogénea."""
    print("\n" + "=" * 70)
    print(f"MODELO: {nombre}")
    print("=" * 70)
    acc_tr = accuracy_score(y_true_tr, y_pred_tr)
    acc_te = accuracy_score(y_true_te, y_pred_te)
    f1m_te = f1_score(y_true_te, y_pred_te, average="macro", zero_division=0)
    f1w_te = f1_score(y_true_te, y_pred_te, average="weighted", zero_division=0)
    print(f"Accuracy (entrenamiento) : {acc_tr:.4f}")
    print(f"Accuracy (prueba)        : {acc_te:.4f}")
    print(f"F1 macro    (prueba)     : {f1m_te:.4f}")
    print(f"F1 ponderado (prueba)    : {f1w_te:.4f}")
    print("\nReporte de clasificación (PRUEBA):")
    print(
        classification_report(
            y_true_te,
            y_pred_te,
            target_names=[str(c) for c in clases],
            zero_division=0,
        )
    )
    print("Matriz de confusión (PRUEBA), orden de etiquetas:", clases)
    print(confusion_matrix(y_true_te, y_pred_te))
    return {"modelo": nombre, "acc_train": acc_tr, "acc_test": acc_te,
            "f1_macro": f1m_te, "f1_weighted": f1w_te}


resultados = []

# ----------------------------------------------------------------------
# 3. Red neuronal propuesta
# ----------------------------------------------------------------------
y_train_oh = to_categorical(y_train, num_classes=n_clases)
y_test_oh = to_categorical(y_test, num_classes=n_clases)

modelo_nn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(n_clases, activation="softmax"),
])
modelo_nn.compile(optimizer="AdamW",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
modelo_nn.fit(X_train, y_train_oh, epochs=100, batch_size=16,
              validation_data=(X_test, y_test_oh), verbose=0)
print(f"\nParámetros entrenables de la red neuronal: {modelo_nn.count_params()}")

nn_pred_tr = np.argmax(modelo_nn.predict(X_train, verbose=0), axis=1)
nn_pred_te = np.argmax(modelo_nn.predict(X_test, verbose=0), axis=1)
resultados.append(
    reporta("Red Neuronal Artificial (propuesta)",
            y_train, nn_pred_tr, y_test, nn_pred_te)
)

# ----------------------------------------------------------------------
# 4. Métodos clásicos de aprendizaje automático
# ----------------------------------------------------------------------
modelos_clasicos = {
    "Árbol de Decisión": DecisionTreeClassifier(random_state=SEED),
    "k-Vecinos más Cercanos (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Regresión Logística": LogisticRegression(max_iter=1000, random_state=SEED),
    "Máquina de Vectores de Soporte (RBF)": SVC(kernel="rbf", random_state=SEED),
}

for nombre, clf in modelos_clasicos.items():
    clf.fit(X_train, y_train)
    pred_tr = clf.predict(X_train)
    pred_te = clf.predict(X_test)
    resultados.append(reporta(nombre, y_train, pred_tr, y_test, pred_te))

# ----------------------------------------------------------------------
# 5. Tabla comparativa final
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("TABLA COMPARATIVA (conjunto de prueba)")
print("=" * 70)
tabla = pd.DataFrame(resultados)
tabla = tabla[["modelo", "acc_train", "acc_test", "f1_macro", "f1_weighted"]]
tabla.columns = ["Modelo", "Acc. Entren.", "Acc. Prueba", "F1 macro", "F1 ponderado"]
print(tabla.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ----------------------------------------------------------------------
# 6. Figuras de matrices de confusión de la red neuronal (una sola corrida)
# ----------------------------------------------------------------------
def guarda_matriz(y_true, y_pred, titulo, cmap, archivo):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=clases, yticklabels=clases)
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(archivo, dpi=150)
    plt.close()


guarda_matriz(y_train, nn_pred_tr,
              "Matriz de Confusión - Entrenamiento", "Greens",
              "cm_entrenamiento.png")
guarda_matriz(y_test, nn_pred_te,
              "Matriz de Confusión - Prueba", "Blues",
              "cm_prueba.png")
print("\nFiguras guardadas: cm_entrenamiento.png, cm_prueba.png")
