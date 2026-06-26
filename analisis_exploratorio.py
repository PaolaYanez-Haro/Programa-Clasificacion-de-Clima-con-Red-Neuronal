# -*- coding: utf-8 -*-
"""
Análisis exploratorio del conjunto de datos de condiciones climáticas.

Atiende la observación de ambas personas revisoras sobre la necesidad de
describir el conjunto de datos e incorporar visualizaciones sencillas
(distribuciones, correlaciones y frecuencias por categoría).

Genera:
  - estadisticas_descriptivas.csv : resumen estadístico de las variables
  - distribucion_clases.png       : frecuencia por categoría COCO
  - matriz_correlacion.png        : correlación entre variables de entrada

Uso:
    python analisis_exploratorio.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATOS = "basededatos.xlsx"
df = pd.read_excel(DATOS, engine="openpyxl")
df.columns = [c.strip() for c in df.columns]

variables = list(df.columns[:6])
objetivo = df.columns[6]

# ----------------------------------------------------------------------
# Estadísticas descriptivas
# ----------------------------------------------------------------------
desc = df[variables].describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
desc = desc.round(2)
desc.to_csv("estadisticas_descriptivas.csv")
print("Estadísticas descriptivas de las variables de entrada:")
print(desc.to_string())

print(f"\nRegistros totales: {len(df)}")
print("\nDistribución de la variable objetivo (código COCO):")
print(df[objetivo].value_counts().sort_index().to_string())

# ----------------------------------------------------------------------
# Figura 1: distribución de clases
# ----------------------------------------------------------------------
conteo = df[objetivo].value_counts().sort_index()
plt.figure(figsize=(6, 4))
ax = sns.barplot(x=conteo.index.astype(int), y=conteo.values, color="#4C72B0")
for i, v in enumerate(conteo.values):
    ax.text(i, v + 0.8, str(v), ha="center", fontsize=10)
plt.xlabel("Código COCO")
plt.ylabel("Número de registros")
plt.title("Distribución de registros por categoría COCO")
plt.tight_layout()
plt.savefig("distribucion_clases.png", dpi=150)
plt.close()

# ----------------------------------------------------------------------
# Figura 2: matriz de correlación entre variables de entrada
# ----------------------------------------------------------------------
plt.figure(figsize=(6, 5))
corr = df[variables].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, cbar_kws={"shrink": 0.8})
plt.title("Matriz de correlación de las variables de entrada")
plt.tight_layout()
plt.savefig("matriz_correlacion.png", dpi=150)
plt.close()

print("\nFiguras guardadas: distribucion_clases.png, matriz_correlacion.png")
