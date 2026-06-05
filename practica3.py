# ==========================================
# PRACTICA 3 - LOGISTICA GLOBAL
# ==========================================

# LIBRERIAS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind

# ==========================================
# GENERACION DE DATOS
# ==========================================

np.random.seed(123)

n = 500

distancia_km = np.random.normal(
    1000,
    250,
    n
)

tiempo_entrega = (
    distancia_km / 100
) + np.random.normal(
    0,
    2,
    n
)

costo = (
    distancia_km * 0.5
) + np.random.normal(
    0,
    50,
    n
)

tipo_transporte = np.random.choice(
    ["Terrestre", "Aereo"],
    n
)

# ==========================================
# DATAFRAME
# ==========================================

df = pd.DataFrame({
    "distancia_km": distancia_km,
    "tiempo_entrega": tiempo_entrega,
    "costo": costo,
    "tipo_transporte": tipo_transporte
})

# ==========================================
# EXPLORACION
# ==========================================

print("\nPRIMERAS FILAS")
print(df.head())

print("\nESTADISTICAS")
print(df.describe())

# ==========================================
# CREAR VALORES NULOS
# ==========================================

df.loc[0:15, "distancia_km"] = np.nan

print("\nVALORES NULOS ANTES")
print(df.isnull().sum())

# ==========================================
# IMPUTACION CON MEDIANA
# ==========================================

mediana = df["distancia_km"].median()

df["distancia_km"] = df[
    "distancia_km"
].fillna(mediana)

print("\nVALORES NULOS DESPUES")
print(df.isnull().sum())

# ==========================================
# ESTADISTICAS
# ==========================================

print("\nMEDIA TIEMPO ENTREGA")
print(
    df["tiempo_entrega"].mean()
)

print("\nDESVIACION TIEMPO ENTREGA")
print(
    df["tiempo_entrega"].std()
)

# ==========================================
# CORRELACION
# ==========================================

correlacion = df[
    "distancia_km"
].corr(
    df["tiempo_entrega"]
)

print("\nCORRELACION")
print(correlacion)

# ==========================================
# AGRUPACION
# ==========================================

agrupacion = df.groupby(
    "tipo_transporte"
).agg({
    "tiempo_entrega":"mean",
    "costo":"mean"
})

print("\nPROMEDIOS POR TRANSPORTE")
print(agrupacion)

# ==========================================
# T TEST
# ==========================================

terrestre = df[
    df["tipo_transporte"]
    == "Terrestre"
]["tiempo_entrega"]

aereo = df[
    df["tipo_transporte"]
    == "Aereo"
]["tiempo_entrega"]

t, p = ttest_ind(
    terrestre,
    aereo
)

print("\nT TEST")

print("t =", t)
print("p =", p)

if p < 0.05:
    print(
        "Existe diferencia significativa"
    )
else:
    print(
        "No existe diferencia significativa"
    )

# ==========================================
# SCATTER PLOT
# ==========================================

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="distancia_km",
    y="tiempo_entrega",
    data=df
)

plt.title(
    "Distancia vs Tiempo Entrega"
)

plt.show()

# ==========================================
# BOXPLOT
# ==========================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x="tipo_transporte",
    y="tiempo_entrega",
    data=df
)

plt.title(
    "Tiempo Entrega por Transporte"
)

plt.show()

# ==========================================
# CONCLUSION
# ==========================================

print("\nCONCLUSION")

if correlacion > 0.5:
    print(
        "La distancia influye "
        "considerablemente en el tiempo."
    )
else:
    print(
        "La influencia parece moderada."
    )
    