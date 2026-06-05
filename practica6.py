# ==========================================
# PRACTICA 6
# SMART CITIES - PCA
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# GENERACION DE DATOS
# ==========================================

np.random.seed(456)

n = 400

datos_urbanos = pd.DataFrame({
    "temp_ambiente": np.random.normal(28, 4, n),
    "humedad_rel": np.random.normal(60, 10, n),
    "densidad_vehiculos": np.random.normal(120, 30, n),
    "velocidad_viento": np.random.normal(15, 5, n),
    "radiacion_solar": np.random.normal(800, 100, n),
    "conteo_peatones": np.random.normal(50, 15, n)
})

# Correlaciones urbanas

datos_urbanos["co2_ppm"] = (
    datos_urbanos["densidad_vehiculos"] * 3.5
    + np.random.normal(300, 50, n)
)

datos_urbanos["no2_ppb"] = (
    datos_urbanos["co2_ppm"] * 0.1
    + np.random.normal(5, 2, n)
)

datos_urbanos["particulas_pm25"] = (
    datos_urbanos["co2_ppm"] * 0.05
    + np.random.normal(10, 3, n)
)

datos_urbanos["nivel_ruido_db"] = (
    datos_urbanos["densidad_vehiculos"] * 0.2
    + 50
    + np.random.normal(0, 5, n)
)

print("\nPRIMERAS FILAS")
print(datos_urbanos.head())

# ==========================================
# ESCALAMIENTO
# ==========================================

scaler = StandardScaler()

datos_escalados = scaler.fit_transform(
    datos_urbanos
)

# ==========================================
# PCA
# ==========================================

pca = PCA()

componentes = pca.fit_transform(
    datos_escalados
)

# ==========================================
# VARIANZA EXPLICADA
# ==========================================

print("\nVARIANZA EXPLICADA")

print(
    pca.explained_variance_ratio_
)

# ==========================================
# VARIANZA ACUMULADA
# ==========================================

varianza_acumulada = np.cumsum(
    pca.explained_variance_ratio_
)

print("\nVARIANZA ACUMULADA")

print(varianza_acumulada)

# ==========================================
# COMPONENTES PARA 85%
# ==========================================

comp85 = np.argmax(
    varianza_acumulada >= 0.85
) + 1

print("\nCOMPONENTES PARA 85%")

print(comp85)

# ==========================================
# SCREE PLOT
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(
    range(
        1,
        len(varianza_acumulada)+1
    ),
    varianza_acumulada,
    marker="o"
)

plt.title(
    "Scree Plot Smart City"
)

plt.xlabel("Componentes")
plt.ylabel("Varianza Acumulada")

plt.grid()

plt.show()

# ==========================================
# LOADINGS
# ==========================================

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[
        f"PC{i+1}"
        for i in range(
            len(datos_urbanos.columns)
        )
    ],
    index=datos_urbanos.columns
)

print("\nLOADINGS PC1 Y PC2")

print(
    loadings[["PC1","PC2"]]
)

# ==========================================
# BIPLOT SIMPLE
# ==========================================

plt.figure(figsize=(8,5))

plt.scatter(
    componentes[:,0],
    componentes[:,1]
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title(
    "Biplot Estado Urbano"
)

plt.grid()

plt.show()

# ==========================================
# CONCLUSION
# ==========================================

print("\nCONCLUSION")

print(
    f"Se requieren {comp85} componentes "
    f"para explicar al menos el 85% "
    f"de la variabilidad."
)