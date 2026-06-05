# ==========================================
# PRACTICA PCA - METAL TECH
# REDUCCION DE DIMENSIONALIDAD
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# GENERACION DEL DATASET
# ==========================================

np.random.seed(42)

n = 200

temp1 = np.random.normal(100, 5, n)

temp2 = temp1 + np.random.normal(0, 1, n)

presion1 = np.random.normal(50, 10, n)

vibracion = (temp1 * 0.5) + np.random.normal(0, 2, n)

datos_sensores = pd.DataFrame({
    "temp_nucleo": temp1,
    "temp_escape": temp2,
    "presion_interna": presion1,
    "vibracion_motor": vibracion,
    "flujo_gas": np.random.normal(20, 2, n),
    "oxigeno": np.random.normal(15, 1, n),
    "co2": temp1 * 0.2 + np.random.normal(0, 0.5, n),
    "humedad": np.random.uniform(30, 40, n),
    "voltaje": np.random.normal(220, 2, n),
    "corriente": np.random.normal(15, 0.5, n),
    "ruido_db": vibracion * 1.2 + np.random.normal(0, 1, n),
    "eficiencia": 100 - (temp1 * 0.1)
})

# ==========================================
# EXPLORACION INICIAL
# ==========================================

print("\nPRIMERAS FILAS")
print(datos_sensores.head())

print("\nINFORMACION GENERAL")
print(datos_sensores.info())

print("\nESTADISTICAS")
print(datos_sensores.describe())

# ==========================================
# ESTANDARIZACION
# ==========================================

scaler = StandardScaler()

datos_escalados = scaler.fit_transform(
    datos_sensores
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
# COMPONENTES NECESARIOS PARA 85%
# ==========================================

componentes_85 = (
    np.argmax(
        varianza_acumulada >= 0.85
    ) + 1
)

print("\nCOMPONENTES PARA 85%")

print(componentes_85)

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

plt.axhline(
    y=0.85,
    linestyle="--"
)

plt.xlabel("Componentes")

plt.ylabel(
    "Varianza Acumulada"
)

plt.title(
    "Scree Plot - Metal Tech"
)

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
            len(datos_sensores.columns)
        )
    ],
    index=datos_sensores.columns
)

print("\nLOADINGS PC1 Y PC2")

print(
    round(
        loadings[["PC1","PC2"]],
        3
    )
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
    "Biplot Sensores Industriales"
)

plt.grid()

plt.show()

# ==========================================
# VARIABLES MAS IMPORTANTES
# ==========================================

print("\nVARIABLES MAS IMPORTANTES EN PC1")

pc1 = (
    loadings["PC1"]
    .abs()
    .sort_values(
        ascending=False
    )
)

print(pc1)

print("\nVARIABLES MAS IMPORTANTES EN PC2")

pc2 = (
    loadings["PC2"]
    .abs()
    .sort_values(
        ascending=False
    )
)

print(pc2)

# ==========================================
# CONCLUSION AUTOMATICA
# ==========================================

print("\nCONCLUSION")

print(
    f"Con {componentes_85} componentes "
    f"se explica al menos el 85% "
    f"de la variabilidad total."
)