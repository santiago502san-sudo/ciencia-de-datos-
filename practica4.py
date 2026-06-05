# ==========================================
# PRACTICA 4 - PCA INDUSTRIAL
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# GENERAR DATOS
# ==========================================

np.random.seed(123)

n = 500

temperatura = np.random.normal(
    80,
    5,
    n
)

presion = np.random.normal(
    100,
    10,
    n
)

vibracion = np.random.normal(
    20,
    3,
    n
)

humedad = np.random.normal(
    60,
    8,
    n
)

velocidad = np.random.normal(
    1500,
    200,
    n
)

# ==========================================
# DATAFRAME
# ==========================================

df = pd.DataFrame({
    "temperatura": temperatura,
    "presion": presion,
    "vibracion": vibracion,
    "humedad": humedad,
    "velocidad": velocidad
})

print("\nPRIMERAS FILAS")
print(df.head())

# ==========================================
# ESTANDARIZACION
# ==========================================

scaler = StandardScaler()

datos_escalados = scaler.fit_transform(df)

print("\nDATOS ESCALADOS")
print(datos_escalados[:5])

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
# COMPONENTES NECESARIOS
# ==========================================

componentes_85 = np.argmax(
    varianza_acumulada >= 0.85
) + 1

print(
    "\nCOMPONENTES PARA 85%"
)

print(componentes_85)

# ==========================================
# SCREE PLOT
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(
    range(
        1,
        len(
            pca.explained_variance_ratio_
        ) + 1
    ),
    varianza_acumulada,
    marker="o"
)

plt.xlabel("Componentes")

plt.ylabel(
    "Varianza Acumulada"
)

plt.title("Scree Plot")

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
            len(df.columns)
        )
    ],
    index=df.columns
)

print("\nLOADINGS")

print(loadings)

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

plt.title("Biplot PCA")

plt.grid()

plt.show()

# ==========================================
# CONCLUSION
# ==========================================

print("\nCONCLUSION")

print(
    f"Se necesitan "
    f"{componentes_85} componentes "
    f"para explicar al menos el 85% "
    f"de la variabilidad."
)