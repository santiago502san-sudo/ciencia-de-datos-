# ==========================================
# PRACTICA 5
# REDUCCION DE DIMENSIONES EN TRAFICO DE RED
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ==========================================
# GENERACION DE DATOS
# ==========================================

np.random.seed(123)

n = 300

datos_red = pd.DataFrame({
    "duracion_ms": np.random.normal(50, 10, n),
    "paquetes_enviados": np.random.normal(100, 20, n),
    "errores_checksum": np.random.poisson(2, n),
    "latencia_avg": np.random.normal(15, 5, n),
    "jitter": np.random.normal(2, 0.5, n),
    "uso_memoria_sw": np.random.normal(40, 10, n),
    "peticiones_http": np.random.normal(200, 50, n)
})

# Crear dependencias (redundancia)

datos_red["bytes_enviados"] = (
    datos_red["paquetes_enviados"] * 1500
    + np.random.normal(0, 500, n)
)

datos_red["reintentos_tcp"] = (
    datos_red["errores_checksum"] * 1.5
    + np.random.normal(0, 0.5, n)
)

datos_red["carga_cpu_router"] = (
    datos_red["paquetes_enviados"] * 0.4
    + datos_red["latencia_avg"] * 0.2
)

# ==========================================
# EXPLORACION
# ==========================================

print("\nPRIMERAS FILAS")
print(datos_red.head())

print("\nINFORMACION GENERAL")
print(datos_red.info())

# ==========================================
# ESCALAMIENTO
# ==========================================

scaler = StandardScaler()

datos_escalados = scaler.fit_transform(datos_red)

# ==========================================
# PCA
# ==========================================

pca_red = PCA()

componentes = pca_red.fit_transform(
    datos_escalados
)

# ==========================================
# VARIANZA EXPLICADA
# ==========================================

print("\nVARIANZA EXPLICADA")

print(
    pca_red.explained_variance_ratio_
)

# ==========================================
# VARIANZA ACUMULADA
# ==========================================

varianza_acumulada = np.cumsum(
    pca_red.explained_variance_ratio_
)

print("\nVARIANZA ACUMULADA")

print(varianza_acumulada)

# ==========================================
# COMPONENTES PARA 85%
# ==========================================

componentes_85 = np.argmax(
    varianza_acumulada >= 0.85
) + 1

print(
    "\nCOMPONENTES NECESARIOS PARA 85%"
)

print(componentes_85)

# ==========================================
# SCREE PLOT
# ==========================================

plt.figure(figsize=(8,5))

plt.plot(
    range(
        1,
        len(varianza_acumulada) + 1
    ),
    varianza_acumulada,
    marker="o"
)

plt.title(
    "Scree Plot Trafico de Red"
)

plt.xlabel("Componentes")

plt.ylabel(
    "Varianza Acumulada"
)

plt.grid()

plt.show()

# ==========================================
# LOADINGS
# ==========================================

loadings = pd.DataFrame(
    pca_red.components_.T,
    columns=[
        f"PC{i+1}"
        for i in range(
            len(datos_red.columns)
        )
    ],
    index=datos_red.columns
)

print("\nLOADINGS PC1 Y PC2")

print(
    loadings[["PC1", "PC2"]]
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
    "Biplot Trafico de Red"
)

plt.grid()

plt.show()

# ==========================================
# CONCLUSIONES
# ==========================================

print("\nCONCLUSIONES")

print(
    f"Se necesitan {componentes_85} "
    f"componentes para explicar "
    f"al menos el 85% de la varianza."
)