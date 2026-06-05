# ==========================================
# PRACTICA 8
# CIBERSEGURIDAD Y DETECCION DE INTRUSIONES
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ==========================================
# GENERACION DE DATOS
# ==========================================

np.random.seed(444)

n = 400

df_red = pd.DataFrame({
    "latencia": np.random.normal(20, 5, n),
    "intentos_fallidos": np.random.poisson(1, n),
    "tamano_paquete": np.random.normal(500, 100, n)
})

# ==========================================
# CREACION DE VARIABLE ATAQUE
# ==========================================

df_red["es_ataque"] = np.where(
    (df_red["intentos_fallidos"] > 2) |
    (df_red["latencia"] > 35),
    1,
    0
)

# ==========================================
# EXPLORACION
# ==========================================

print("\nPRIMERAS FILAS")
print(df_red.head())

print("\nINFORMACION GENERAL")
print(df_red.info())

print("\nDISTRIBUCION DE ATAQUES")
print(df_red["es_ataque"].value_counts())

# ==========================================
# VARIABLES
# ==========================================

X = df_red[
    [
        "latencia",
        "intentos_fallidos",
        "tamano_paquete"
    ]
]

y = df_red["es_ataque"]

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=123
)

# ==========================================
# REGRESION LOGISTICA
# ==========================================

modelo_log = LogisticRegression(
    max_iter=1000
)

modelo_log.fit(
    X_train,
    y_train
)

pred_log = modelo_log.predict(
    X_test
)

acc_log = accuracy_score(
    y_test,
    pred_log
)

print("\nACCURACY REGRESION LOGISTICA")
print(acc_log)

print("\nCOEFICIENTES")

print(modelo_log.coef_)

# ==========================================
# KNN
# ==========================================

scaler = StandardScaler()

X_train_s = scaler.fit_transform(
    X_train
)

X_test_s = scaler.transform(
    X_test
)

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(
    X_train_s,
    y_train
)

pred_knn = knn.predict(
    X_test_s
)

acc_knn = accuracy_score(
    y_test,
    pred_knn
)

print("\nACCURACY KNN")
print(acc_knn)

# ==========================================
# KMEANS
# ==========================================

datos_cluster = scaler.fit_transform(
    X
)

kmeans = KMeans(
    n_clusters=2,
    random_state=111,
    n_init=10
)

clusters = kmeans.fit_predict(
    datos_cluster
)

df_red["cluster"] = clusters

# ==========================================
# COMPARACION
# ==========================================

print("\nTABLA ATAQUE VS CLUSTER")

print(
    pd.crosstab(
        df_red["es_ataque"],
        df_red["cluster"]
    )
)

# ==========================================
# GRAFICA
# ==========================================

plt.figure(figsize=(8,5))

plt.scatter(
    df_red["latencia"],
    df_red["intentos_fallidos"],
    c=df_red["cluster"]
)

plt.xlabel("Latencia")
plt.ylabel("Intentos Fallidos")

plt.title(
    "Clusters de Trafico de Red"
)

plt.grid()

plt.show()

# ==========================================
# CONCLUSION
# ==========================================

print("\nCONCLUSION")

print(
    "La regresion logistica clasifica ataques, "
    "KNN valida la clasificacion y "
    "KMeans identifica patrones de trafico."
)