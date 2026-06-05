# ==========================================
# PRACTICA 7
# MACHINE LEARNING - RECURSOS HUMANOS
# ==========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==========================================
# GENERACION DE DATOS
# ==========================================

np.random.seed(789)

n = 500

df_empleados = pd.DataFrame({
    "experiencia": np.random.normal(5, 2, n),
    "certificaciones": np.random.poisson(3, n),
    "habilidades_sociales": np.random.uniform(1, 10, n),
    "remoto": np.random.choice([0, 1], n)
})

# ==========================================
# SALARIO
# ==========================================

df_empleados["salario"] = (
    25000
    + df_empleados["experiencia"] * 5000
    + df_empleados["certificaciones"] * 2000
    + np.random.normal(0, 3000, n)
)

# ==========================================
# RETENCION
# ==========================================

prob = 1 / (
    1 + np.exp(
        -(
            -2
            + 0.0001 * df_empleados["salario"]
            + 0.2 * df_empleados["habilidades_sociales"]
        )
    )
)

df_empleados["retencion"] = np.where(
    np.random.rand(n) < prob,
    1,
    0
)

# ==========================================
# EXPLORACION
# ==========================================

print("\nPRIMERAS FILAS")
print(df_empleados.head())

print("\nESTADISTICAS")
print(df_empleados.describe())

# ==========================================
# REGRESION LINEAL
# ==========================================

X = df_empleados[
    ["experiencia", "certificaciones"]
]

y = df_empleados["salario"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=123
)

modelo_salario = LinearRegression()

modelo_salario.fit(
    X_train,
    y_train
)

print("\nCOEFICIENTES DEL MODELO")

print(
    modelo_salario.coef_
)

print(
    "Intercepto:",
    modelo_salario.intercept_
)

# ==========================================
# PREDICCIONES
# ==========================================

pred_salario = modelo_salario.predict(
    X_test
)

# ==========================================
# KNN
# ==========================================

X_knn = df_empleados[
    [
        "experiencia",
        "habilidades_sociales",
        "salario"
    ]
]

y_knn = df_empleados["retencion"]

X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
    X_knn,
    y_knn,
    test_size=0.20,
    random_state=123
)

scaler = StandardScaler()

X_train_knn = scaler.fit_transform(
    X_train_knn
)

X_test_knn = scaler.transform(
    X_test_knn
)

knn = KNeighborsClassifier(
    n_neighbors=5
)

knn.fit(
    X_train_knn,
    y_train_knn
)

pred_knn = knn.predict(
    X_test_knn
)

accuracy = accuracy_score(
    y_test_knn,
    pred_knn
)

print("\nACCURACY KNN")

print(accuracy)

# ==========================================
# K-MEANS
# ==========================================

datos_cluster = df_empleados[
    ["experiencia", "salario"]
]

datos_cluster = scaler.fit_transform(
    datos_cluster
)

kmeans = KMeans(
    n_clusters=3,
    random_state=456,
    n_init=10
)

clusters = kmeans.fit_predict(
    datos_cluster
)

df_empleados["cluster"] = clusters

# ==========================================
# VISUALIZACION
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df_empleados["experiencia"],
    df_empleados["salario"],
    c=df_empleados["cluster"]
)

plt.xlabel("Experiencia")
plt.ylabel("Salario")

plt.title(
    "Segmentacion de Empleados"
)

plt.grid()

plt.show()

# ==========================================
# RESUMEN CLUSTERS
# ==========================================

print("\nPROMEDIOS POR CLUSTER")

print(
    df_empleados.groupby("cluster")[
        ["experiencia", "salario"]
    ].mean()
)

# ==========================================
# CONCLUSION
# ==========================================

print("\nCONCLUSION")

print(
    "La regresion estima salarios, "
    "KNN clasifica retencion y "
    "KMeans segmenta empleados."
)