# ==========================================
# PRACTICA 2 - ANALISIS DE CALIDAD INDUSTRIAL
# ==========================================

# LIBRERIAS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# GENERACION DE DATOS
# ==========================================

np.random.seed(123)

n = 500

temperatura = np.random.normal(80, 5, n)

tasa_error = temperatura * 0.2 + np.random.normal(0, 2, n)

turno = np.random.choice(
    ["Matutino", "Vespertino"],
    n
)

# CREACION DEL DATAFRAME

df = pd.DataFrame({
    "temperatura": temperatura,
    "tasa_error": tasa_error,
    "turno": turno
})

# ==========================================
# EXPLORACION INICIAL
# ==========================================

print("\nPRIMERAS FILAS")
print(df.head())

print("\nINFORMACION GENERAL")
print(df.info())

print("\nESTADISTICAS")
print(df.describe())

# ==========================================
# MEDIA, MEDIANA Y DESVIACION ESTANDAR
# ==========================================

print("\nMEDIA TEMPERATURA")
print(df["temperatura"].mean())

print("\nMEDIANA TEMPERATURA")
print(df["temperatura"].median())

print("\nDESVIACION ESTANDAR")
print(df["temperatura"].std())

# ==========================================
# CREAR VALORES FALTANTES
# ==========================================

df.loc[0:10, "temperatura"] = np.nan

print("\nVALORES NULOS ANTES DE LIMPIAR")
print(df.isnull().sum())

# ==========================================
# LIMPIEZA DE DATOS
# ==========================================

df["temperatura"] = df["temperatura"].fillna(
    df["temperatura"].mean()
)

print("\nVALORES NULOS DESPUES DE LIMPIAR")
print(df.isnull().sum())

# ==========================================
# CORRELACION DE PEARSON
# ==========================================

correlacion = df["temperatura"].corr(
    df["tasa_error"]
)

print("\nCORRELACION TEMPERATURA VS ERROR")
print(correlacion)

# ==========================================
# AGRUPACION POR TURNO
# ==========================================

promedio_turno = df.groupby(
    "turno"
)["tasa_error"].mean()

print("\nPROMEDIO DE ERROR POR TURNO")
print(promedio_turno)

# ==========================================
# BOXPLOT
# ==========================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x="turno",
    y="tasa_error",
    data=df
)

plt.title("Boxplot de Errores por Turno")

plt.show()

# ==========================================
# SCATTER PLOT CON REGRESION
# ==========================================

plt.figure(figsize=(8,5))

sns.regplot(
    x="temperatura",
    y="tasa_error",
    data=df
)

plt.title(
    "Relacion Temperatura vs Tasa Error"
)

plt.show()

# ==========================================
# CONCLUSIONES
# ==========================================

print("\nCONCLUSION")

if correlacion > 0.5:
    print(
        "Existe una relacion positiva importante "
        "entre temperatura y errores."
    )
else:
    print(
        "La relacion entre temperatura y errores "
        "no parece ser fuerte."
    )