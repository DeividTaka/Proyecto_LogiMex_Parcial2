"""
PROYECTO LOGIMEX DISTRIBUCIÓN - PARCIAL 2
Código 02: Modelos No Supervisados (PCA y K-Means Clustering)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Cargar el dataset preprocesado
try:
    df = pd.read_csv('dataset_preprocesado/entregas_preprocesadas.csv')
    print("Dataset cargado correctamente.")
except FileNotFoundError:
    print("Error: Asegúrate de tener el archivo 'entregas_preprocesadas.csv' en la carpeta 'dataset_preprocesado/'.")
    exit()

# Seleccionar variables continuas para el análisis no supervisado
num_features = ['distancia_km', 'tiempo_plan_min', 'tiempo_real_min', 'retraso_min', 'costo_mxn']
X_numeric = df[num_features]

# Estandarización de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)

print("\n==================================================")
print("1. REDUCCIÓN DE DIMENSIONALIDAD CON PCA")
print("==================================================")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

var_exp = pca.explained_variance_ratio_
print(f"Varianza explicada por Componente 1 (PC1): {var_exp[0]*100:.2f}%")
print(f"Varianza explicada por Componente 2 (PC2): {var_exp[1]*100:.2f}%")
print(f"Varianza acumulada total: {sum(var_exp)*100:.2f}%")

print("\n==================================================")
print("2. SELECCIÓN DE K ÓPTIMO Y AGUPAMIENTO K-MEANS")
print("==================================================")

best_k = 3
best_score = -1

for k in range(2, 6):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    print(f"Silhouette Score para K={k}: {score:.4f}")
    if score > best_score:
        best_score = score
        best_k = k

# Entrenar K-Means con el K óptimo
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster_operacional'] = kmeans.fit_predict(X_pca)

print(f"\nModelo K-Means entrenado exitosamente con K={best_k} (Mejor Silhouette Score: {best_score:.4f})")

# Perfilamiento de clusters
print("\nPerfil Promedio por Cluster Operacional:")
print(df.groupby('cluster_operacional')[num_features + ['entregado_a_tiempo']].mean())

# 3. Generación y guardado de la gráfica
plt.figure(figsize=(9, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster_operacional'], cmap='viridis', alpha=0.8, edgecolors='k')
plt.xlabel(f'Componente Principal 1 ({var_exp[0]*100:.1f}% Varianza)')
plt.ylabel(f'Componente Principal 2 ({var_exp[1]*100:.1f}% Varianza)')
plt.title('Segmentación LogiMex: Clusters Operacionales en Espacio PCA')
plt.colorbar(scatter, label='Cluster Operacional')
plt.grid(True, linestyle='--', alpha=0.5)

# Guardar la gráfica en disco
# LÍNEA CORREGIDA:
plt.savefig('clusters_pca_plot.png', dpi=300, bbox_inches='tight')
print("\nGráfica de clusters guardada como 'modelos_codigo/clusters_pca_plot.png'")