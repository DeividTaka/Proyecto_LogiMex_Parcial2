import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, classification_report, roc_auc_score

try:
    df = pd.read_csv('dataset_preprocesado/entregas_preprocesadas.csv')
    print("Dataset cargado correctamente.")
except FileNotFoundError:
    print("Error: Asegúrate de tener el archivo 'entregas_preprocesadas.csv' en la carpeta 'dataset_preprocesado/'.")
    exit()

features = ['distancia_km', 'tiempo_plan_min', 'costo_mxn', 'tipo_servicio', 'clima', 'origen', 'destino']
X = df[features]
y_reg = df['retraso_min']
y_clas = df['entregado_a_tiempo']

cat_cols = ['tipo_servicio', 'clima', 'origen', 'destino']
num_cols = ['distancia_km', 'tiempo_plan_min', 'costo_mxn']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ]
)

print("\n==================================================")
print("1. ENTRENAMIENTO DE MODELO DE REGRESIÓN (retraso_min)")
print("==================================================")

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

pipeline_reg = Pipeline([
    ('prep', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

param_grid_reg = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [10, 20, None]
}

grid_reg = GridSearchCV(pipeline_reg, param_grid_reg, cv=5, scoring='r2')
grid_reg.fit(X_train_r, y_train_r)

y_pred_r = grid_reg.best_estimator_.predict(X_test_r)

print(f"Mejores Parámetros Regresión: {grid_reg.best_params_}")
print(f"Error Absoluto Medio (MAE): {mean_absolute_error(y_test_r, y_pred_r):.2f} minutos")
print(f"Coeficiente R2: {r2_score(y_test_r, y_pred_r):.4f}")


print("\n==================================================")
print("2. ENTRENAMIENTO DE MODELO DE CLASIFICACIÓN (entregado_a_tiempo)")
print("==================================================")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_clas, test_size=0.2, random_state=42, stratify=y_clas
)

pipeline_clas = Pipeline([
    ('prep', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline_clas.fit(X_train_c, y_train_c)
y_pred_c = pipeline_clas.predict(X_test_c)
y_proba_c = pipeline_clas.predict_proba(X_test_c)[:, 1]

print("Reporte de Clasificación:")
print(classification_report(y_test_c, y_pred_c))
print(f"Área Bajo la Curva ROC (ROC-AUC): {roc_auc_score(y_test_c, y_proba_c):.4f}")