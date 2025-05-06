import pandas as pd
import numpy as np
import time
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve, GridSearchCV
import matplotlib.pyplot as plt

# Preprocesamiento
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import sparse

# Modelos de redes neuronales con sklearn
from sklearn.neural_network import MLPClassifier

# Métricas
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Algoritmos clásicos
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# -------------------
# Inciso 1: Cargar datos
# -------------------
print("Inciso 1. Cargando conjuntos de entrenamiento y prueba, como antes")
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

# -------------------
# Inciso 2: Crear variable categórica
# -------------------
print("Inciso 2. Generando variable categórica 'PriceCategory'")
g1, g2 = train['SalePrice'].quantile([0.33, 0.66])
train['PriceCategory'] = pd.cut(
    train['SalePrice'], bins=[-np.inf, g1, g2, np.inf],
    labels=['barata', 'media', 'cara']
)
X = train.drop(columns=['Id', 'SalePrice', 'PriceCategory'])
y = train['PriceCategory']

# -------------------
# Inciso 3: Preprocesamiento
# -------------------
#separacion datos
print("Inciso 3. Preprocesando características (imputación, escalado, one-hot)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 80% train, 20% test
)
num_feats = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_feats = X.select_dtypes(include=['object']).columns.tolist()
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_feats),
    ('cat', cat_pipe, cat_feats)
])
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

if sparse.issparse(X_train_proc):
    X_train_proc = X_train_proc.toarray()
    X_test_proc = X_test_proc.toarray()
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# -------------------
# Inciso 4: Modelos RNA con MLPClassifier
# -------------------
print("Inciso 4. Definiendo y evaluando modelos RNA")
models = [
    (MLPClassifier(hidden_layer_sizes=(32,), activation='relu', solver='adam', max_iter=200), 'RNA_32relu'),
    (MLPClassifier(hidden_layer_sizes=(64,32), activation='tanh', solver='adam', max_iter=200), 'RNA_64-32tanh')
]
results = []
for clf, name in models:
    print(f"Evaluando {name}...")
    t0 = time.time()
    clf.fit(X_train_proc, y_train_enc)
    train_time = time.time() - t0
    t0 = time.time()
    preds_train = clf.predict(X_train_proc)
    preds_test = clf.predict(X_test_proc)
    pred_time = time.time() - t0
    acc_train = accuracy_score(y_train_enc, preds_train)
    acc_test = accuracy_score(y_test_enc, preds_test)
    cm = confusion_matrix(y_test_enc, preds_test)
    print(f"Matriz de confusión ({name}):\n{cm}")
    print(classification_report(y_test_enc, preds_test, target_names=le.classes_, digits=4))
    results.append({'modelo': name, 'accuracy': acc_train,
                    'tiempo_train': train_time, 'tiempo_pred': pred_time})

# -------------------
# Incisos 5-6: Comparación con algoritmos clásicos
# -------------------
print("Incisos 5-6. Comparando con algoritmos clásicos de sklearn...")
sk_models = [
    (DecisionTreeClassifier(), 'DecisionTree'),
    (RandomForestClassifier(), 'RandomForest'),
    (GaussianNB(), 'NaiveBayes'),
    (KNeighborsClassifier(), 'KNN'),
    (LogisticRegression(max_iter=1000), 'LogisticRegression'),
    (SVC(), 'SVM')
]
for clf, name in sk_models:
    print(f"Evaluando {name}...")
    X_input_train = X_train_proc
    X_input_test = X_test_proc
    if name == 'NaiveBayes' and sparse.issparse(X_input_train):
        X_input_train = X_input_train.toarray()
        X_input_test = X_input_test.toarray()
    t0 = time.time()
    clf.fit(X_input_train, y_train_enc)
    train_time = time.time() - t0
    t0 = time.time()
    preds_train = clf.predict(X_input_train)
    preds_test = clf.predict(X_input_test)
    pred_time = time.time() - t0
    acc_train = accuracy_score(y_train_enc, preds_train)
    acc_test = accuracy_score(y_test_enc, preds_test)
    cm = confusion_matrix(y_test_enc, preds_test)
    print(f"Matriz de confusión ({name}):\n{cm}")
    print(classification_report(y_test_enc, preds_test, target_names=le.classes_, digits=4))
    results.append({'modelo': name, 'accuracy': acc_train,
                    'tiempo_train': train_time, 'tiempo_pred': pred_time})

# Guardar resumen comparativo
df_results = pd.DataFrame(results)
print("\nResumen comparativo de modelos:")
print(df_results)

# -------------------
# Inciso 9: Seleccionar SalePrice como variable respuesta
# -------------------
print("Inciso 9. Seleccionando SalePrice como variable respuesta")
X_reg = train.drop(columns=['Id', 'SalePrice', 'PriceCategory'])
y_reg = train['SalePrice']

# volver a dividir en train/test
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Preprocesar de nuevo
X_train_r_proc = preprocessor.fit_transform(X_train_r)
X_test_r_proc  = preprocessor.transform(X_test_r)
if sparse.issparse(X_train_r_proc):
    X_train_r_proc = X_train_r_proc.toarray()
    X_test_r_proc  = X_test_r_proc.toarray()

# -------------------
# Inciso 10: Dos modelos de regresión con redes neuronales
# -------------------
print("Inciso 10. Definiendo y evaluando modelos de regresión con MLP")
reg_models = [
    (MLPRegressor(hidden_layer_sizes=(32,),  activation='relu', solver='adam',
                  max_iter=200, random_state=42), 'Reg_32relu'),
    (MLPRegressor(hidden_layer_sizes=(64,32), activation='tanh', solver='adam',
                  max_iter=200, random_state=42), 'Reg_64-32tanh')
]

reg_results = []
for model, name in reg_models:
    print(f"Evaluando {name}...")
    t0 = time.time()
    model.fit(X_train_r_proc, y_train_r)
    train_time = time.time() - t0
    t0 = time.time()
    pred_train = model.predict(X_train_r_proc)
    pred_test  = model.predict(X_test_r_proc)
    pred_time  = time.time() - t0

    mse_tr = mean_squared_error(y_train_r, pred_train)
    mse_te = mean_squared_error(y_test_r,  pred_test)
    mae_tr = mean_absolute_error(y_train_r, pred_train)
    mae_te = mean_absolute_error(y_test_r,  pred_test)
    r2_tr  = r2_score(y_train_r, pred_train)
    r2_te  = r2_score(y_test_r,  pred_test)

    print(f"{name} → Train MSE: {mse_tr:.2f}, Test MSE: {mse_te:.2f}")
    print(f"{name} → Train R2: {r2_tr:.4f}, Test R2: {r2_te:.4f}\n")

    reg_results.append({
        'modelo': name,
        'mse_train': mse_tr, 'mse_test': mse_te,
        'mae_train': mae_tr, 'mae_test': mae_te,
        'r2_train': r2_tr,  'r2_test': r2_te,
        'tiempo_train': train_time, 'tiempo_pred': pred_time
    })

df_reg_results = pd.DataFrame(reg_results)
print("Resumen comparativo de modelos de regresión:")
print(df_reg_results)

# -------------------
# Inciso 11: Determinar el mejor según MSE de test
# -------------------
print("Inciso 11. Determinar el mejor según MSE de test")
best = min(reg_results, key=lambda x: x['mse_test'])
print(f"\nEl mejor modelo es {best['modelo']} con Test MSE = {best['mse_test']:.2f}")

# crea un dict { 'Reg_32relu': modelo_obj, ... }
name_to_model = { name: model for model, name in reg_models }
best_model_obj = name_to_model[best['modelo']]

# -------------------
# Inciso 12: Curva de aprendizaje para el mejor modelo
# -------------------
print("\nInciso 12. Generando curva de aprendizaje...")
train_sizes, train_scores, val_scores = learning_curve(
    best_model_obj, X_train_r_proc, y_train_r,
    cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)

# convertir a MSE positivo y promediar
train_mse = -train_scores.mean(axis=1)
val_mse   = -val_scores.mean(axis=1)

plt.figure()
plt.plot(train_sizes, train_mse, label='Train MSE')
plt.plot(train_sizes, val_mse,   label='Validación MSE')
plt.xlabel('Número de muestras de entrenamiento')
plt.ylabel('Mean Squared Error')
plt.title('Curva de Aprendizaje')
plt.legend()
plt.show()

# -------------------
# Inciso 13: Tuning de hiperparámetros con GridSearchCV
# -------------------
print("\nInciso 13. Tunéando hiperparámetros del mejor modelo...")
pipeline_reg = Pipeline([
    ('preproc', preprocessor),
    ('reg',     MLPRegressor(max_iter=200, random_state=42))
])

param_grid = {
    'reg__hidden_layer_sizes': [(32,), (64,32), (100,50)],
    'reg__activation': ['relu', 'tanh'],
    'reg__alpha': [1e-4, 1e-3, 1e-2],
    'reg__learning_rate_init': [1e-3, 1e-2]
}

grid = GridSearchCV(
    pipeline_reg, param_grid,
    cv=5, scoring='neg_mean_squared_error',
    n_jobs=-1, verbose=1
)
grid.fit(X_train_r, y_train_r)

print("Mejores parámetros encontrados:")
print(grid.best_params_)
best_pipe = grid.best_estimator_
pred_test = best_pipe.predict(X_test_r)
print(f"Después de tuning → Test MSE: {mean_squared_error(y_test_r, pred_test):.2f}, R2: {r2_score(y_test_r, pred_test):.4f}")

#Esto llevo probablemente mucho más de lo que debería
