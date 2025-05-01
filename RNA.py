import pandas as pd
import numpy as np
import time

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

