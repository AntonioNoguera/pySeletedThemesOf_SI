##RNA PLAYGROUND

import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Cargamos un conjunto de datos de ejemplo (Iris)
data = load_iris()
X = data.data
y = data.target

# Dividimos los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creamos y entrenamos un modelo de RNA
modelo_rna = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
modelo_rna.fit(X_train, y_train)

# Realizamos predicciones en el conjunto de prueba
y_pred = modelo_rna.predict(X_test)

# Calculamos la precisión del modelo en el conjunto de prueba
precision = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {precision}')

# Guardamos el modelo en un archivo con pickle
with open('modelo_rna.pkl', 'wb') as archivo:
    pickle.dump(modelo_rna, archivo)

    # Guardamos el modelo en un archivo con pickle
with open('modelo_rna.pkl', 'wb') as archivo:
    pickle.dump(modelo_rna, archivo)


# Cargamos el modelo desde el archivo
with open('modelo_rna.pkl', 'rb') as archivo:
    modelo_cargado = pickle.load(archivo)

# Realizamos predicciones con el modelo cargado
y_pred_cargado = modelo_cargado.predict(X_test)

# Calculamos la precisión del modelo cargado
precision_cargado = accuracy_score(y_test, y_pred_cargado)
print(f'Precisión del modelo cargado: {precision_cargado}')