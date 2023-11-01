from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Cargar el conjunto de datos Iris
iris = load_iris()
x, y = iris.data, iris.target

print(x)
 

# Dividir los datos en conjuntos de entrenamiento y prueba 

# Crear una instancia de MLPClassifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Entrenar el clasificador MLP en los datos de entrenamiento
mlp_classifier.fit(x, y)

# Realizar predicciones en el conjunto de prueba
y_pred = mlp_classifier.predict(x)


# Calcular la precisión del modelo
plt.plot(y)
plt.plot(y_pred,color='#FF0000') 
plt.show()
