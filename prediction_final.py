import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Cargar el archivo JSON
with open("outputs.vfinder/training_cluster.json", "r") as f:
    data = json.load(f)

# 2. Extraer los vectores del JSON
sm_mesh_training = data["sm_mesh_training"]  # Área de superficie
vacancias = data["vacancias"]                # Número de vacancias (target)
vecinos = data["vecinos"]                    # Número de vecinos

# 3. Preparar las variables para el modelo
# Creamos la matriz de características (X) usando 'vecinos' y 'sm_mesh_training'
X = np.array(list(zip(vecinos, sm_mesh_training)))
y = np.array(vacancias)

# 4. Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)


# 6. Realizar predicciones sobre el conjunto de prueba
y_pred = modelo.predict(X_test)

# 7. Evaluar el modelo
error = mean_squared_error(y_test, y_pred)
print("Error Cuadrático Medio:", error)
print("Coeficientes:", modelo.coef_)
print("Intercepto:", modelo.intercept_)

# 8. Comparar algunas predicciones con valores conocidos del conjunto de prueba
print("\nComparación de predicciones en el conjunto de prueba:")
for i in range(5):
    entrada = X_test[i]
    prediccion = modelo.predict(entrada.reshape(1, -1))
    valor_real = y_test[i]
    print(f"Entrada (vecinos, área): {entrada} -> Predicción: {prediccion[0]:.2f} | Valor real: {valor_real}")

# 9. Probar predicción con un valor específico conocido
entrada_nueva = np.array([86,913])  # Ejemplo de datos conocidos
prediccion_nueva = modelo.predict(entrada_nueva.reshape(1, -1))
print(f"\nPara la entrada {entrada_nueva} la predicción es: {prediccion_nueva[0]:.2f}")

