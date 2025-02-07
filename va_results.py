import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

with open("outputs.vfinder/training_cluster.json", "r") as f:
    data = json.load(f)

sm_mesh_training = data["sm_mesh_training"]
vacancias = data["vacancias"]
vecinos = data["vecinos"]

X = np.array(list(zip(vecinos, sm_mesh_training)))
y = np.array(vacancias)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

error = mean_squared_error(y_test, y_pred)
print("Error Cuadrático Medio:", error)
print("Coeficientes:", modelo.coef_)
print("Intercepto:", modelo.intercept_)

print("\nComparación de predicciones en el conjunto de prueba:")
for i in range(5):
    entrada = X_test[i]
    prediccion = modelo.predict(entrada.reshape(1, -1))
    valor_real = y_test[i]
    print(f"Entrada (vecinos, área): {entrada} -> Predicción: {prediccion[0]:.2f} | Valor real: {valor_real}")

entrada_nueva = np.array([86, 913])
prediccion_nueva = modelo.predict(entrada_nueva.reshape(1, -1))
print(f"\nPara la entrada {entrada_nueva} la predicción es: {prediccion_nueva[0]:.2f}")
