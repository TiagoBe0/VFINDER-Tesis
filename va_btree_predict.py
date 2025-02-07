import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class VacancyPredictor:
    def __init__(self, training_file, features_to_use=None, test_size=0.2, random_state=42):
        self.training_file = training_file
        self.test_size = test_size
        self.random_state = random_state
        self.features_to_use = features_to_use if features_to_use is not None else [0, 1, 2, 3]
        self.modelo = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def cargar_datos(self):
        with open(self.training_file, 'r') as f:
            datos = json.load(f)
        sm_mesh_training = datos['sm_mesh_training']
        vecinos = datos['vecinos']
        max_distancias = datos['max_distancias']
        min_distancias = datos['min_distancias']
        vacancias = datos['vacancias']
        # Se arma una lista de tuplas con las 4 columnas
        data_full = list(zip(sm_mesh_training, vecinos, max_distancias, min_distancias))
        # Aquí se seleccionan únicamente las columnas indicadas en self.features_to_use
        X = np.array([[ejemplo[idx] for idx in self.features_to_use] for ejemplo in data_full])
        y = np.array(vacancias)
        return X, y
    def entrenar_modelo(self):
        X, y = self.cargar_datos()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.modelo = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        self.modelo.fit(self.X_train, self.y_train)

    def evaluar_modelo(self):
        if self.modelo is None:
            raise ValueError("El modelo no ha sido entrenado aún. Ejecute entrenar_modelo().")
        y_pred = self.modelo.predict(self.X_test)
        error = mean_squared_error(self.y_test, y_pred)
        print("Error Cuadrático Medio en el conjunto de prueba:", error)
        print("Predicciones en el conjunto de prueba:")
        print(y_pred)
        print("Valores reales:")
        print(self.y_test)
        return error
    def predecir_por_cluster(self, clusters_file, single_vacancy_file):
        with open(clusters_file, 'r') as f:
            datos_clusters = json.load(f)
        # Asegurarse de que cada fila tenga al menos la cantidad de columnas necesarias:
        clusters = [fila for fila in datos_clusters["clusters"] if len(fila) >= max(self.features_to_use) + 1]

        with open(single_vacancy_file, 'r') as f:
            datos_sv = json.load(f)
        sms_sv = datos_sv["sm_mesh_training"]
        nb_sv = datos_sv["vecinos"]

        tol_area = 1e-3
        tol_nb = 0
        print("\nRealizando predicciones para cada cluster:")
        total_vacancias = 0

        for i, cluster in enumerate(clusters, start=1):
            area_cluster = cluster[0]
            nb_cluster = cluster[1]
            is_single_vacancy = False

            for sv_area, sv_nb in zip(sms_sv, nb_sv):
                if abs(area_cluster - sv_area) <= tol_area and abs(nb_cluster - sv_nb) <= tol_nb:
                    is_single_vacancy = True
                    break

            if is_single_vacancy:
                print(f"Cluster {i}: SINGLE VACANCY detectado (Área: {area_cluster}, Vecinos: {nb_cluster}).")
                total_vacancias += 1
            else:
                # Extraemos únicamente los features indicados (por ejemplo, si features_to_use=[0,1])
                features = [cluster[idx] for idx in self.features_to_use]
                features_array = np.array(features).reshape(1, -1)
                prediccion = self.modelo.predict(features_array)
                total_vacancias += np.ceil(prediccion[0])
                print(f"Cluster {i}:")
                print(f"  Características usadas: {features}")
                print(f"  Predicción de vacancias: {np.ceil(prediccion[0])}\n")

        print(f"Vacancias totales acumuladas: {round(total_vacancias)} (valor total: {total_vacancias})")
        return total_vacancias
