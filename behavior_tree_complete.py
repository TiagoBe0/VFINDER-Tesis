import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class VacancyPredictor:
    def __init__(self, training_file, features_to_use=None, test_size=0.2, random_state=42):
        """
        training_file: ruta al archivo JSON con los datos de entrenamiento.
        features_to_use: lista de índices (0-based) de las columnas a usar en X.
            Por defecto se usan todas las 4 columnas: [0, 1, 2, 3].
        """
        self.training_file = training_file
        self.test_size = test_size
        self.random_state = random_state
        # Si no se especifica, usar todas las columnas.
        self.features_to_use = features_to_use if features_to_use is not None else [0, 1, 2, 3]
        self.modelo = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def cargar_datos(self):
        """Carga y procesa los datos de entrenamiento desde el archivo JSON."""
        with open(self.training_file, 'r') as f:
            datos = json.load(f)
        # Extraer las listas
        sm_mesh_training = datos['sm_mesh_training']    # Área de superficie
        vecinos = datos['vecinos']                        # Número de vecinos
        max_distancias = datos['max_distancias']          # Mayor distancia
        min_distancias = datos['min_distancias']          # Menor distancia
        vacancias = datos['vacancias']                    # Target: número de vacancias

        # Crear una lista de ejemplos con las 4 columnas originales
        data_full = list(zip(sm_mesh_training, vecinos, max_distancias, min_distancias))
        # Seleccionar solo las columnas indicadas en features_to_use
        X = np.array([[ejemplo[idx] for idx in self.features_to_use] for ejemplo in data_full])
        y = np.array(vacancias)
        return X, y

    def entrenar_modelo(self):
        """Carga los datos, los divide en entrenamiento/prueba y entrena un RandomForest."""
        X, y = self.cargar_datos()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        self.modelo = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        self.modelo.fit(self.X_train, self.y_train)

    def evaluar_modelo(self):
        """Realiza predicciones en el conjunto de prueba y muestra el error y los resultados."""
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
        """
        Itera sobre cada cluster definido en el archivo JSON de clusters y realiza una predicción usando las columnas indicadas.
        Se asume que el archivo JSON de clusters tiene la estructura:
            { "num_clusters": ..., "clusters": [ [...], [...], ... ] }
        y que cada cluster es una lista donde los primeros 4 valores son:
            [Área de superficie, número de vecinos, mayor distancia, menor distancia]
        Las últimas 3 columnas se ignoran.
        
        Además, se filtran los clusters que se consideran "single vacancy" según los datos en single_vacancy_file.
        Los clusters que tengan un área cercana a algún valor en "sms_sv" y un número de vecinos cercano a algún valor en "nb_sv"
        se descartarán de la predicción y se sumará 1 vacancia directamente.
        """
        # Cargar los datos de los clusters
        with open(clusters_file, 'r') as f:
            datos_clusters = json.load(f)
        clusters = datos_clusters["clusters"]

        # Cargar los datos para single vacancy
        with open(single_vacancy_file, 'r') as f:
            datos_sv = json.load(f)
        sms_sv = datos_sv["sms_sv"]
        nb_sv = datos_sv["nb_sv"]

        # Definir tolerancias para considerar que dos valores son "cercanos"
        tol_area = 1e-3  # Puedes ajustar este valor
        # Para el número de vecinos, al tratarse de enteros, se puede comparar exacto
        tol_nb = 0

        print("\nRealizando predicciones para cada cluster:")
        total_vacancias = 0
        for i, cluster in enumerate(clusters, start=1):
            # Primero, revisar si el cluster es una single vacancy.
            area_cluster = cluster[0]      # Área de superficie
            nb_cluster = cluster[1]        # Número de vecinos
            is_single_vacancy = False
            for sv_area, sv_nb in zip(sms_sv, nb_sv):
                # Comparar si el área es lo suficientemente cercana y el número de vecinos coincide
                if abs(area_cluster - sv_area) <= tol_area and abs(nb_cluster - sv_nb) <= tol_nb:
                    is_single_vacancy = True
                    break

            if is_single_vacancy:
                # Si es una single vacancy, se suma 1 y se omite la predicción con el modelo
                print(f"Cluster {i}: SINGLE VACANCY detectado (Área: {area_cluster}, Vecinos: {nb_cluster}).")
                total_vacancias += 1
            else:
                # Seleccionar las columnas indicadas en features_to_use de los primeros 4 valores
                features = [cluster[idx] for idx in self.features_to_use if idx < 4]
                features_array = np.array(features).reshape(1, -1)
                prediccion = self.modelo.predict(features_array)
                total_vacancias += np.ceil(prediccion[0])
                print(f"Cluster {i}:")
                print(f"  Características usadas: {features}")
                print(f"  Predicción de vacancias: {np.ceil(prediccion[0])} \n")
        print(f"Vacancias totales acumuladas: {round(total_vacancias)} (valor total: {total_vacancias})")
        return total_vacancias

