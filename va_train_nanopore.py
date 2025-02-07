import os
import json
import math
import sys
import numpy as np
import pandas as pd
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier,AffineTransformationModifier, DeleteSelectedModifier, ConstructSurfaceModifier, VoronoiAnalysisModifier, InvertSelectionModifier, ClusterAnalysisModifier
from va_input_params import LAYERS as LY
class ClusterFilter:
    def __init__(self, input_file, output_file=None):
        self.input_file = input_file
        self.output_file = output_file if output_file is not None else input_file
        self.data = None
        
    def load_data(self):
        with open(self.input_file, 'r') as f:
            self.data = json.load(f)

    def filter_clusters(self):
        clusters = self.data.get("clusters", [])
        filtered = [cluster for cluster in clusters if not all(value == 0 for value in cluster)]
        self.data["clusters"] = filtered
        self.data["num_clusters"] = len(filtered)

    def save_data(self):
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(self.output_file, 'w') as f:
            json.dump(self.data, f, indent=4)

    def run(self):
        self.load_data()
        self.filter_clusters()
        self.save_data()
class ClusterProcessor:
    def __init__(self, layers):
        # Se toma el primer elemento de LAYERS
        self.primer_elemento = layers[0]
        self.relax = self.primer_elemento['relax']
        self.defect = self.primer_elemento['defect']
        self.radius = self.primer_elemento['radius']
        self.smoothing_level = self.primer_elemento['smoothing level']
        self.cutoff_radius = self.primer_elemento['cutoff radius']
        self.stees = self.primer_elemento['strees']
        self.columns_train=self.primer_elemento['columns_train']
        self.smoothing_level_training= self.primer_elemento['smoothing_level_training']
        self.radius_training = self.primer_elemento['radius_training']
        # Listas para almacenar los datos de los clusters
        self.clusters = []
        self.surface_area = []
        self.vecinos = []
        self.menor_norma = []
        self.mayores_norma = []
        self.coordenadas_cm = []

    def load_clusters(self, json_file):
        """Carga y parsea los datos del JSON con los clusters."""
        with open(json_file, 'r') as f:
            datos = json.load(f)
        self.clusters = datos.get("clusters", [])
        self._parse_clusters()

    def _parse_clusters(self):
        """Extrae las columnas de la matriz de clusters y las almacena en listas."""
        # Reiniciar las listas
        self.surface_area = []
        self.vecinos = []
        self.menor_norma = []
        self.mayores_norma = []
        self.coordenadas_cm = []
        for fila in self.clusters:
            self.surface_area.append(fila[0])
            self.vecinos.append(fila[1])
            self.menor_norma.append(fila[2])
            self.mayores_norma.append(fila[3])
            self.coordenadas_cm.append(fila[4:7])
        #print(self.mayores_norma)
        #print(self.menor_norma)
    
    def get_max_cluster(self):
        """Obtiene el valor máximo de 'mayores_norma', el índice y las coordenadas del centro de masa."""
        if not self.mayores_norma:
            raise ValueError("No se han cargado datos de 'mayores_norma'.")
        max_radius = np.max(self.mayores_norma)
        indice_max = np.argmax(self.mayores_norma)
        centro_masa_max = self.coordenadas_cm[indice_max]
        return max_radius, indice_max, centro_masa_max

    def process_pipeline_ids(self, output_ids_file):
        """
        Realiza modificaciones en el pipeline a partir del cluster con mayor norma y exporta 
        los IDs al archivo especificado.
        """
        max_radius, indice_max, centro_masa_max = self.get_max_cluster()
        # Construir la condición para seleccionar partículas dentro de la esfera definida.
        
        condition = (
                    f"(Position.X-{centro_masa_max[0]})*(Position.X-{centro_masa_max[0]})+"
                    f"(Position.Y-{centro_masa_max[1]})*(Position.Y-{centro_masa_max[1]})+"
                    f"(Position.Z-{centro_masa_max[2]})*(Position.Z-{centro_masa_max[2]})<="
                    f"{self.radius_training*self.radius_training}"
                    )
        
                 
        pipeline = import_file(self.relax)
        pipeline.modifiers.append(ExpressionSelectionModifier(expression=condition))
        print(condition)
        pipeline.modifiers.append(InvertSelectionModifier())
        pipeline.modifiers.append(DeleteSelectedModifier())
        id_pip=pipeline.compute()
        if id_pip.particles.count<2:
                pipeline.modifiers.clear()
                conditionn = (
                f"(Position.X-{centro_masa_max[0]})*(Position.X-{centro_masa_max[0]})+"
                f"(Position.Y-{centro_masa_max[1]})*(Position.Y-{centro_masa_max[1]})+"
                f"(Position.Z-{centro_masa_max[2]})*(Position.Z-{centro_masa_max[2]})<="
                f"{8*8}"
                )
                pipeline.modifiers.append(ExpressionSelectionModifier(expression=conditionn))
                pipeline.modifiers.append(InvertSelectionModifier())
                pipeline.modifiers.append(DeleteSelectedModifier())

        try:
            export_file(pipeline, output_ids_file, "lammps/dump",
                        columns=["Particle Identifier", "Particle Type",
                                 "Position.X", "Position.Y", "Position.Z"])
            pipeline.modifiers.clear()
        except Exception as e:
            print(f"Error al exportar el archivo: {e}")
    def augment_data(self,X, y, factor=5, noise_level=0.01):
        """
        Genera 'factor' veces más datos a partir de los datos originales X, y.
        Se agrega ruido gaussiano proporcional al valor de cada característica.
        """
        X_aug = []
        y_aug = []
        for features, target in zip(X, y):
            X_aug.append(features)  # Incluye el dato original
            y_aug.append(target)
            for _ in range(factor - 1):
                # Para cada característica, añade ruido relativo
                features_aug = [feat * (1 + np.random.normal(0, noise_level)) for feat in features]
                X_aug.append(features_aug)
                y_aug.append(target)  # Suponiendo que la perturbación no cambia la etiqueta
        return np.array(X_aug), np.array(y_aug)

    @staticmethod
    def extraer_ids(archivo):
        """
        Lee el archivo y extrae los IDs de las partículas,
        saltándose las primeras 9 líneas (por ejemplo, encabezados).
        """
        ids = []
        with open(archivo, 'r') as f:
            # Saltar las primeras 9 líneas (ajustar según el archivo)
            for _ in range(9):
                next(f)
            # Extraer el primer valor de cada línea
            for linea in f:
                valores = linea.split()
                ids.append(valores[0])
        return ids

    @staticmethod
    def crear_condicion_ids(ids_eliminar):
        """
        A partir de una lista de IDs, crea una cadena de condición con la forma:
        ParticleIdentifier==id1 || ParticleIdentifier==id2 || ...
        """
        condicion = " || ".join([f"ParticleIdentifier=={id}" for id in ids_eliminar])
        return condicion

    def compute_max_distance(self, data):
        positions = data.particles.positions
        if positions.size == 0:
            # Puedes decidir qué valor devolver o lanzar un error controlado
            return 0  # o: raise ValueError("No hay partículas para calcular la distancia.")
        center_of_mass = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center_of_mass, axis=1)
        if distances.size == 0:
            return 0  # o lanzar un error controlado
        return np.max(distances)

    
    def compute_min_distance(self, data):
        positions = data.particles.positions
        if positions.size == 0:
            # Puedes decidir qué valor devolver o lanzar un error controlado
            return 0  # o: raise ValueError("No hay partículas para calcular la distancia.")
        center_of_mass = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center_of_mass, axis=1)
        if distances.size == 0:
            return 0  # o lanzar un error controlado
        return np.min(distances)

    def run_training(self, ids_file, output_training_file):
        """
        Ejecuta la iteración en la que se eliminan partículas incrementalmente, 
        se calcula la malla superficial, se cuentan los vecinos y se calculan las mayores
        y menores distancias al centro de masa (dimensiones del nanoporo). Los resultados se exportan
        en formato JSON. Además, se generan datos aumentados y se exportan a 'training_cluster.json'.
        """
        total_ids = self.extraer_ids(ids_file)
        
        # Transformación afín inicial
        pipeline_trans = import_file(self.relax)
        pipeline_trans.modifiers.append(AffineTransformationModifier(
            operate_on={'particles','cell'},
            transformation=[[self.stees[0], 0, 0, 0],
                            [0, self.stees[1], 0, 0],
                            [0, 0, self.stees[2], 0]]
        ))
        pipeline_trans.compute()
        try:
            export_file(pipeline_trans, "outputs.dump/train_affine_transformation.dump", "lammps/dump", 
                        columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"])
        except Exception as e:
            print(f"Error al exportar el archivo: {e}")
        
        # Cargamos el pipeline transformado para generar los datos de entrenamiento
        pipeline_2 = import_file("outputs.dump/train_affine_transformation.dump")
        sm_mesh_training = []
        vacancias = []
        vecinos = []
        max_distancias = []  # Mayor distancia al centro de masa
        min_distancias = []  # Menor distancia al centro de masa
        
        # Bucle para generar los datos de entrenamiento (originales)
        for index in range(len(total_ids)):
            ids_a_eliminar = total_ids[:index+1]
            condition_f = self.crear_condicion_ids(ids_a_eliminar)
            pipeline_2.modifiers.append(ExpressionSelectionModifier(expression=condition_f))
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            pipeline_2.modifiers.append(ConstructSurfaceModifier(
                radius=self.radius,
                smoothing_level=self.smoothing_level_training,
                identify_regions=True,
                select_surface_particles=True
            ))
            data_2 = pipeline_2.compute()
            sm_elip = data_2.attributes.get('ConstructSurfaceMesh.surface_area', 0)
            sm_mesh_training.append(sm_elip)
            vacancias.append(index+1)
            
            pipeline_2.modifiers.append(InvertSelectionModifier())
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            data_3 = pipeline_2.compute()
            max_dist = self.compute_max_distance(data_3)
            min_dist = self.compute_min_distance(data_3)
            max_distancias.append(max_dist)
            min_distancias.append(min_dist)
            vecinos.append(data_3.particles.count)
            
            pipeline_2.modifiers.clear()
        
        # Exporta el archivo original de entrenamiento
        datos_exportar = {
            "sm_mesh_training": sm_mesh_training,
            "vacancias": vacancias,
            "vecinos": vecinos,
            "max_distancias": max_distancias,
            "min_distancias": min_distancias
        }
        with open(output_training_file, "w") as f:
            json.dump(datos_exportar, f, indent=4)
        
        # --------------------------------------------------------------------------------
        # Generar datos aumentados a partir de los datos originales.
        #
        # Por ejemplo, si queremos usar 4 características (área, vecinos, máxima y mínima distancia),
        # combinamos estas listas en una matriz X y el vector objetivo y es 'vacancias'
        X = np.array(list(zip(sm_mesh_training, vecinos, max_distancias, min_distancias)))
        y = np.array(vacancias)
        
        # Llamamos a la función para aumentar los datos
        X_aug, y_aug = self.augment_data(X, y, factor=5, noise_level=0.01)
        
        # Preparamos el diccionario para exportar. Aquí se guarda la matriz de características
        # y el vector de vacancias aumentados.
        datos_exportar_aug = {
            "sm_mesh_training": X_aug.tolist(),   # Cada fila: [sm_mesh, vecinos, max_dist, min_dist]
            "vacancias": y_aug.tolist()
        }
        
        # Exportamos el archivo JSON con los datos aumentados.
        output_aug_file = "training_cluster.json"
        with open(output_aug_file, "w") as f:
            json.dump(datos_exportar_aug, f, indent=4)
        
        print(f"Datos originales exportados en: {output_training_file}")
        print(f"Datos aumentados exportados en: {output_aug_file}")
