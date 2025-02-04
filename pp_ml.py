import pandas as pd
import numpy as np
import json
import os
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
import ovito
from ovito.io import import_file, export_file
from ovito.modifiers import (
    ExpressionSelectionModifier,
    DeleteSelectedModifier,
    ClusterAnalysisModifier,
    ConstructSurfaceModifier
)

class ClusterProcessor:
    def __init__(self, json_critical_files, cutoff_radius, smoothing_level):
        self.critic_files = self.cargar_critical_files(json_critical_files)
        self.cutoff_radius = cutoff_radius
        self.smoothing_level = smoothing_level

    def cargar_critical_files(self, ruta_json):
        with open(ruta_json, "r") as f:
            return json.load(f)
    def unificar_formato(self):
        for cl in self.critic_files:
                pipeline=import_file(cl)
                pipeline.modifiers.append(ClusterAnalysisModifier(
                    cutoff=self.cutoff_radius,
                    cluster_coloring=True,
                    unwrap_particles=True,
                    sort_by_size=True,
                    compute_com=True
                ))
                date = pipeline.compute()
                try:
                    export_file(pipeline, cl, "lammps/dump",
                                columns=["Particle Identifier", "Particle Type",
                                        "Position.X", "Position.Y", "Position.Z","Cluster"])
                    pipeline.modifiers.clear()
                except Exception as e:
                    print(f"Error al exportar el archivo: {e}")


    def extraer_coordenadas(self, archivo):
        coordenadas = []
        with open(archivo, 'r') as f:
            for _ in range(9):
                next(f)
            for linea in f:
                id_, tipo, x, y, z,cluster = linea.split()
                coordenadas.append((float(x), float(y), float(z)))
        return coordenadas

    def procesar_clusters(self):
        num_cluster = len(self.critic_files)
        matriz = np.zeros((num_cluster, 7))
        for i, archivo_nombre in enumerate(self.critic_files, start=1):
            print(f"#################### Critical Area {i} ###############")
            max_sm = 0
            j_max_sm = 0

            for j in range(round(self.cutoff_radius), 7):
                pipeline_3 = import_file(archivo_nombre)
                pipeline_3.modifiers.append(ConstructSurfaceModifier(
                    radius=j,
                    smoothing_level=self.smoothing_level,
                    identify_regions=True,
                    select_surface_particles=True
                ))
                data_3 = pipeline_3.compute()
                vecinos = data_3.particles.count

                if max_sm < data_3.attributes['ConstructSurfaceMesh.surface_area']:
                    max_sm = data_3.attributes['ConstructSurfaceMesh.surface_area']
                    j_max_sm = j

                coordenadas = self.extraer_coordenadas(archivo_nombre)

                pipeline_3.modifiers.append(ClusterAnalysisModifier(
                    cutoff=self.cutoff_radius,
                    cluster_coloring=True,
                    unwrap_particles=True,
                    sort_by_size=True,
                    compute_com=True
                ))
                data_3 = pipeline_3.compute()
                cluster_table_3 = data_3.tables['clusters']
                centro_masa = cluster_table_3['Center of Mass'][0]

                vectores = []
                for coord in coordenadas:
                    x, y, z = coord
                    vector_cm_position = (x - centro_masa[0], y - centro_masa[1], z - centro_masa[2])
                    norma_vector = np.linalg.norm(vector_cm_position)
                    vectores.append((vector_cm_position, norma_vector))

                vector_mayor_norma = max(vectores, key=lambda x: x[1])[0]
                vector_menor_norma = min(vectores, key=lambda x: x[1])[0]
                norma_mayor = np.linalg.norm(vector_mayor_norma)
                norma_menor = np.linalg.norm(vector_menor_norma)

                matriz[i - 1, 0] = max_sm
                matriz[i - 1, 1] = vecinos
                matriz[i - 1, 2] = norma_menor
                matriz[i - 1, 3] = norma_mayor
                matriz[i - 1, 4] = centro_masa[0]
                matriz[i - 1, 5] = centro_masa[1]
                matriz[i - 1, 6] = centro_masa[2]

                pipeline_3.modifiers.clear()

        matriz_lista = matriz.tolist()
        datos_finales = {
            "num_clusters": num_cluster,
            "clusters": matriz_lista
        }

        os.makedirs("outputs.json", exist_ok=True)
        with open("outputs.json/key_areas_matrix_FINAL.json", "w") as archivo:
            json.dump(datos_finales, archivo, indent=4)


