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
    ClusterAnalysisModifier
)

class ClusterProcessor:
    def __init__(self, json_critical_files, cutoff_radius):
        # Rutas y parámetros
        self.critic_files = self.cargar_critical_files(json_critical_files)
        self.cutoff_radius = cutoff_radius

    ##########################################################
    #               M E T O D O S   U T I L E S              #
    ##########################################################

    def cargar_critical_files(self, ruta_json):
        with open(ruta_json, "r") as f:
            data = json.load(f)
        return data

    def leer_lammps_dump(self, ruta_archivo):
        data = []
        leer_flag = False  # Indica si estamos en la sección que contiene las coordenadas
        with open(ruta_archivo, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith("ITEM: ATOMS"):
                    leer_flag = True
                    continue
                if line.startswith("ITEM:") and not line.startswith("ITEM: ATOMS"):
                    leer_flag = False
                if leer_flag and line:
                    partes = line.split()
                    data.append(partes)
        columnas = ["id", "type", "x", "y", "z", "Cluster"]
        df = pd.DataFrame(data, columns=columnas)
        df = df.astype({
            "id": int,
            "type": int,
            "x": float,
            "y": float,
            "z": float,
            "Cluster": int
        })
        return df

    def extraer_xyz_from_array(self, matriz):
        # Columns x=2, y=3, z=4
        return matriz[:, 2:5]

    def calcular_distancias_y_dispersion_total(self, coords):
        num_puntos = coords.shape[0]
        if num_puntos < 2:
            return np.array([]), 0.0  # No se pueden calcular distancias con menos de 2 puntos
        distancias = pdist(coords, metric='euclidean')
        dispersion = np.std(distancias)
        return distancias, dispersion

    def calcular_centro_de_masa(self, coords):
        if coords.shape[0] == 0:
            return np.array([0.0, 0.0, 0.0])
        return np.mean(coords, axis=0)

    def calcular_distancias_y_puntos_mas_alejados(self, coords):
        num_puntos = len(coords)
        if num_puntos < 2:
            raise ValueError("Se necesitan al menos dos puntos para calcular distancias.")
        max_dist = -1
        punto_A, punto_B = None, None
        for i in range(num_puntos):
            for j in range(i + 1, num_puntos):
                dist_ij = np.linalg.norm(coords[i] - coords[j])
                if dist_ij > max_dist:
                    max_dist = dist_ij
                    punto_A, punto_B = coords[i], coords[j]
        return max_dist, punto_A, punto_B

    def calcular_distancias_y_dispersion(self, coords, punto):
        if coords.shape[0] == 0:
            return np.array([]), 0.0
        distancias = np.linalg.norm(coords - punto, axis=1)
        dispersion = np.std(distancias)
        return distancias, dispersion

    def separar_tres_clusters(self, df):
        cluster_1 = np.array([])
        cluster_2 = np.array([])
        cluster_3 = np.array([])
        if 0 in df["Cluster"].values:
            cluster_1 = df[df["Cluster"] == 0][["x", "y", "z"]].values
        if 1 in df["Cluster"].values:
            cluster_2 = df[df["Cluster"] == 1][["x", "y", "z"]].values
        if 2 in df["Cluster"].values:
            cluster_3 = df[df["Cluster"] == 2][["x", "y", "z"]].values
        return cluster_1, cluster_2, cluster_3

    def aplicar_kmeans_dot(self, coordenadas, cm_1, cm_2, cm_3):
        kmeans = KMeans(n_clusters=3, init=np.array([cm_1, cm_2, cm_3]))
        etiquetas = kmeans.fit_predict(coordenadas)
        return etiquetas

    def exportar_matriz_a_txt(self, matriz, nombre_archivo):
        fmt = ["%d", "%d", "%.6f", "%.6f", "%.6f", "%d"]
        np.savetxt(nombre_archivo, matriz, fmt=fmt, delimiter=" ")
        print(f"✅ Matriz exportada exitosamente a: {nombre_archivo}")

    def copiar_encabezado_y_exportar(self, archivo_entrada, matriz, archivo_salida):
        encabezado = []
        with open(archivo_entrada, "r") as f:
            for line in f:
                encabezado.append(line)
                if line.startswith("ITEM: ATOMS"):
                    break
        fmt = ["%d", "%d", "%.6f", "%.6f", "%.6f", "%d"]
        with open(archivo_salida, "w") as f:
            f.writelines(encabezado)
            np.savetxt(f, matriz, fmt=fmt, delimiter=" ")
        print(f"✅ Archivo exportado con encabezado en: {archivo_salida}")

    ##########################################################
    #               P I P E L I N E   P R I N C I P A L       #
    ##########################################################

    def procesar_clusters(self):
        for cl in self.critic_files:
            # 1) Leer y hacer cluster analysis con Ovito
            pipeline = import_file(cl)
            pipeline.modifiers.append(ClusterAnalysisModifier(
                cutoff=self.cutoff_radius,
                cluster_coloring=True,
                unwrap_particles=True,
                sort_by_size=True
            ))
            pipeline.compute()
            export_file(pipeline, cl, "lammps/dump",
                        columns=["Particle Identifier", "Particle Type",
                                 "Position.X", "Position.Y", "Position.Z","Cluster"])
            pipeline.modifiers.clear()

            # 2) Procesar DataFrame
            df_datos = self.leer_lammps_dump(cl)
            print("DataFrame con las columnas extraídas:")
            print(df_datos)
            matriz = df_datos.values
            xyz_array = self.extraer_xyz_from_array(matriz)
            print(xyz_array)

            # 3) Calcular centro de masa y puntos más alejados
            centro_masa = self.calcular_centro_de_masa(xyz_array)
            print("\nCentro de masa (cx, cy, cz):")
            print(centro_masa)
            max_dist, pA, pB = self.calcular_distancias_y_puntos_mas_alejados(xyz_array)
            print(f"maxima distancia : {max_dist}")

            # 4) Aplicar KMeans inicial
            etiquetas_dot = self.aplicar_kmeans_dot(xyz_array, centro_masa, pA, pB)
            print(etiquetas_dot)
            if len(df_datos) == len(etiquetas_dot):
                df_datos["Cluster"] = etiquetas_dot
                print("\nDataFrame con etiquetas actualizadas:")
                print(df_datos)
            else:
                print(f"⚠️ Error: Número de etiquetas ({len(etiquetas_dot)}) ",
                      f"no coincide con el número de filas ({len(df_datos)}) en el DataFrame.")

            # 5) Exportar la matriz a un archivo de texto
            self.exportar_matriz_a_txt(df_datos.values, "datos_exportados.txt")

            # 6) Separar en tres clusters y calcular dispersiones
            cluster_1, cluster_2, cluster_3 = self.separar_tres_clusters(df_datos)
            clusters_t = [cluster_1, cluster_2, cluster_3]
            dispersions = []

            for group in clusters_t:
                if len(group) > 0:
                    cm = self.calcular_centro_de_masa(group)
                    _, dispersion = self.calcular_distancias_y_dispersion(group, cm)
                    dispersions.append(dispersion)
                else:
                    dispersions.append(np.inf)

            # 7) Identificar cluster con menor dispersion y fusionar los demas
            idx_min_dispersion = np.argmin(dispersions)
            print(f"🔹 Cluster con menor dispersión: {idx_min_dispersion}")
            cluster_labels = [0, 1, 2]
            cluster_fijo = cluster_labels[idx_min_dispersion]
            clusters_a_unir = [i for i in range(3) if i != idx_min_dispersion]
            nuevo_cluster_label = min(set(cluster_labels) - {cluster_fijo})
            print(f"🔄 Se mantiene el cluster {cluster_fijo}, "
                  f"los clusters {clusters_a_unir} se fusionan en {nuevo_cluster_label}")

            df_datos.loc[df_datos["Cluster"] == clusters_a_unir[0], "Cluster"] = nuevo_cluster_label
            df_datos.loc[df_datos["Cluster"] == clusters_a_unir[1], "Cluster"] = nuevo_cluster_label

            # 8) Calcular la dispersión final de los clusters unificados
            dispersion_clusters = {}
            for i, cluster in enumerate([cluster_1, cluster_2]):
                if len(cluster) > 0:
                    cm = self.calcular_centro_de_masa(cluster)
                    _, dispersion = self.calcular_distancias_y_dispersion(cluster, cm)
                    dispersion_clusters[i] = dispersion
                    print(f"📊 Dispersión del Cluster {i}: {dispersion:.6f}")

            clusters_excedidos = [i for i, disp in dispersion_clusters.items() if disp > 1]
            if clusters_excedidos:
                print("\n⚠️ Clusters con dispersión > 1 detectados:")
                for cluster_id in clusters_excedidos:
                    print(f"🔴 Cluster {cluster_id} tiene una dispersión de {dispersion_clusters[cluster_id]:.6f}")

            # 9) Exportar la matriz con clusters fusionados
            self.copiar_encabezado_y_exportar(cl, df_datos.values, cl)
            print("\nDataFrame con clusters fusionados:")
            print(df_datos)

            # 10) Dividir el cluster en dos: Cluster == 0, Cluster == 1 (en un for i in range(0,2))
            #    y verificar dispersion para cada sub-archivo

            data_json = []
            for i in range(0, 2):
                pipeline = import_file(cl)
                pipeline.modifiers.append(ExpressionSelectionModifier(expression=f"Cluster=={i}"))
                pipeline.modifiers.append(DeleteSelectedModifier())
                pipeline.compute()

                export_file(pipeline, f"{cl}.{i}", "lammps/dump",
                            columns=["Particle Identifier", "Particle Type",
                                     "Position.X", "Position.Y", "Position.Z","Cluster"])
                pipeline.modifiers.clear()

                df_sub = self.leer_lammps_dump(f"{cl}.{i}")
                matriz_sub = df_sub.values
                xyz_array_sub = self.extraer_xyz_from_array(matriz_sub)
                print(xyz_array_sub)
                cm_sub = self.calcular_centro_de_masa(xyz_array_sub)
                _, dispersion_sub = self.calcular_distancias_y_dispersion(xyz_array_sub, cm_sub)
                print(f"dispersion: {dispersion_sub}")

                # 10a) Si dispersion > 1.1, exportar a JSON critico
                if dispersion_sub > 1.1:
                    data_json.append(f"{cl}.{i}")
                    with open(f"outputs.json/clusters_criticos_iteracion_{i}.json", "w") as f:
                        json.dump(data_json, f, indent=4)

                # 10b) Si no, actualizar JSON: se elimina "{cl}" y se agrega "{cl}.{i}"
                else:
                    json_file = "outputs.json/lista_nombres_clusters.json"
                    if os.path.exists(json_file):
                        with open(json_file, "r") as file:
                            try:
                                data = json.load(file)
                                if not isinstance(data, list):
                                    raise ValueError("El archivo JSON no contiene una lista válida.")
                            except json.JSONDecodeError:
                                print("Error: El archivo JSON está corrupto o vacío.")
                                data = []
                    else:
                        data = []

                    nombre_a_eliminar = f"{cl}"
                    nombres_a_agregar = [f"{cl}.{i}"]

                    if nombre_a_eliminar in data:
                        data.remove(nombre_a_eliminar)

                    for nuevo_nombre in nombres_a_agregar:
                        if nuevo_nombre not in data:
                            data.append(nuevo_nombre)

                    with open(json_file, "w") as file:
                        json.dump(data, file, indent=4)
                    print("Archivo actualizado correctamente.")

            print(f"\n✅ Proceso finalizado para: {cl}\n")


# Uso de la clase (para integración con main)
if __name__ == "__main__":
    from input_params import LAYERS
    primer_elemento = LAYERS[0]
    cut_radius = primer_elemento['cutoff radius']
    cluster_processor = ClusterProcessor("outputs.json/critic_files.json", cut_radius)
    cluster_processor.procesar_clusters()
