import pandas as pd
import numpy as np
import json
import os
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier, ClusterAnalysisModifier
from va_input_params import LAYERS

class ClusterProcessor:
    def __init__(self, json_critical_files, cutoff_radius, cutoff_cluster, cluster_divisions):
        self.critic_files = self.cargar_critical_files(json_critical_files)
        
        self.cutoff_radius = cutoff_radius
        self.cutoff_cluster = cutoff_cluster
        self.cl_div = cluster_divisions

    def cargar_critical_files(self, ruta_json):
        with open(ruta_json, "r") as f:
            data = json.load(f)
        return data

    def leer_lammps_dump(self, ruta_archivo):
        data = []
        leer_flag = False
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
        return matriz[:, 2:5]

    def calcular_distancias_y_dispersion_total(self, coords):
        num_puntos = coords.shape[0]
        if num_puntos < 2:
            return np.array([]), 0.0
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
        # Método original en función de self.cl_div (1, 2 o 3)
        if self.cl_div == 3:
            kmeans = KMeans(n_clusters=3, init=np.array([cm_1, cm_2, cm_3]), n_init=1)
            etiquetas = kmeans.fit_predict(coordenadas)
        elif self.cl_div == 2:
            kmeans = KMeans(n_clusters=2, init=np.array([cm_2, cm_3]), n_init=1)
            etiquetas = kmeans.fit_predict(coordenadas)
        else:
            kmeans = KMeans(n_clusters=1, init=np.array([cm_1]), n_init=1)
            etiquetas = kmeans.fit_predict(coordenadas)
        return etiquetas

    def eliminar_nombre_del_archivo(self,nombre, ruta_json):
        # Verificar si el archivo existe
        if not os.path.exists(ruta_json):
            print(f"El archivo {ruta_json} no existe.")
            return

        # Cargar la lista de nombres desde el archivo JSON
        with open(ruta_json, 'r') as file:
            try:
                lista_nombres = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error al leer el archivo JSON: {e}")
                return

        # Verificar si el nombre está en la lista y eliminarlo
        if nombre in lista_nombres:
            lista_nombres.remove(nombre)
            # Guardar la lista actualizada en el archivo JSON
            with open(ruta_json, 'w') as file:
                json.dump(lista_nombres, file, indent=4)
            print(f"El nombre '{nombre}' ha sido eliminado del archivo.")
        else:
            print(f"El nombre '{nombre}' no se encontró en el archivo.")


    def aplicar_kmeans_dot_optimo(self, coordenadas, cm_1, cm_2, cm_3):
        """
        Itera sobre los posibles números de clusters (1, 2 y 3) utilizando los centros
        iniciales indicados para cada caso, calcula la dispersión promedio (en cada cluster)
        y selecciona la partición que minimiza la dispersión.
        """
        mejores_etiquetas = None
        mejor_dispersion = np.inf
        mejor_cl_div = None

        # Iteramos sobre los posibles números de clusters: 1, 2 y 3.
        for n_clusters in [1, 2, 3]:
            if n_clusters == 3:
                init_centers = np.array([cm_1, cm_2, cm_3])
            elif n_clusters == 2:
                init_centers = np.array([cm_2, cm_3])
            else:
                init_centers = np.array([cm_1])

            # Especificamos n_init=1 para usar nuestros centros iniciales sin re-inicializaciones.
            kmeans = KMeans(n_clusters=n_clusters, init=init_centers, n_init=1)
            etiquetas = kmeans.fit_predict(coordenadas)

            # Calculamos la dispersión promedio de los clusters obtenidos.
            dispersion_total = 0
            for cluster_label in range(n_clusters):
                cluster_points = coordenadas[etiquetas == cluster_label]
                if cluster_points.shape[0] == 0:
                    continue
                cm_cluster = self.calcular_centro_de_masa(cluster_points)
                _, dispersion = self.calcular_distancias_y_dispersion(cluster_points, cm_cluster)
                dispersion_total += dispersion

            dispersion_promedio = dispersion_total / n_clusters

            if dispersion_promedio < mejor_dispersion:
                mejor_dispersion = dispersion_promedio
                mejores_etiquetas = etiquetas
                mejor_cl_div = n_clusters

        # Actualizamos el atributo cl_div según la opción óptima encontrada.
        self.cl_div = mejor_cl_div
        return mejores_etiquetas

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

    def procesar_clusters(self):
        for cl in self.critic_files:
            pipeline = import_file(cl)
            print(cl)
            ruta_archivo = "outputs.json/critic_files.json"
            
            self.eliminar_nombre_del_archivo(cl, ruta_archivo)
            pipeline.modifiers.append(ClusterAnalysisModifier(
                cutoff=self.cutoff_radius,
                cluster_coloring=True,
                unwrap_particles=True,
                sort_by_size=True
            ))
            pipeline.compute()
            export_file(pipeline, cl, "lammps/dump",
                        columns=["Particle Identifier", "Particle Type",
                                 "Position.X", "Position.Y", "Position.Z", "Cluster"])
            
            pipeline.modifiers.clear()
            df_datos = self.leer_lammps_dump(cl)
            matriz = df_datos.values
            xyz_array = self.extraer_xyz_from_array(matriz)
            centro_masa = self.calcular_centro_de_masa(xyz_array)
            max_dist, pA, pB = self.calcular_distancias_y_puntos_mas_alejados(xyz_array)
            
            # Usamos el método optimizado para seleccionar la división de clusters con menor dispersión
            etiquetas_dot = self.aplicar_kmeans_dot_optimo(xyz_array, centro_masa, pA, pB)
            
            if len(df_datos) == len(etiquetas_dot):
                df_datos["Cluster"] = etiquetas_dot
            else:
                print(f"⚠️ Error: Número de etiquetas ({len(etiquetas_dot)}) no coincide con el número de filas ({len(df_datos)}) en el DataFrame.")
            
            self.exportar_matriz_a_txt(df_datos.values, "datos_exportados.txt")
            
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
            idx_min_dispersion = np.argmin(dispersions)
            cluster_labels = [0, 1, 2]
            cluster_fijo = cluster_labels[idx_min_dispersion]
            clusters_a_unir = [i for i in range(3) if i != idx_min_dispersion]
            nuevo_cluster_label = min(set(cluster_labels) - {cluster_fijo})
            df_datos.loc[df_datos["Cluster"] == clusters_a_unir[0], "Cluster"] = nuevo_cluster_label
            df_datos.loc[df_datos["Cluster"] == clusters_a_unir[1], "Cluster"] = nuevo_cluster_label
            
            dispersion_clusters = {}
            for i, cluster in enumerate([cluster_1, cluster_2]):
                if len(cluster) > 0:
                    cm = self.calcular_centro_de_masa(cluster)
                    _, dispersion = self.calcular_distancias_y_dispersion(cluster, cm)
                    dispersion_clusters[i] = dispersion
            clusters_excedidos = [i for i, disp in dispersion_clusters.items() if disp > 1.7]
            
            self.copiar_encabezado_y_exportar(cl, df_datos.values, cl)
            
            data_json = []
            for i in range(0, 2):
                pipeline = import_file(cl)
                pipeline.modifiers.append(ExpressionSelectionModifier(expression=f"Cluster=={i}"))
                pipeline.modifiers.append(DeleteSelectedModifier())
                pipeline.compute()
                export_file(pipeline, f"{cl}.{i}", "lammps/dump",
                            columns=["Particle Identifier", "Particle Type",
                                     "Position.X", "Position.Y", "Position.Z", "Cluster"])
                pipeline.modifiers.clear()
                df_sub = self.leer_lammps_dump(f"{cl}.{i}")
                matriz_sub = df_sub.values
                xyz_array_sub = self.extraer_xyz_from_array(matriz_sub)
                cm_sub = self.calcular_centro_de_masa(xyz_array_sub)
                _, dispersion_sub = self.calcular_distancias_y_dispersion(xyz_array_sub, cm_sub)
                print(f"dispersion: {dispersion_sub}")
                if dispersion_sub > self.cutoff_cluster:
                    data_json.append(f"{cl}.{i}")
                    with open(f"outputs.json/clusters_criticos_iteracion_{i}.json", "w") as f:
                        json.dump(data_json, f, indent=4)
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
            print(f"\n✅ Proceso finalizado para: {cl}\n")
