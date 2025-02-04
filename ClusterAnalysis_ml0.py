import random
import numpy as np
import json
import pandas as pd
import ovito
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier

from ExportClusterDate import ProcesadorArchivo
from LAMMPS_formater import EncabezadoLammps

class CriticalClusterAnalyzer:
 

    def __init__(self):
       
        self.bClusteringAnalysis = False
        self.LISTA_ARCHIVOS_IMPORTANTES = []

    def extraer_datos(self, archivo):
        datos = []
        con_datos = False

        with open(archivo, 'r') as f:
            for linea in f:
                if linea.strip() == "ITEM: ATOMS id type x y z Cluster":
                    con_datos = True
                elif con_datos:
                    valores = linea.split()
                    if len(valores) == 6:
                        datos.append([
                            int(valores[0]),
                            int(valores[1]),
                            float(valores[2]),
                            float(valores[3]),
                            float(valores[4]),
                            float(valores[5])
                        ])
        return datos

    def calcular_centro_masa(self, datos):
        if len(datos) == 0:
            return None
        sum_x, sum_y, sum_z = 0, 0, 0
        num_atomos = len(datos)
        for atomo in datos:
            sum_x += atomo[0]
            sum_y += atomo[1]
            sum_z += atomo[2]
        centro_masa = (sum_x / num_atomos, sum_y / num_atomos, sum_z / num_atomos)
        return centro_masa

    def calcular_centro_masa_coordenadas(self, datos):
        if len(datos) == 0:
            return None
        sum_x, sum_y, sum_z = 0, 0, 0
        num_atomos = len(datos)
        for atomo in datos:
            sum_x += atomo[0]
            sum_y += atomo[1]
            sum_z += atomo[2]
        centro_masa = (sum_x / num_atomos, sum_y / num_atomos, sum_z / num_atomos)
        return centro_masa

    def extraer_datos_por_cluster(self, datos, cluster):
        datos_por_cluster = []
        for atomo in datos:
            if atomo[5] == cluster:
                datos_por_cluster.append([
                    atomo[0],
                    atomo[2],
                    atomo[3],
                    atomo[4],
                    atomo[5]
                ])
        centro_masa = self.calcular_centro_masa(datos_por_cluster)
        return datos_por_cluster, centro_masa

    def calcular_distancia(self, punto1, punto2):
        return np.linalg.norm(np.array(punto1) - np.array(punto2))

    def calcular_mayor_distancia(self, coordenadas):
        mayor_distancia = 0
        puntos_mayor_distancia = None
        for i in range(len(coordenadas)):
            for j in range(i + 1, len(coordenadas)):
                distancia = self.calcular_distancia(coordenadas[i], coordenadas[j])
                if distancia > mayor_distancia:
                    mayor_distancia = distancia
                    puntos_mayor_distancia = (coordenadas[i], coordenadas[j])
        return mayor_distancia, puntos_mayor_distancia

    def calcular_distancias_y_varianza(self, datos, centro_masas):
        centro_masa = centro_masas
        datos_por_cluster = datos
        if centro_masa is None:
            return [], 0

        distancias = []
        for atomo in datos_por_cluster:
            x, y, z = atomo[2], atomo[3], atomo[4]
            distancia = np.sqrt((x - centro_masa[0]) ** 2 + (y - centro_masa[1]) ** 2 + (z - centro_masa[2]) ** 2)
            distancias.append(distancia)

        media = sum(distancias) / len(distancias)
        varianza = sum((distancia - media) ** 2 for distancia in distancias) / len(distancias)

        return distancias, varianza

    def modificar_indice_cluster(self, num_cluster, num_cluster_perteneciente, etiquetas):
        for i in range(len(etiquetas)):
            if etiquetas[i] == 1:
                etiquetas[i] = num_cluster + 1
            elif etiquetas[i] == 0:
                etiquetas[i] = num_cluster_perteneciente
        return etiquetas

    def run(self):


        bClusteringAnalysis = False  

        with open('outputs.json/key_areas_matrix.json', 'r') as archivo:
            cm_datos = json.load(archivo)

        centro_masas = []
        for cluster in cm_datos['clusters']:
            primeras_tres_componentes = cluster[:3]
            centro_masas.append(primeras_tres_componentes)

        with open('outputs.json/centros_masas.json', 'w') as archivo_salida:
            json.dump(centro_masas, archivo_salida, indent=4)

        archivo = "outputs.dump/key_areas_clustering.dump"
        datos = self.extraer_datos(archivo)

        with open('outputs.json/matrix_t.json', 'w') as f:
            json.dump(datos, f)

        with open('outputs.json/clusters.json', 'r') as f:
            clusters = json.load(f)

        num_clusters = clusters['num_clusters']
        print(f"numero de clusters:{num_clusters}")

        numero_cluster_critic = []
        for i in range(1, num_clusters + 1):
            coordenadas = []
            cm_nuevo = []
            cluster_1 = []
            cluster_2 = []
            cluster_3 = []
            cluster_particion = []
            cluster = i

            with open(f'outputs.json/cluster_{i}.json', 'r') as archivo:
                datos_cluster_i = json.load(archivo)

            coordenadas = [fila[2:5] for fila in datos_cluster_i]
            datos_por_cluster = [fila[0:6] for fila in datos_cluster_i]
            for coordenada in coordenadas:
                print(coordenada)

            distancias, varianza = self.calcular_distancias_y_varianza(
                datos_por_cluster, 
                self.calcular_centro_masa(coordenadas)
            )
            print('Varianza de las distancias:', varianza)

            self.LISTA_ARCHIVOS_IMPORTANTES.append(f"outputs.dump/key_area_{i}.dump")
            if varianza > 1:
                numero_cluster_critic.append(i)

        print(f"numero de cluster criticos: {numero_cluster_critic}")




        cm_nuevo = []
        cluster_1 = []
        cluster_2 = []
        cluster_3 = []
        cluster_particion = []
        dispersion_sentinela = 3
        desviacion_cluster_1 = 3
        desviacion_cluster_2 = 3

        for cluster in numero_cluster_critic:
            datos_por_cluster = []
            print(f"#############cluster:{cluster}########################")

            with open(f'outputs.json/cluster_{cluster}.json', 'r') as archivo:
                datos_cluster = json.load(archivo)

            # Extraer las coordenadas x, y, z
            coordenadas = [fila[2:5] for fila in datos_cluster]
            datos_por_cluster = [fila[0:6] for fila in datos_cluster]

            for coordenada in datos_por_cluster:
                print(coordenada)

            centro_masa = self.calcular_centro_masa(coordenadas)

            mayor_distancia, puntos_mayor_distancia = self.calcular_mayor_distancia(coordenadas)
            print("La mayor distancia es:", mayor_distancia)
            print("Los puntos que corresponden a la mayor distancia son:", puntos_mayor_distancia)

            clustering = ProcesadorArchivo(coordenadas)
            etiquetas_dot = clustering.aplicar_kmeans_dot(
                puntos_mayor_distancia[0], 
                puntos_mayor_distancia[1], 
                centro_masa
            )
            print(etiquetas_dot)

            cluster_1.clear()
            cluster_2.clear()
            cluster_3.clear()
            for j in range(len(datos_por_cluster)):
                datos_por_cluster[j][5] = etiquetas_dot[j]
                if datos_por_cluster[j][5] == 0:
                    cluster_1.append([datos_por_cluster[j][2:5]])
                elif datos_por_cluster[j][5] == 1:
                    cluster_2.append([datos_por_cluster[j][2:5]])
                else:
                    cluster_3.append([datos_por_cluster[j][2:5]])

            # Centros de masa de cada sub-cluster
            centro_masa_cluster_1 = np.mean(np.array(cluster_1), axis=0) if len(cluster_1) else None
            centro_masa_cluster_2 = np.mean(np.array(cluster_2), axis=0) if len(cluster_2) else None
            centro_masa_cluster_3 = np.mean(np.array(cluster_3), axis=0) if len(cluster_3) else None

            # Distancias
            distancias_cluster_1 = np.linalg.norm(np.array(cluster_1) - centro_masa_cluster_1, axis=1) if len(cluster_1) else []
            distancias_cluster_2 = np.linalg.norm(np.array(cluster_2) - centro_masa_cluster_2, axis=1) if len(cluster_2) else []
            distancias_cluster_3 = np.linalg.norm(np.array(cluster_3) - centro_masa_cluster_3, axis=1) if len(cluster_3) else []

            # Dispersión
            dispersion_cluster_1 = np.std(distancias_cluster_1) if len(distancias_cluster_1) else 0
            dispersion_cluster_2 = np.std(distancias_cluster_2) if len(distancias_cluster_2) else 0
            dispersion_cluster_3 = np.std(distancias_cluster_3) if len(distancias_cluster_3) else 0

            print("Dispersión del cluster 1:", dispersion_cluster_1)
            print("Dispersión del cluster 2:", dispersion_cluster_2)
            print("Dispersión del cluster 3:", dispersion_cluster_3)

            indice_particion = 1
            if dispersion_cluster_1 < 1.2:
                indice_particion = 0
            elif dispersion_cluster_2 < 1.2:
                indice_particion = 1
            elif dispersion_cluster_3 < 1.2:
                indice_particion = 2
            print(f"indice de particion:{indice_particion}")

            cluster_particion.clear()
            for j in range(len(datos_por_cluster)):
                print(datos_por_cluster[j][:6])
                if datos_por_cluster[j][5] == indice_particion:
                    cluster_particion.append([datos_por_cluster[j][0:6]])

            for j in range(len(datos_por_cluster)):
                datos_por_cluster[j][5] = etiquetas_dot[j]
                if datos_por_cluster[j][5] != indice_particion:
                    datos_por_cluster[j][5] = cluster
                else:
                    datos_por_cluster[j][5] = 3

            with open(f'outputs.json/cluster_{cluster}_particion', 'w') as archivo:
                for fila in cluster_particion:
                    archivo.write(' '.join(map(str, fila[0])) + '\n')

            with open(f'outputs.json/cluster_{cluster}_actualizado', 'w') as archivo:
                for fila in datos_por_cluster:
                    archivo.write(' '.join(map(str, fila)) + '\n')

            df = pd.DataFrame(datos_por_cluster, columns=['id', 'type','x', 'y', 'z', 'num_cluster'])
            df.to_csv(f'outputs.json/cluster_{cluster}_actualizado.csv', index=False)

            # FORMATO .DUMP
            archivo_in = f'outputs.dump/key_area_{cluster}.dump'
            archivo_out = f'outputs.json/cluster_{cluster}_actualizado'
            encabezado = EncabezadoLammps(archivo_in, archivo_out)
            encabezado.copiar_encabezado()
        i=0
        for cluster in numero_cluster_critic:
            archivo_out = f'outputs.json/cluster_{cluster}_actualizado'
            pipeline = import_file(archivo_out)
            pipelineb = import_file(archivo_out)

            pipeline.modifiers.append(ExpressionSelectionModifier(expression=f"Cluster==3"))
            pipeline.modifiers.append(DeleteSelectedModifier())
            pipelineb.modifiers.append(ExpressionSelectionModifier(expression=f"Cluster!=3"))
            pipelineb.modifiers.append(DeleteSelectedModifier())
            pipeline.compute()
            pipelineb.compute()
            try:
                export_file(pipeline, f"outputs.dump/key_area_{archivo_out[21:23]}.{i}.dump", "lammps/dump",
                            columns=["Particle Identifier","Particle Type",  "Position.X", "Position.Y", "Position.Z"])
                pipeline.modifiers.clear()
                export_file(pipelineb, f"outputs.dump/key_area_{archivo_out[21:23]}.{i+1}.dump", "lammps/dump",
                            columns=["Particle Identifier","Particle Type",  "Position.X", "Position.Y", "Position.Z"])
                pipelineb.modifiers.clear()

                self.LISTA_ARCHIVOS_IMPORTANTES.append(f"outputs.dump/key_area_{archivo_out[21:23]}.{i}.dump")
                self.LISTA_ARCHIVOS_IMPORTANTES.append(f"outputs.dump/key_area_{archivo_out[21:23]}.{i+1}.dump")
                
                self.LISTA_ARCHIVOS_IMPORTANTES = list(set(self.LISTA_ARCHIVOS_IMPORTANTES))

                with open('outputs.json/lista_nombres_clusters.json', 'w') as archivo_json:
                    json.dump(self.LISTA_ARCHIVOS_IMPORTANTES, archivo_json, indent=4)

            except Exception as e:
                print(f"Error al exportar el archivo: {e}")

        print(self.LISTA_ARCHIVOS_IMPORTANTES)
        with open('outputs.json/lista_nombres_clusters.json', 'w') as archivo_json:
            json.dump(self.LISTA_ARCHIVOS_IMPORTANTES, archivo_json, indent=4)

        self.bClusteringAnalysis = True
        print("bClusteringAnalysis =", self.bClusteringAnalysis)



