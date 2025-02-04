import random
import numpy as np
import json
import pandas as pd
from clustering_ml import ProcesadorArchivo
from LAMMPS_formater import EncabezadoLammps
import ovito
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier

class ClusterAnalyzer:
    def __init__(self, archivo_clusters, archivo_matrices, archivo_centros):
        self.archivo_clusters = archivo_clusters
        self.archivo_matrices = archivo_matrices
        self.archivo_centros = archivo_centros
        self.LISTA_ARCHIVOS_IMPORTANTES = []
        self.numero_cluster_critic = []
        self.centro_masas = self.cargar_centros_masas()
        self.num_clusters = self.cargar_num_clusters()
    
    def cargar_centros_masas(self):
        with open(self.archivo_centros, 'r') as archivo:
            datos = json.load(archivo)
        return [cluster[:3] for cluster in datos['clusters']]
    
    def cargar_num_clusters(self):
        with open(self.archivo_clusters, 'r') as f:
            clusters = json.load(f)
        return clusters['num_clusters']
    
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
                            int(valores[0]), int(valores[1]),
                            float(valores[2]), float(valores[3]), float(valores[4]),
                            float(valores[5])
                        ])
        return datos
    
    def calcular_centro_masa(self, datos):
        if not datos:
            return None
        sum_x, sum_y, sum_z = sum(d[0] for d in datos), sum(d[1] for d in datos), sum(d[2] for d in datos)
        num_atomos = len(datos)
        return sum_x / num_atomos, sum_y / num_atomos, sum_z / num_atomos
    
    def calcular_mayor_distancia(self, coordenadas):
        mayor_distancia, puntos_mayor_distancia = 0, None
        for i in range(len(coordenadas)):
            for j in range(i + 1, len(coordenadas)):
                distancia = np.linalg.norm(np.array(coordenadas[i]) - np.array(coordenadas[j]))
                if distancia > mayor_distancia:
                    mayor_distancia, puntos_mayor_distancia = distancia, (coordenadas[i], coordenadas[j])
        return mayor_distancia, puntos_mayor_distancia
    
    def analizar_clusters(self):
        for i in range(1, self.num_clusters + 1):
            with open(f'outputs.json/cluster_{i}.json', 'r') as archivo:
                datos = json.load(archivo)
            coordenadas = [fila[2:5] for fila in datos]
            datos_por_cluster = [fila[0:6] for fila in datos]
            distancias, varianza = self.calcular_distancias_y_varianza(datos_por_cluster, self.calcular_centro_masa(coordenadas))
            print(f'Varianza del cluster {i}:', varianza)
            if varianza > 1:
                self.numero_cluster_critic.append(i)
        print(f"Clusters críticos: {self.numero_cluster_critic}")

    def calcular_distancias_y_varianza(self, datos, centro_masas):
        if centro_masas is None:
            return [], 0
        distancias = [np.linalg.norm(np.array(atomo[2:5]) - np.array(centro_masas)) for atomo in datos]
        media = np.mean(distancias)
        varianza = np.var(distancias)
        return distancias, varianza
    
    def exportar_lista_archivos(self):
        with open('outputs.json/lista_nombres_clusters.json', 'w') as archivo_json:
            json.dump(self.LISTA_ARCHIVOS_IMPORTANTES, archivo_json, indent=4)


