import os
import json
import math
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import ovito
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier, ConstructSurfaceModifier, VoronoiAnalysisModifier, InvertSelectionModifier, ClusterAnalysisModifier
from va_train_sv import SingleVacancyProcessor

from va_format import EncabezadoLammps
from va_input_params import LAYERS as LY
from input_params_som import LAYERS, PARAMS
from va_ClusterAnalysis import CriticalClusterAnalyzer
from va_CData import ClusterAnalyzer as ExportDateJSon
import subprocess
class DumpClusterSeparator:
    def __init__(self, input_file='outputs.dump/key_areas_clustering.dump', output_folder='outputs.json'):
        self.input_file = input_file
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def separate_clusters(self):
        datos = []
        leyendo_atomos = False
        with open(self.input_file, 'r') as f:
            for linea in f:
                if linea.strip() == "ITEM: ATOMS id type x y z Cluster":
                    leyendo_atomos = True
                    continue
                if leyendo_atomos:
                    if linea.startswith("ITEM:"):
                        break
                    columnas = linea.strip().split()
                    if len(columnas) == 6:
                        fila = [int(columnas[0]), int(columnas[1]), float(columnas[2]), float(columnas[3]), float(columnas[4]), int(columnas[5])]
                        datos.append(fila)
        if not datos:
            raise ValueError("No se encontraron átomos o 'ITEM: ATOMS id type x y z Cluster' en el archivo.")
        clusters_dict = {}
        for fila in datos:
            cluster_id = fila[-1]
            if cluster_id not in clusters_dict:
                clusters_dict[cluster_id] = []
            clusters_dict[cluster_id].append(fila)
        matrix_json_path = os.path.join(self.output_folder, "matrix_key_areas.json")
        with open(matrix_json_path, 'w') as m_out:
            json.dump(datos, m_out, indent=4)
        for cluster_id, atomos_cluster in clusters_dict.items():
            out_file = os.path.join(self.output_folder, f"cluster_{cluster_id}.json")
            with open(out_file, 'w') as c_out:
                json.dump(atomos_cluster, c_out, indent=4)
            #print(f"Generado: {out_file}")
        cluster_keys_path = os.path.join(self.output_folder, "cluster_keys.json")
        with open(cluster_keys_path, 'w') as ck_out:
            json.dump(list(clusters_dict.keys()), ck_out, indent=4)

class KeyAreasProcessor:
    def __init__(self, archivo):
        primer_elemento = LY[0]
        self.relax = primer_elemento['relax']
        self.defect = archivo
        self.bModifiersMethod=primer_elemento['bOvitoModifiers']
        self.radius = primer_elemento['radius']
        self.smoothing_level = primer_elemento['smoothing level']
        self.cutoff_radius = primer_elemento['cutoff radius']

        self.cutoff_clust = primer_elemento['cutoff cluster']
        self.LISTA_ARCHIVOS_IMPORTANTES = []

    def extraer_archivos_json(self, ruta):
        with open(ruta, 'r') as f:
            datos = json.load(f)
        return datos

    def calcular_distancias_y_varianza(self, datos, centro_masas):
        if centro_masas is None or (hasattr(centro_masas, '__len__') and len(centro_masas) == 0):
            return [], 0
        distancias = []
        for atomo in datos:
            x, y, z = atomo[0], atomo[1], atomo[2]
            distancia = np.sqrt((x - centro_masas[0])**2 + (y - centro_masas[1])**2 + (z - centro_masas[2])**2)
            distancias.append(distancia)
        if len(distancias) == 0:
            return [], 0
        media = sum(distancias) / len(distancias)
        varianza = sum((d - media)**2 for d in distancias) / len(distancias)
        return distancias, varianza

    def calcular_centro_masa(self, datos):
        if len(datos) == 0:
            return None
        sum_x = sum(d[0] for d in datos)
        sum_y = sum(d[1] for d in datos)
        sum_z = sum(d[2] for d in datos)
        num = len(datos)
        return (sum_x / num, sum_y / num, sum_z / num)

    def extraer_coordenadas(self, archivo):
        coordenadas = []
        with open(archivo, 'r') as f:
            for _ in range(9):
                next(f)
            for linea in f:
                id_, tipo, x, y, z = linea.split()
                coordenadas.append((float(x), float(y), float(z)))
        return coordenadas

    def main(self):
        pipeline = import_file(self.defect)
        pipeline.modifiers.append(VoronoiAnalysisModifier(compute_indices=True))
        try:
            export_file(pipeline, "key_areas_format.dump", "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Atomic Volume", "Cavity Radius", "Max Face Order"])
            pipeline.modifiers.clear()

        except Exception as e:
            print(f"Error al exportar el archivo: {e}")
        #print("postprossesing SOMs")
        if self.bModifiersMethod:
            print('OvitoModifiersMethod')
            pipeline_m = import_file(self.defect)
            pipeline_m.modifiers.append(ConstructSurfaceModifier(
                radius=self.radius,
                smoothing_level=self.smoothing_level,
                identify_regions=True,
                select_surface_particles=True
            ))   
            pipeline_m.modifiers.append(InvertSelectionModifier())
            pipeline_m.modifiers.append(DeleteSelectedModifier())
            pipeline_m.modifiers.append(ClusterAnalysisModifier(cutoff=self.cutoff_radius, cluster_coloring=True, unwrap_particles=True, sort_by_size=True))
            data = pipeline_m.compute()
            atm_ig = data.particles.count
            cluster_table = data.tables['clusters']
            cluster_sizes = cluster_table['Cluster Size'][...]
            #print("Cluster Size Table")
            #print(cluster_sizes)
            num_clusters = data.attributes["ClusterAnalysis.cluster_count"]
            datos_clusters = {"num_clusters": num_clusters}
            os.makedirs("outputs.json", exist_ok=True)
            with open("outputs.json/clusters.json", "w") as archivo:
                json.dump(datos_clusters, archivo, indent=4)
            print(f"Se identificaron {num_clusters} áreas claves en la muestra.")
            try:
                os.makedirs("outputs.dump", exist_ok=True)
                export_file(pipeline_m, "outputs.dump/key_areas.dump", "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"])
                export_file(pipeline_m, "outputs.dump/key_areas_clustering.dump", "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
                pipeline_som.modifiers.clear()
            except Exception as e:
                print(f"Error al exportar el archivo: {e}")
            pipeline_2 = import_file("outputs.dump/key_areas.dump")
            clusters = [f"Cluster=={i}" for i in range(1, num_clusters + 1)]           

        else:    
            import main_som
            pipeline_som = import_file("SOM_key_areas_format.dump")
            pipeline_som.modifiers.append(ConstructSurfaceModifier(radius=self.radius, smoothing_level=self.smoothing_level, identify_regions=True, select_surface_particles=True))
            pipeline_som.modifiers.append(ExpressionSelectionModifier(expression="layer_1==0"))
            pipeline_som.modifiers.append(DeleteSelectedModifier())
            pipeline_som.modifiers.append(ClusterAnalysisModifier(cutoff=self.cutoff_radius, cluster_coloring=True, unwrap_particles=True, sort_by_size=True))
            data = pipeline_som.compute()
            atm_ig = data.particles.count
            cluster_table = data.tables['clusters']
            cluster_sizes = cluster_table['Cluster Size'][...]
            #print("Cluster Size Table")
            #print(cluster_sizes)
            num_clusters = data.attributes["ClusterAnalysis.cluster_count"]
            datos_clusters = {"num_clusters": num_clusters}
            os.makedirs("outputs.json", exist_ok=True)
            with open("outputs.json/clusters.json", "w") as archivo:
                json.dump(datos_clusters, archivo, indent=4)
            print(f"Se identificaron {num_clusters} áreas claves en la muestra.")
            try:
                os.makedirs("outputs.dump", exist_ok=True)
                export_file(pipeline_som, "outputs.dump/key_areas.dump", "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"])
                export_file(pipeline_som, "outputs.dump/key_areas_clustering.dump", "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
                pipeline_som.modifiers.clear()
            except Exception as e:
                print(f"Error al exportar el archivo: {e}")
            pipeline_2 = import_file("outputs.dump/key_areas.dump")
            clusters = [f"Cluster=={i}" for i in range(1, num_clusters + 1)]
        for i, cluster_expr in enumerate(clusters, start=1):
            pipeline_2 = import_file("outputs.dump/key_areas.dump")
            pipeline_2.modifiers.append(ClusterAnalysisModifier(cutoff=self.cutoff_radius, cluster_coloring=True, unwrap_particles=True, sort_by_size=True))
            pipeline_2.modifiers.append(ExpressionSelectionModifier(expression=cluster_expr))
            pipeline_2.modifiers.append(InvertSelectionModifier())
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            try:
                export_file(pipeline_2, f"outputs.dump/key_area_{i}.dump", "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"])
                pipeline_2.modifiers.clear()
                self.LISTA_ARCHIVOS_IMPORTANTES.append(f"outputs.dump/key_area_{i}.dump")
            except Exception as e:
                print(f"Error al exportar el archivo: {e}")
        matriz = np.zeros((len(clusters), 7))
        for i, cluster_expr in enumerate(clusters, start=1):
            #print(f"#################### Critical Area {i} ###############")
            max_sm = 0
            j_max_sm = 0
            archivo_nombre = f"outputs.dump/key_area_{i}.dump"
            for j in range(round(self.radius), 7):
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
                data_3_n = pipeline_3.compute()
                cluster_table = data_3_n.tables['clusters']
                centros_de_masa = cluster_table['Center of Mass'][...]
                if centros_de_masa.size == 0:
                    print(f"Warning: No se encontró 'Center of Mass' en el cluster {i} para radio {j}. Se omite esta iteración.")
                    pipeline_3.modifiers.clear()
                    continue
                atm_cluster = cluster_table['Cluster Size'][0]
                centro_masa = centros_de_masa[0]
                vectores = []
                for coord in coordenadas:
                    x, y, z = coord
                    vector_cm_position = (x - centro_masa[0], y - centro_masa[1], z - centro_masa[2])
                    norma_vector = math.sqrt(vector_cm_position[0]**2 + vector_cm_position[1]**2 + vector_cm_position[2]**2)
                    vectores.append((vector_cm_position, norma_vector))
                if not vectores:
                    print(f"Warning: No se pudieron calcular vectores para el cluster {i} en radio {j}.")
                    pipeline_3.modifiers.clear()
                    continue
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
            datos_finales = {"num_clusters": num_clusters, "clusters": matriz_lista}
            os.makedirs("outputs.json", exist_ok=True)
            with open("outputs.json/key_areas_matrix.json", "w") as archivo:
                json.dump(datos_finales, archivo, indent=4)


if __name__ == "__main__":
    primer_archivo = LY[0]
    #print(primer_archivo['defect'])
    processor = KeyAreasProcessor(primer_archivo['defect'])
    processor.main()
    separator = DumpClusterSeparator(input_file='outputs.dump/key_areas_clustering.dump', output_folder='outputs.json')
    separator.separate_clusters()
    analyzer = CriticalClusterAnalyzer(primer_archivo['cutoff cluster'])
    analyzer.run()
    lista_archivos = processor.extraer_archivos_json('outputs.json/lista_nombres_clusters.json')
    #print("Archivos encontrados en el JSON:")
    train_sv = SingleVacancyProcessor( LY[0])
    train_sv.run()
    critic_files_cl = []
    for archivo in lista_archivos:
        #print(f"format.dump: {archivo}")
        coordenadas = processor.extraer_coordenadas(archivo)
        #print(archivo)
        #print(coordenadas)
        cm = processor.calcular_centro_masa(coordenadas)
        #print(f"centro de masa : {cm}")
        distancias, varianza = processor.calcular_distancias_y_varianza(np.array(coordenadas), cm)
        #print(f"distancias={distancias}")
        print(f"varianza={varianza}")
        if varianza > processor.cutoff_clust:
            pipeline = import_file(archivo)
            pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=processor.cutoff_radius, sort_by_size=True, compute_com=True))
            try:
                export_file(pipeline, f"{archivo}", "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
                pipeline.modifiers.clear()
                critic_files_cl.append(f"{archivo}")
            except Exception as e:
                print(f"Error al exportar el archivo: {e}")
        else:
            print("cls add")
            
    #print(f"critical files:{critic_files_cl}")

    from va_kmeans_iterator import ClusterProcessor as ClusterAnalizer
    with open("outputs.json/critic_files.json", "w") as f:
        json.dump(critic_files_cl, f, indent=4)
    if len(critic_files_cl) != 0:
        primer_elemento = LY[0]
        cut_radius = primer_elemento['cutoff radius']
        div_clusters=primer_archivo['divisions_of_cluster']
        cluster_processor = ClusterAnalizer("outputs.json/critic_files.json",cut_radius,primer_archivo['cutoff cluster'],div_clusters)
        cluster_processor.procesar_clusters()
    from va_pp_ml import ClusterProcessor
    primer_elemento = LY[0]
    cut_radius = primer_elemento['cutoff radius']
    smoothing_level = primer_elemento['smoothing level']
    cluster_processor = ClusterProcessor("outputs.json/lista_nombres_clusters.json", cut_radius, smoothing_level)
    cluster_processor.unificar_formato()
    cluster_processor.procesar_clusters()
    from va_train_nanopore import ClusterProcessor,ClusterFilter
    filtro = ClusterFilter("outputs.json/key_areas_matrix_FINAL.json", "outputs.json/key_areas_matrix_FINAL.json")
    filtro.run()
    processor = ClusterProcessor(LY)
    processor.load_clusters('outputs.json/key_areas_matrix_FINAL.json')
    #print("Surface Area:", processor.surface_area)
    #print("Vecinos:", processor.vecinos)
    #print("Menor Norma:", processor.menor_norma)
    #print("Mayores Norma:", processor.mayores_norma)
    #print("Coordenadas CM:", processor.coordenadas_cm)
    max_radius, idx, centro_masa_max = processor.get_max_cluster()
    #print("Max radius:", max_radius)
    #print("Índice del cluster máximo:", idx)
    #print("Centro de masa del cluster máximo:", centro_masa_max)
    processor.process_pipeline_ids("outputs.vfinder/ids.dump")
    #print("PUNTO DE CONTROOOOOOOL")
    processor.run_training("outputs.vfinder/ids.dump", "outputs.vfinder/training_cluster.json")
    from va_btree_predict import VacancyPredictor
    #features = primer_elemento['columns_train']  #de cero a tres!  [surface area, vecinos , minima distancia , maxima distancia]
    #predictor = VacancyPredictor('outputs.vfinder/training_cluster.json', features_to_use=features)
    #predictor.entrenar_modelo()
    #predictor.evaluar_modelo()
    #predictor.predecir_por_cluster('outputs.json/key_areas_matrix_FINAL.json', 'outputs.json/key_single_vacancy.json')
        # Ruta a los archivos de entrenamiento, clusters y single vacancy.
    training_file = "outputs.vfinder/training_cluster.json"
    clusters_file = 'outputs.json/key_areas_matrix_FINAL.json'
    single_vacancy_file = 'outputs.json/key_single_vacancy.json'

    predictor = VacancyPredictor(training_file="outputs.vfinder/training_cluster.json", features_to_use=primer_elemento['columns_train'])

    predictor.entrenar_modelo()
    predictor.evaluar_modelo()
    vector_predicciones = predictor.predecir_por_cluster(clusters_file, single_vacancy_file)
