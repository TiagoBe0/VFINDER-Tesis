import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

class ProcesadorArchivo:
    def __init__(self, nombre_archivo):
        self.nombre_archivo = nombre_archivo
        self.coordenadas = []

    

    def extraer_coordenadas_primer_atomo(self):
        with open(self.nombre_archivo, 'r') as f:
            for _ in range(8):  
                next(f)
            
            next(f)
            
            linea = next(f)
            _, _, x, y, z = linea.split()
            
            x, y, z = float(x), float(y), float(z)
            
            return (x, y, z)

    def extraer_coordenadas(self):
        self.coordenadas.append(self.extraer_coordenadas_primer_atomo())
        
        with open(self.nombre_archivo, 'r') as f:
            
            for _ in range(9):  
                next(f)
            
            
            for linea in f:
                _, _, x, y, z = linea.split()  
                self.coordenadas.append((float(x), float(y), float(z)))
        
        return self.coordenadas
    def aplicar_kmeans_dot(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init=np.array([[            0.9620519999999999,
            1.4430800000000001,
            0.4810260000000001],[ -1.33,-1.74,-0.5],[4.13,5.84,1.84]]))
        etiquetas = kmeans.fit_predict(self.coordenadas)
        return etiquetas

    def aplicar_kmeans(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, 
                        init='k-means++', 
                        n_init=2, 
                        max_iter=300, 
                        tol=1, 
                        random_state=None)
        etiquetas = kmeans.fit_predict(self.coordenadas)
        return etiquetas

    def agrega_etiquetas(self, etiquetas, archivo_salida):
        with open(self.nombre_archivo, 'r') as f_entrada, open(archivo_salida, 'w') as f_salida:
            coordenadas_leidas = False
            etiqueta_index = 1
            for linea in f_entrada:
                if linea.startswith('ITEM: ATOMS'):
                    f_salida.write(linea)
                    coordenadas_leidas = True
                elif coordenadas_leidas:
                    id, tipo, x, y, z = linea.split()
                    f_salida.write(f'{id} {tipo} {x} {y} {z} {etiquetas[etiqueta_index]}\n')
                    etiqueta_index += 1
                else:
                    f_salida.write(linea)

    def escribir_layer_c(self, archivo_salida,property):
        with open(archivo_salida, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.strip().startswith("ITEM: ATOMS"):
                lines[i] = line.replace("\n", "") + " " + property + "\n"
                break

        with open(archivo_salida, 'w') as f:
            f.writelines(lines)


