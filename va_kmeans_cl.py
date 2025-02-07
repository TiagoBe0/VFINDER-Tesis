import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
class ProcesadorArchivo:
    def __init__(self, coordenadas):
        self.coordenadas = coordenadas

    def aplicar_kmeans(self, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, 
                        init='k-means++', 
                        n_init=10,  
                        max_iter=300, 
                        tol=1, 
                        random_state=42)  
        etiquetas = kmeans.fit_predict(self.coordenadas)
        return etiquetas
    def aplicar_kmeans_dot(self,cm_1,cm_2,cm_3):
        kmeans = KMeans(n_clusters=3, init=np.array([cm_1,
            cm_2,cm_3]))
        etiquetas = kmeans.fit_predict(self.coordenadas)
        return etiquetas


