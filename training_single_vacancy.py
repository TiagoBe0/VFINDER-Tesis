import ovito
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier,CoordinationAnalysisModifier
from ovito.modifiers import ConstructSurfaceModifier,InvertSelectionModifier,ClusterAnalysisModifier,VoronoiAnalysisModifier
import os
import numpy as np
import json
import math
import random
from input_params import LAYERS
import re
#####PASO 1#####

def extraer_ids(archivo):
    # Abrir el archivo y leer su contenido
    with open(archivo, 'r') as f:
        contenido = f.read()

    # Utilizar expresiones regulares para extraer los IDs
    ids = re.findall(r'^\s*([0-9]+)', contenido, re.MULTILINE)

    # Convertir los IDs a enteros y devolverlos
    return [int(id) for id in ids]






# Acceder al primer elemento de la lista
primer_elemento = LAYERS[0]

# Acceder a los valores individuales
relax = primer_elemento['relax']

cutoff_radius = primer_elemento['cutoff radius']
radius = primer_elemento['radius']
smoothing_level = primer_elemento['smoothing level']
pipeline_0=import_file(relax)


ids = extraer_ids(relax)
sms_sv=[] #guardamos tres areas encontradas para la single vancancy
nb_sv=[] #guardamos el numero de vecinos

id_aleatorio = int(random.choice(ids))
pipeline_0.modifiers.append(ExpressionSelectionModifier(expression=f'ParticleIdentifier=={id_aleatorio}'))
pipeline_0.modifiers.append(DeleteSelectedModifier())
pipeline_0.modifiers.append(VoronoiAnalysisModifier(compute_indices=True))
pipeline_0.modifiers.append(ClusterAnalysisModifier(
    cutoff=cutoff_radius,cluster_coloring=True,unwrap_particles=True,sort_by_size=True))
data = pipeline_0.compute()
vecinos=data.particles.count  
    
try:
    export_file(pipeline_0, f"single_vacancy_training.dump", "lammps/dump",
                columns=["Particle Identifier","Particle Type",  "Position.X", "Position.Y", "Position.Z","Atomic Volume" ,"Cavity Radius", "Max Face Order"])
    pipeline_0.modifiers.clear()
        
except Exception as e:
    print(f"Error al exportar el archivo: {e}")

# Guardar los vectores en un archivo JSON
datos = {
    'sms_sv': sms_sv,
    'nb_sv': nb_sv
}

with open('outputs.json/key_single_vacancy.json', 'w') as f:
    json.dump(datos, f, indent=4)