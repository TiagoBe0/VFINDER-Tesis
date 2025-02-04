import ovito
from ovito.io import import_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier
from ovito.modifiers import ConstructSurfaceModifier,InvertSelectionModifier

import json
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

radius = primer_elemento['radius']
smoothing_level = primer_elemento['smoothing level']
pipeline_0=import_file(relax)


ids = extraer_ids(relax)
sms_sv=[] #guardamos tres areas encontradas para la single vancancy
nb_sv=[] #guardamos el numero de vecinos
for i in range(1,4):

    id_aleatorio = int(random.choice(ids))
    pipeline_0.modifiers.append(ExpressionSelectionModifier(expression=f'ParticleIdentifier=={id_aleatorio}'))
    pipeline_0.modifiers.append(DeleteSelectedModifier())
    pipeline_0.modifiers.append(ConstructSurfaceModifier(radius=radius,smoothing_level=smoothing_level,identify_regions=True,select_surface_particles=True))
    pipeline_0.modifiers.append(InvertSelectionModifier())
    pipeline_0.modifiers.append(DeleteSelectedModifier())
    data = pipeline_0.compute()
    vecinos=data.particles.count
    sm_single_vacancy=data.attributes['ConstructSurfaceMesh.surface_area']  
    sms_sv.append(sm_single_vacancy)
    nb_sv.append(vecinos)
    pipeline_0.modifiers.clear()

# Guardar los vectores en un archivo JSON
datos = {
    'sms_sv': sms_sv,
    'nb_sv': nb_sv
}

with open('outputs.json/key_single_vacancy.json', 'w') as f:
    json.dump(datos, f, indent=4)