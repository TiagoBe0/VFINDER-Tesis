import ovito
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, ConstructSurfaceModifier,DeleteSelectedModifier, VoronoiAnalysisModifier, ClusterAnalysisModifier
import os
import json
import random
import re
from va_input_params import LAYERS

class SingleVacancyProcessor:
    def __init__(self, layer):
        primer_elemento = LAYERS[0]
        self.layer = primer_elemento
        self.relax = primer_elemento['relax']
        self.cutoff_radius = primer_elemento['cutoff radius']
        self.radius = primer_elemento['radius']
        self.smoothing_level = primer_elemento['smoothing level']
        self.pipeline = None
        self.smoothing_level_training=primer_elemento['smoothing_level_training']
        self.sms_sv = []
        self.nb_sv = []

    @staticmethod
    def extraer_ids(archivo):
        with open(archivo, 'r') as f:
            contenido = f.read()
        ids = re.findall(r'^\s*([0-9]+)', contenido, re.MULTILINE)
        return [int(id_str) for id_str in ids]

    def run(self):
        self.pipeline = import_file(self.relax)
        ids = self.extraer_ids(self.relax)
        if not ids:
            print("No se encontraron IDs en el archivo.")
            return
        id_aleatorio = int(random.choice(ids))
        print(f"id eliminado: {id_aleatorio}")
        self.pipeline.modifiers.append(ExpressionSelectionModifier(expression=f'ParticleIdentifier=={id_aleatorio}'))
        self.pipeline.modifiers.append(DeleteSelectedModifier())
        self.pipeline.modifiers.append(VoronoiAnalysisModifier(compute_indices=True))
        #self.pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=self.cutoff_radius, cluster_coloring=True, unwrap_particles=True, sort_by_size=True))
        data = self.pipeline.compute()
        vecinos = data.particles.count
        try:
            export_file(self.pipeline, "single_vacancy_training.dump", "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Atomic Volume", "Cavity Radius", "Max Face Order"])
            self.pipeline.modifiers.clear()
        except Exception as e:
            print(f"Error al exportar el archivo: {e}")
        pipeline_f = import_file("single_vacancy_training.dump")
        pipeline_f.modifiers.append(ConstructSurfaceModifier(
            radius=self.radius,
            smoothing_level=self.smoothing_level_training,
            identify_regions=True,
            select_surface_particles=True
        ))
        data_f=pipeline_f.compute()
        self.sms_sv=data_f.attributes['ConstructSurfaceMesh.surface_area']
        datos = {'sms_sv': self.sms_sv, 'nb_sv': self.nb_sv}
        output_path = 'outputs.json/key_single_vacancy.json'
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, 'w') as f:
            json.dump(datos, f, indent=4)
