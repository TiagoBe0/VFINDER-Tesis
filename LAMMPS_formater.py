class EncabezadoLammps:
    def __init__(self, archivo_in, archivo_out):
        self.archivo_in = archivo_in
        self.archivo_out = archivo_out

    def copiar_encabezado(self):
        with open(self.archivo_in, 'r') as origen:
            encabezado = []
            for linea in origen:
                if linea.startswith('ITEM: ATOMS'):
                    encabezado.append(linea.replace('ITEM: ATOMS id type x y z', 'ITEM: ATOMS id type x y z Cluster'))
                    break
                else:
                    encabezado.append(linea)

        with open(self.archivo_out, 'r') as destino:
            lineas = destino.readlines()

        with open(self.archivo_out, 'w') as destino:
            destino.write(''.join(encabezado))
            destino.writelines(lineas)

