Repositorio de Análisis de Vacancias y Clústeres en Simulaciones

Este repositorio contiene una colección de scripts en Python diseñados para analizar datos de simulaciones de materiales, centrándose en la detección y caracterización de vacancias y clústeres críticos. Las herramientas implementadas integran técnicas de procesamiento de datos, análisis geométrico y algoritmos de machine learning, y se apoyan en la biblioteca OVITO para el manejo y transformación de los datos de simulación.

Los scripts permiten configurar pipelines de procesamiento que aplican diversos modificadores de OVITO—como análisis de Voronoi, análisis de clústeres, selección y eliminación de partículas—para extraer características relevantes de las simulaciones. Por ejemplo, el módulo SingleVacancyProcessor se encarga de procesar archivos de entrada para identificar vacancias simples, extrayendo IDs de partículas y aplicando filtros y análisis que permiten generar datos de entrenamiento para modelos predictivos.

Por otro lado, el módulo ClusterProcessor está orientado al análisis avanzado de clústeres. Este componente calcula propiedades importantes de cada clúster, como el área superficial, el número de vecinos y las distancias máximas y mínimas respecto al centro de masa. Además, implementa estrategias de clustering (por ejemplo, utilizando KMeans y métodos personalizados) para fusionar o separar clústeres según criterios geométricos y de dispersión, facilitando la identificación de clústeres críticos en el sistema.

El repositorio también incluye scripts que entrenan modelos de machine learning, como regresión lineal y RandomForest, para predecir propiedades de las simulaciones (por ejemplo, el número de vacancias) a partir de las características geométricas extraídas. Estos modelos permiten establecer relaciones entre las propiedades observadas en las simulaciones y parámetros que pueden ser optimizados en estudios posteriores.

Para correr el programa es necesario colocar todos los archivos del repositorio en la misma ubicación donde se encuentre el archivo main.py de MultiSOM. En el archivo va_input_params se configuran los parámetros de entrada, los cuales incluyen la muestra relajada, la muestra defectuosa, el smoothing level, el radio de la sonda exploradora y el radio de corte para clústeres. Por ejemplo, se define un objeto LAYERS de la siguiente forma:

LAYERS = [{
    'relax' : 'tests.dump/d',
    'defect' : 'tests.dump/test_critical_1',
    'radius' : 2,
    'smoothing level' : 13,
    'cutoff radius' : 3,
    'iteraciones': 1   # Solo si está activado step_refactor
}]

Una vez ajustado este archivo de parámetros, se debe ejecutar va_main_run. Los resultados del procesamiento se almacenarán en las carpetas outputs.json, outputs.dump y outputs.vfinder.

La estructura del repositorio se basa en el uso de archivos de configuración que definen los parámetros de entrada, lo que facilita la adaptación de las herramientas a diferentes conjuntos de datos y configuraciones de simulación. Los resultados generados se exportan en formatos JSON y dump, permitiendo su integración en flujos de trabajo adicionales o su posterior visualización.

En resumen, este repositorio ofrece un conjunto de herramientas modulares y personalizables para el análisis avanzado de simulaciones de materiales. Su enfoque en la detección de vacancias y el análisis de clústeres lo convierte en una solución integral para investigadores que buscan comprender y optimizar las propiedades estructurales y defectos en materiales simulados.
