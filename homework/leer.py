import json

# Definir la ruta del archivo
file_path = 'files/output/metrics.json'

# Leer el archivo
with open(file_path, 'r') as file:
    metrics_data = [json.loads(line) for line in file]

# Mostrar el contenido
for metric in metrics_data:
    print(metric)