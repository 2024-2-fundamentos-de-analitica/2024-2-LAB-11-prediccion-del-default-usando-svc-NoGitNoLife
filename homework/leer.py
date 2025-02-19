import pickle

# Ruta del archivo .pkl
file_path = 'files/grading/y_train.pkl'

# Cargar el archivo Pickle
with open(file_path, 'rb') as file:
    x_test = pickle.load(file)

# Ver los primeros datos cargados (si es adecuado según el tipo de datos)
print(x_test)